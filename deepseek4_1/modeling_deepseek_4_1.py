import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_deepseek_4_1 import DeepseekV41Config
from .engram import EngramLayout, NgramHashState

try:
    from . import kernel as _kernel
except Exception:
    _kernel = None

FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32
FP4_KV_BLOCK_SIZE = 16


@dataclass
class DeepseekV41CausalLMOutputWithPast(CausalLMOutputWithPast):
    """Standard causal-LM output plus the indexer distillation loss."""

    indexer_loss: torch.Tensor | None = None

# engram.py was written against the reference ModelArgs names; this view maps them to config names.
_ARGS_ALIASES = {
    "dim": "hidden_size",
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "moe_inter_dim": "moe_intermediate_size",
    "n_activated_experts": "num_experts_per_tok",
    "score_func": "scoring_func",
    "route_scale": "routed_scaling_factor",
    "rope_head_dim": "qk_rope_head_dim",
    "norm_eps": "rms_norm_eps",
    "window_size": "sliding_window",
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "candidate_source_layer": "candidate_source_layer_id",
    "engram_pad_id": "engram_pad_token_id",
}
_ARGS_DEFAULTS = {
    "max_batch_size": 4,
    "max_seq_len": 4096,
    "temperature": 1.0,
}


class _ArgsView:
    def __init__(self, config):
        object.__setattr__(self, "_config", config)

    def __getattr__(self, name):
        config = object.__getattribute__(self, "_config")
        target = _ARGS_ALIASES.get(name, name)
        if hasattr(config, target):
            return getattr(config, target)
        if name in _ARGS_DEFAULTS:
            return _ARGS_DEFAULTS[name]
        raise AttributeError(name)


def rope_settings(config):
    rope = getattr(config, "rope_scaling", None) or getattr(config, "rope_parameters", None) or {}
    return {
        "theta": getattr(config, "rope_theta", None) or rope.get("rope_theta", 10000.0),
        "original_seq_len": rope.get("original_max_position_embeddings", 0),
        "factor": rope.get("factor", 1),
        "beta_fast": rope.get("beta_fast", 32),
        "beta_slow": rope.get("beta_slow", 1),
    }


def _kernels_usable(t: torch.Tensor) -> bool:
    return _kernel is not None and t.is_cuda and not torch.is_grad_enabled()


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return y

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _ste(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and x.requires_grad:
        return _StraightThrough.apply(x, y)
    return y


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(v)) for positive float32 v, exact (same result as the kernel's bit trick)."""
    mantissa, exponent = torch.frexp(v)
    exponent = exponent - (mantissa == 0.5).to(exponent.dtype)
    return torch.ldexp(torch.ones_like(v), exponent)


@lru_cache(maxsize=8)
def _fp4_tables(device):
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device)
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    return mids, vals


def _fp4_round(v: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value, ties to the even code (matches hardware conversion)."""
    mids, vals = _fp4_tables(v.device)
    a = v.abs()
    lo = torch.bucketize(a, mids, right=False)
    hi = torch.bucketize(a, mids, right=True)
    idx = torch.where((hi != lo) & (lo % 2 == 1), hi, lo)
    return torch.copysign(vals[idx], v)


def fake_quant_fp8(x: torch.Tensor, block: int = FP8_BLOCK_SIZE) -> torch.Tensor:
    """Quantize-dequantize to FP8 E4M3 with power-of-two block scales, straight-through gradient."""
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2_ceil(amax * (1.0 / 448.0))
    y = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return _ste(x, y.flatten(-2).to(x.dtype))


def fake_quant_fp4(x: torch.Tensor, block: int = FP4_BLOCK_SIZE, e4m3_scale: bool = False) -> torch.Tensor:
    """Quantize-dequantize to FP4 E2M1 with E8M0 (default) or E4M3 block scales, straight-through gradient."""
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        amax = amax.clamp_min(6 * 2**-9)
        scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2**-126)
        scale = _pow2_ceil(amax * (1.0 / 6.0))
    y = _fp4_round((xf / scale).clamp(-6.0, 6.0)) * scale
    return _ste(x, y.flatten(-2).to(x.dtype))


@lru_cache(maxsize=8)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, device):
    """Rotary frequencies as complex exponentials, one row per position. YaRN when original_seq_len > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs).to(device)


def rotate_tail(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """Rotate the last rd channels of x over adjacent pairs, out of place. x is [b, s, d] or [b, s, h, d]."""
    head, tail = x[..., :-rd], x[..., -rd:]
    z = torch.view_as_complex(tail.float().contiguous().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if z.ndim == 3:
        freqs_cis = freqs_cis.reshape(1, z.size(1), z.size(-1))
    else:
        freqs_cis = freqs_cis.reshape(1, z.size(1), 1, z.size(-1))
    tail = torch.view_as_real(z * freqs_cis).flatten(-2).to(x.dtype)
    return torch.cat([head, tail], dim=-1)


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int, device):
    """Sliding-window ring slots each query attends to; -1 marks an empty slot."""
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat(
            [torch.arange(oldest, window_size, device=device), torch.arange(oldest, device=device)]
        )
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _sparse_attn_chunk(q, kv, sink, idx, scale):
    b, c, h, d = q.shape
    valid = idx >= 0
    safe = idx.clamp_min(0).long()
    batch = torch.arange(b, device=q.device).view(b, 1, 1)
    kv_sel = kv[batch, safe].float()
    scores = torch.einsum("bchd,bckd->bchk", q.float(), kv_sel) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    sink_col = sink.float().view(1, 1, h, 1).expand(b, c, h, 1)
    probs = torch.cat([scores, sink_col], dim=-1).softmax(dim=-1)[..., :-1]
    return torch.einsum("bchk,bckd->bchd", probs, kv_sel).to(q.dtype)


def sparse_attn(q, kv, attn_sink, topk_idxs, scale, chunk=128):
    """Attention over gathered KV positions with a learnable sink logit. q: [b, m, h, d], kv: [b, n, d],
    topk_idxs: [b, m, k] int32, -1 for empty slots. Uses the tilelang kernel for no-grad CUDA bf16 calls."""
    if _kernels_usable(q) and q.dtype == torch.bfloat16:
        return _kernel.sparse_attn(q, kv, attn_sink.float(), topk_idxs, scale)
    needs_grad = torch.is_grad_enabled() and (q.requires_grad or kv.requires_grad or attn_sink.requires_grad)
    outs = []
    for i in range(0, q.size(1), chunk):
        args = (q[:, i : i + chunk], kv, attn_sink, topk_idxs[:, i : i + chunk], scale)
        if needs_grad:
            outs.append(checkpoint(_sparse_attn_chunk, *args, use_reentrant=False))
        else:
            outs.append(_sparse_attn_chunk(*args))
    return torch.cat(outs, dim=1)


def _hc_split_sinkhorn_eager(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    if _kernels_usable(mixes):
        return _kernel.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    return _hc_split_sinkhorn_eager(mixes, hc_scale, hc_base, hc_mult, iters, eps)


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class SharedAttentionRuntime:
    """What attention layers hand down the stack. Each block works on a copy and returns it, so the
    flow is functional and safe under activation checkpointing."""

    __slots__ = (
        "compress_kv",
        "index_k",
        "topk_idxs",
        "candidates",
        "indexer_score",
        "indexer_valid",
        "indexer_loss",
        "collect_indexer_loss",
    )

    def __init__(
        self,
        compress_kv=None,
        index_k=None,
        topk_idxs=None,
        candidates=None,
        indexer_score=None,
        indexer_valid=None,
        indexer_loss=None,
        collect_indexer_loss=False,
    ):
        self.compress_kv = compress_kv
        self.index_k = index_k
        self.topk_idxs = topk_idxs
        self.candidates = candidates
        self.indexer_score = indexer_score
        self.indexer_valid = indexer_valid
        self.indexer_loss = indexer_loss
        self.collect_indexer_loss = collect_indexer_loss

    def copy(self):
        return SharedAttentionRuntime(
            self.compress_kv,
            self.index_k,
            self.topk_idxs,
            self.candidates,
            self.indexer_score,
            self.indexer_valid,
            self.indexer_loss,
            self.collect_indexer_loss,
        )


class _CacheModule(nn.Module):
    """Holds lazily allocated inference state, so nothing is created on the meta device at init."""

    _cache_names: tuple = ()

    def _get_cache(self, name, shape, device, dtype, fill=0.0):
        buf = getattr(self, name)
        if buf is None or buf.device != device or buf.dtype != dtype or tuple(buf.shape) != tuple(shape):
            buf = torch.full(shape, fill, dtype=dtype, device=device)
            setattr(self, name, buf)
        return buf

    def clear_cache(self):
        for name in self._cache_names:
            setattr(self, name, None)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class EngramEmbedding(nn.Module):
    """Dense n-gram table. In training, `sparse=True` makes autograd return a sparse gradient, which
    avoids materializing a full table-sized dense gradient. Sparse gradients require a sparse-capable
    optimizer (for example Adagrad, SparseAdam or SGD); AdamW does not support them."""

    def __init__(self, num_embeddings: int, dim: int, sparse: bool = False):
        super().__init__()
        if num_embeddings < 1 or dim < 1:
            raise ValueError("num_embeddings and dim must be >= 1")
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.sparse = sparse
        self.weight = nn.Parameter(torch.empty(num_embeddings, dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"engram indices must be integer, got {indices.dtype}")
        return F.embedding(indices, self.weight, sparse=self.sparse and self.training)

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.dim}, sparse={self.sparse}"


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream."""

    def __init__(self, config, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.clamp_value = 1e-6
        self.n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.embed = EngramEmbedding(
            layout.num_embeddings[self.layer_hash_index],
            layout.head_dim,
            sparse=getattr(config, "engram_sparse_grad", True),
        )
        self.wkv = nn.Linear(self.n_hash_cols * layout.head_dim, self.dim * (self.hc_mult + 1), bias=False)
        self.eps = config.rms_norm_eps
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]."""
        if hash_ids.size(-1) != self.n_hash_cols:
            raise ValueError(f"expected {self.n_hash_cols} hash columns, got {hash_ids.size(-1)}")
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()
        h, eps = x.float(), self.eps
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Compressor(_CacheModule):
    """Pools compress_ratio consecutive tokens into one KV latent with a learned softmax gate.
    Returns the latent before RoPE, or None while a group is still filling up."""

    _cache_names = ("kv_state", "score_state")

    def __init__(self, config, layer_id: int):
        super().__init__()
        ratio = config.compress_ratios[layer_id]
        self.compress_ratio = ratio
        self.head_dim = config.head_dim
        self.max_batch_size = config.max_batch_size
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.wkv = nn.Linear(
            config.hidden_size, self.head_dim, bias=False, dtype=torch.float32 if ratio > 1 else None
        )
        if ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
        self.kv_state = None
        self.score_state = None

    def forward(self, x: torch.Tensor, start_pos: int, use_cache: bool) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        if bsz > self.max_batch_size:
            raise ValueError(f"batch {bsz} > max_batch_size {self.max_batch_size}")
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:
            return self.norm(self.wkv(x))

        x32 = x.float()
        kv = F.linear(x32, self.wkv.weight.float())
        score = F.linear(x32, self.wgate.weight.float())
        if start_pos == 0:
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder and use_cache:
                assert bsz <= self.max_batch_size, f"batch {bsz} > max_batch_size {self.max_batch_size}"
                shape = (self.max_batch_size, ratio, self.head_dim)
                kv_state = self._get_cache("kv_state", shape, x.device, torch.float32, 0.0)
                score_state = self._get_cache("score_state", shape, x.device, torch.float32, -torch.inf)
                kv_state[:bsz, :remainder] = kv[:, cutoff:].detach()
                score_state[:bsz, :remainder] = score[:, cutoff:].detach()
            if seqlen < ratio:
                return None
            kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
            score = score[:, :cutoff].unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            assert use_cache, "decoding needs use_cache=True"
            shape = (self.max_batch_size, ratio, self.head_dim)
            kv_state = self._get_cache("kv_state", shape, x.device, torch.float32, 0.0)
            score_state = self._get_cache("score_state", shape, x.device, torch.float32, -torch.inf)
            slot = start_pos % ratio
            kv_state[:bsz, slot] = kv.squeeze(1)
            score_state[:bsz, slot] = score.squeeze(1)
            if (start_pos + 1) % ratio != 0:
                return None
            kv = (kv_state[:bsz] * score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        return self.norm(kv.to(dtype))


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the topk_blocks highest-scoring blocks per query."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(_CacheModule):
    """Keeps the index_topk best compressed positions per query.

    The top-k selection itself is discrete, so the language-model loss gives these parameters no
    gradient. With `train_indexer=False` (default) the whole indexer is frozen. With
    `train_indexer=True` it is trained by a distillation loss computed in Attention: the indexer's
    softmax over compressed positions is aligned with the main attention's distribution over the same
    positions. `indexer_detach_inputs=True` keeps that auxiliary loss from backpropagating into the
    backbone; the indexer's own projections still learn.
    """

    _cache_names = ("k_cache",)

    def __init__(self, config, layer_id: int):
        super().__init__()
        self.train_indexer = bool(getattr(config, "train_indexer", False))
        self.detach_inputs = bool(getattr(config, "indexer_detach_inputs", True))
        candidate_source = config.candidate_source_layer_id
        self.owns_k = layer_id in config.kv_source_layer_ids
        self.compress_ratio = config.compress_ratios[layer_id]
        if self.compress_ratio < 1:
            raise ValueError(f"indexer layer {layer_id} must have compress_ratios > 0")
        self.is_candidate_source = layer_id == candidate_source
        self.uses_candidates = 0 <= candidate_source < layer_id
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        if self.index_topk < 1:
            raise ValueError("index_topk must be >= 1")
        self.softmax_scale = self.index_head_dim**-0.5
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.fake_quant = getattr(config, "fake_quant", True)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        if self.owns_k:
            self.wk = nn.Linear(config.head_dim, self.index_head_dim, bias=False)
            self.k_norm = RMSNorm(self.index_head_dim, config.rms_norm_eps)
        self.k_cache = None
        if not self.train_indexer:
            # Frozen: the LM loss cannot train a discrete top-k selector. Keeping requires_grad=False
            # also removes these parameters from optimizers that filter on it.
            self.requires_grad_(False)

    def _quant(self, t):
        return fake_quant_fp4(t, FP4_BLOCK_SIZE) if self.fake_quant else t

    def forward(self, x, qr, latent, start_pos, offset, freqs_table, shared, use_cache):
        if self.train_indexer and torch.is_grad_enabled():
            return self._forward(x, qr, latent, start_pos, offset, freqs_table, shared, use_cache)
        with torch.no_grad():
            return self._forward(x, qr, latent, start_pos, offset, freqs_table, shared, use_cache)

    def _forward(self, x, qr, latent, start_pos, offset, freqs_table, shared, use_cache):
        """Returns (topk indices, raw scores [B, L, T], valid mask [L, T] or None).

        The raw scores are taken before candidate masking and before the -inf mask, because the
        distillation loss compares them with the main attention over all valid compressed positions.
        """
        bsz, seqlen, _ = x.size()
        if bsz > self.max_batch_size:
            raise ValueError(f"batch {bsz} > max_batch_size {self.max_batch_size}")
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
        detach = self.train_indexer and self.detach_inputs

        if self.owns_k:
            k = None
            if latent is not None:
                freqs = (
                    freqs_table[: seqlen - seqlen % ratio : ratio]
                    if start_pos == 0
                    else freqs_table[start_pos + 1 - ratio].unsqueeze(0)
                )
                k = self.k_norm(self.wk(latent.detach() if detach else latent))
                k = self._quant(rotate_tail(k, freqs, rd))
            if start_pos == 0:
                index_k = k if k is not None else x.new_empty(bsz, 0, self.index_head_dim)
                if use_cache and k is not None:
                    cache = self._get_cache(
                        "k_cache",
                        (self.max_batch_size, self.max_seq_len // ratio, self.index_head_dim),
                        x.device,
                        x.dtype,
                    )
                    cache[:bsz, : k.size(1)] = k
            else:
                cache = self._get_cache(
                    "k_cache",
                    (self.max_batch_size, self.max_seq_len // ratio, self.index_head_dim),
                    x.device,
                    x.dtype,
                )
                if k is not None:
                    cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
                    # Keep the new key in the graph: reading it back from the plain cache tensor
                    # would detach it and the distillation loss would not train wk/k_norm on
                    # decode steps.
                    index_k = torch.cat([cache[:bsz, : start_pos // ratio], k], dim=1)
                else:
                    index_k = cache[:bsz, : end_pos // ratio]
            shared.index_k = index_k

        q = self.wq_b(qr.detach() if detach else qr).unflatten(-1, (self.n_heads, self.index_head_dim))
        q = self._quant(rotate_tail(q, freqs_table[start_pos:end_pos], rd))

        index_k = shared.index_k[:bsz, : end_pos // ratio]
        if detach and not self.owns_k:
            # Layers 24/28/32/36 reuse layer 20's index keys. Keep their distillation loss from
            # reaching layer 20's wk/k_norm; layer 20 is still trained by its own loss.
            index_k = index_k.detach()
        weights = self.weights_proj(x.detach() if detach else x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (F.relu(index_score) * weights.unsqueeze(-1)).sum(dim=2)
        raw_score = index_score

        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            valid = torch.arange(seqlen // ratio, device=x.device) < compress_lens
            index_score = index_score.masked_fill(~valid.unsqueeze(0), -torch.inf)
        else:
            compress_lens = end_pos // ratio
            valid = None

        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~shared.candidates, -torch.inf)

        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int(), raw_score, valid


class Attention(_CacheModule):
    """Latent attention over a sliding window of raw KV plus, when compress_ratio > 0, index_topk
    compressed positions reaching further back. Q and the output projection are low-rank."""

    _cache_names = ("window_kv_cache", "compress_kv_cache")

    def __init__(self, config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.window_size = config.sliding_window
        self.compress_ratio = config.compress_ratios[layer_id]
        self.eps = config.rms_norm_eps
        self.max_batch_size = config.max_batch_size
        self.max_seq_len = config.max_seq_len
        self.fake_quant = getattr(config, "fake_quant", True)
        assert self.n_heads % self.n_groups == 0, "num_attention_heads must be divisible by o_groups"

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        self.is_kv_source = layer_id in config.kv_source_layer_ids
        self.is_index_source = layer_id in config.index_source_layer_ids
        if self.is_index_source and self.compress_ratio < 1:
            raise ValueError(f"index source layer {layer_id} must have compress_ratios > 0")
        self.compressor = Compressor(config, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(config, layer_id) if self.is_index_source else None

        rope = rope_settings(config)
        if self.compress_ratio:
            original_seq_len, theta = rope["original_seq_len"], config.compress_rope_theta
        else:
            original_seq_len, theta = 0, rope["theta"]
        self._rope_args = (
            self.rope_head_dim,
            self.max_seq_len,
            original_seq_len,
            float(theta),
            rope["factor"],
            rope["beta_fast"],
            rope["beta_slow"],
        )
        self.window_kv_cache = None
        self.compress_kv_cache = None

    def _freqs(self, device):
        return precompute_freqs_cis(*self._rope_args, device)

    def _window_kv(self, x, freqs_cis, start_pos, use_cache):
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        kv = rotate_tail(kv, freqs_cis, self.rope_head_dim)
        if self.fake_quant:
            kv = fake_quant_fp8(kv, FP8_BLOCK_SIZE)
        if start_pos == 0:
            if use_cache:
                cache = self._get_cache(
                    "window_kv_cache", (self.max_batch_size, win, self.head_dim), x.device, x.dtype
                )
                if seqlen <= win:
                    cache[:bsz, :seqlen] = kv.detach()
                else:
                    cutoff = seqlen % win
                    tail = kv.detach()[:, -win:]
                    cache[:bsz, cutoff:win], cache[:bsz, :cutoff] = tail.split([win - cutoff, cutoff], dim=1)
            window_kv = kv
        else:
            cache = self._get_cache(
                "window_kv_cache", (self.max_batch_size, win, self.head_dim), x.device, x.dtype
            )
            cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = cache[:bsz]
        return window_kv, get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device)

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_len, freqs_table, shared, use_cache):
        shared.indexer_score = None
        shared.indexer_valid = None
        if not self.is_index_source:
            return shared.topk_idxs

        bsz, seqlen, _ = x.size()
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            idxs, score, valid = self.indexer(x, qr, latent, start_pos, offset, freqs_table, shared, use_cache)
            shared.indexer_score = score
            shared.indexer_valid = valid
        shared.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset, freqs_table, shared, use_cache):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos, use_cache)
        idxs = self._compress_topk_idxs(
            x, qr, latent, start_pos, offset, compress_len, freqs_table, shared, use_cache
        )
        if self.is_kv_source:
            if latent is not None:
                freqs = (
                    freqs_table[: seqlen - seqlen % ratio : ratio]
                    if start_pos == 0
                    else freqs_table[start_pos + 1 - ratio].unsqueeze(0)
                )
                latent = rotate_tail(latent, freqs, self.rope_head_dim)
                if self.fake_quant:
                    latent = fake_quant_fp4(latent, FP4_KV_BLOCK_SIZE, e4m3_scale=True)
            shape = (self.max_batch_size, self.max_seq_len // ratio, self.head_dim)
            if start_pos == 0:
                full = latent if latent is not None else x.new_empty(bsz, 0, self.head_dim)
                if use_cache and latent is not None:
                    cache = self._get_cache("compress_kv_cache", shape, x.device, x.dtype)
                    cache[:bsz, : latent.size(1)] = latent.detach()
            else:
                cache = self._get_cache("compress_kv_cache", shape, x.device, x.dtype)
                if latent is not None:
                    cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
                full = cache[:bsz, :compress_len]
            shared.compress_kv = full
        return shared.compress_kv[:, :compress_len], idxs

    def _indexer_distillation_loss(self, q, compress_kv, shared):
        """Cross-entropy from the main-attention distribution to the indexer over compressed positions.

        This is KL(main || indexer) up to the teacher entropy (a constant), since the main-attention
        target is detached and only the indexer's log-probabilities carry gradient. Rows with no
        complete compressed position are ignored.
        """
        score = shared.indexer_score
        valid = shared.indexer_valid
        if score is None or score.size(-1) != compress_kv.size(1) or compress_kv.size(1) == 0:
            return None
        target_logits = torch.einsum(
            "bld,btd->blt", q.float().mean(dim=2), compress_kv.float()
        ) * self.softmax_scale
        if valid is not None:
            mask = ~valid.unsqueeze(0)
            row_has_valid = valid.any(dim=-1).view(1, valid.size(0), 1)
            target_logits = target_logits.masked_fill(mask, -torch.inf)
            # Softmax of an all -inf row is NaN. Those rows are dropped below, but keep the
            # intermediate finite so the valid rows in the same batch are not poisoned.
            target_logits = torch.where(row_has_valid, target_logits, torch.zeros_like(target_logits))
            score_for_loss = score.float().masked_fill(mask, -torch.inf)
            score_for_loss = torch.where(row_has_valid, score_for_loss, torch.zeros_like(score_for_loss))
        else:
            score_for_loss = score.float()
        target = torch.softmax(target_logits, dim=-1).detach()
        log_probs = torch.log_softmax(score_for_loss, dim=-1)
        if valid is not None:
            # `target` is exactly zero on invalid positions, but 0 * (-inf) is NaN, so zero the
            # log-probabilities there before the product. Those positions are dropped by the mask.
            log_probs = log_probs.masked_fill(mask, 0.0)
        loss = -(target * log_probs).sum(dim=-1)
        if valid is not None:
            # `loss` is [batch, seq], while `valid` is [seq, compressed]. The loss has already
            # summed over compressed positions, so keep rows with at least one valid position.
            loss = loss.masked_select(valid.any(dim=-1).unsqueeze(0))
        return loss.mean() if loss.numel() else None

    def forward(self, x: torch.Tensor, start_pos: int, shared: SharedAttentionRuntime, use_cache: bool):
        bsz, seqlen, _ = x.size()
        if bsz > self.max_batch_size:
            raise ValueError(f"batch {bsz} > max_batch_size {self.max_batch_size}")
        assert start_pos + seqlen <= self.max_seq_len, (
            f"sequence {start_pos + seqlen} > max_seq_len {self.max_seq_len}; raise config.max_seq_len"
        )
        shared.indexer_loss = None
        shared.indexer_score = None
        shared.indexer_valid = None
        freqs_table = self._freqs(x.device)
        freqs_cis = freqs_table[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = rotate_tail(q, freqs_cis, rd)

        kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos, use_cache)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(
                x, qr, start_pos, kv.size(1), freqs_table, shared, use_cache
            )
            if (
                shared.collect_indexer_loss
                and self.indexer is not None
                and self.indexer.train_indexer
                and self.training
                and torch.is_grad_enabled()
            ):
                shared.indexer_loss = self._indexer_distillation_loss(q, compress_kv, shared)
            shared.indexer_score = None
            shared.indexer_valid = None
            kv = torch.cat([kv, compress_kv], dim=1)
            topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        o = rotate_tail(o, freqs_cis, rd, inverse=True)

        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


class Gate(nn.Module):
    """MoE gating. The correction bias steers expert selection only; weights come from unbiased scores.
    The bias receives no gradient: balance it with DeepseekV41ForCausalLM.update_gate_bias."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.topk = config.num_experts_per_tok
        self.score_func = config.scoring_func
        self.gate_temp = getattr(config, "gate_temp", 1.0)
        self.norm_topk_prob = config.norm_topk_prob
        self.route_scale = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, self.dim))
        # Aux-loss-free correction: updated only by update_gate_bias(), never by autograd.
        self.bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32), requires_grad=False
        )
        self.load = None
        self.last_counts = None

    def note_load(self, counts: torch.Tensor):
        """Stage this forward's expert counts. The model commits them after the layer call, so a
        gradient-checkpoint recomputation cannot double-count them."""
        self.last_counts = counts.detach().float()

    def commit_load(self):
        if self.last_counts is None:
            return
        self.load = self.last_counts if self.load is None else self.load + self.last_counts
        self.last_counts = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = (scores + self.bias.float()).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """One SwiGLU FFN. Up branch clamped on both sides, gate branch only from above."""

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through."""

    def __init__(self, config):
        super().__init__()
        assert config.n_shared_experts == 1
        self.dim = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.gate = Gate(config)
        self.experts = nn.ModuleList(
            [
                Expert(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)
                for _ in range(self.n_routed_experts)
            ]
        )
        self.shared_experts = Expert(config.hidden_size, config.moe_intermediate_size, config.swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.reshape(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts_t = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
        if self.training:
            self.gate.note_load(counts_t)
        for i, count in enumerate(counts_t.tolist()):
            if count == 0:
                continue
            idx, top = torch.where(indices == i)
            y.index_add_(0, idx, self.experts[i](x[idx], weights[idx, top, None]).float())
        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)


class DeepseekV41Block(nn.Module):
    """Residual stream carried as hc_mult parallel copies (Hyper-Connections). The coefficients a
    sublayer computes are used by the next one, see forward."""

    def __init__(self, config, layer_id: int, engram_layout: EngramLayout | None = None):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = config.rms_norm_eps
        self.attn = Attention(config, layer_id)
        self.ffn = MoE(config)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(config, layer_id, engram_layout)
        self.attn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        fp32 = torch.float32
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=fp32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=fp32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=fp32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=fp32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=fp32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=fp32))

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        """x: [b,s,hc,d]. Returns pre / post / comb coefficients from one projection of the stream."""
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn.float()) * rsqrt
        return hc_split_sinkhorn(
            mixes, hc_scale.float(), hc_base.float(), self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
        )

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """[b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x, residual, post, comb):
        """x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]"""
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x, start_pos, pre_mix, shared_in, use_cache):
        shared = shared_in.copy()
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, shared, use_cache)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre, shared, shared.indexer_loss


def _ce_sum(hidden, weight, labels):
    logits = F.linear(hidden.float(), weight.float())
    return F.cross_entropy(logits, labels, ignore_index=-100, reduction="sum")


class DeepseekV41PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV41Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    main_input_name = "input_ids"
    _no_split_modules = ["DeepseekV41Block"]
    _is_stateful = True
    _keep_in_fp32_modules = [
        "hc_attn_fn",
        "hc_ffn_fn",
        "hc_attn_base",
        "hc_ffn_base",
        "hc_attn_scale",
        "hc_ffn_scale",
        "attn_sink",
        "wgate",
        "bias",
        "head",
    ]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, EngramEmbedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Gate):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            nn.init.zeros_(module.bias)
        elif isinstance(module, Attention):
            nn.init.zeros_(module.attn_sink)
        elif isinstance(module, Engram):
            nn.init.ones_(module.q_weight)
            nn.init.ones_(module.k_weight)
        elif isinstance(module, DeepseekV41Block):
            for name in ("hc_attn_fn", "hc_ffn_fn"):
                nn.init.normal_(getattr(module, name), mean=0.0, std=std)
            for name in ("hc_attn_base", "hc_ffn_base"):
                nn.init.zeros_(getattr(module, name))
            for name in ("hc_attn_scale", "hc_ffn_scale"):
                nn.init.ones_(getattr(module, name))


class DeepseekV41Model(DeepseekV41PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.hc_mult = config.hc_mult
        self.gradient_checkpointing = False
        self.engram_layout = EngramLayout.from_args(_ArgsView(config))
        self.engram_hash = None
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV41Block(config, layer_id, self.engram_layout) for layer_id in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.indexer_training = any(
            isinstance(module, Indexer) and module.train_indexer for module in self.modules()
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def set_tokenizer(self, tokenizer):
        """Builds the engram n-gram hash state. Call once after loading, on the final device."""
        if self.engram_layout is None:
            return
        with torch.device(self.embed.weight.device):
            self.engram_hash = NgramHashState(_ArgsView(self.config), self.engram_layout, tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
        return_indexer_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """input_ids: [b, s]. start_pos 0 processes the whole chunk; start_pos > 0 decodes one token and
        needs use_cache=True after a prefill with use_cache=True.

        With return_indexer_loss=True, returns (hidden, indexer_loss). The loss is only produced when
        train_indexer=True, the model is in training mode and the per-layer indexers were asked to
        collect it.
        """
        if start_pos > 0 and (input_ids.size(1) != 1 or not use_cache):
            raise ValueError("start_pos > 0 needs one token per step and use_cache=True")
        if input_ids.size(0) > self.config.max_batch_size:
            raise ValueError(f"batch {input_ids.size(0)} > max_batch_size {self.config.max_batch_size}")
        hashes = None
        if self.engram_layout is not None:
            if self.engram_hash is None:
                raise RuntimeError("engram needs the tokenizer: call model.set_tokenizer(tokenizer) once after loading")
            # NgramHashState runs under inference_mode. Clone outside it so the hash ids are normal
            # tensors and can be consumed by a grad-enabled embedding lookup.
            hashes = self.engram_hash(input_ids, start_pos, None).clone()
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        shared = SharedAttentionRuntime(collect_indexer_loss=return_indexer_loss)
        indexer_loss = None
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :])
            if checkpointing:
                h, pre_mix, shared, layer_indexer_loss = checkpoint(
                    layer, h, start_pos, pre_mix, shared, use_cache, use_reentrant=False
                )
            else:
                h, pre_mix, shared, layer_indexer_loss = layer(h, start_pos, pre_mix, shared, use_cache)
            if self.training:
                # Commit here, not inside MoE.forward: checkpoint recomputation calls the layer again
                # but not this loop, so the counts are added exactly once.
                layer.ffn.gate.commit_load()
            if return_indexer_loss and layer_indexer_loss is not None:
                indexer_loss = (
                    layer_indexer_loss if indexer_loss is None else indexer_loss + layer_indexer_loss
                )
        h = self.layers[-1].hc_pre(h, pre_mix)
        h = self.norm(h)
        if return_indexer_loss:
            return h, indexer_loss
        return h


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV41Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def set_tokenizer(self, tokenizer):
        self.model.set_tokenizer(tokenizer)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def reset_cache(self):
        for module in self.modules():
            if isinstance(module, _CacheModule):
                module.clear_cache()

    @torch.no_grad()
    def update_gate_bias(self, gamma: float = 1e-3):
        """Aux-loss-free balancing: push each expert bias against its load since the last call, then reset
        the counters. Call once per optimizer step; loads are all-reduced when torch.distributed is up.

        The bias is deliberately not an autograd parameter. Returns the number of gates updated."""
        updated = 0
        for module in self.modules():
            if not isinstance(module, Gate):
                continue
            module.last_counts = None
            if module.load is None:
                continue
            load = module.load
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(load)
            module.bias.requires_grad_(False)
            module.bias.add_(gamma * torch.sign(load.mean() - load).to(module.bias.dtype))
            module.load = None
            updated += 1
        return updated

    def _lm_loss(self, hidden, labels, num_items=None, chunk=2048):
        h = hidden[:, :-1].reshape(-1, hidden.size(-1))
        y = labels[:, 1:].reshape(-1).to(h.device)
        weight = self.head.weight
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, h.size(0), chunk):
            hi, yi = h[i : i + chunk], y[i : i + chunk]
            if torch.is_grad_enabled() and hi.requires_grad:
                total = total + checkpoint(_ce_sum, hi, weight, yi, use_reentrant=False)
            else:
                total = total + _ce_sum(hi, weight, yi)
        denom = num_items if num_items is not None else (y != -100).sum().clamp_min(1)
        return total / denom

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        if cache_position is not None and int(cache_position[0]) > 0:
            input_ids = input_ids[:, cache_position]
        model_inputs = {"input_ids": input_ids, "cache_position": cache_position, "use_cache": True}
        if "logits_to_keep" in kwargs:
            model_inputs["logits_to_keep"] = kwargs["logits_to_keep"]
        return model_inputs

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        return_indexer_loss: bool = False,
        **kwargs,
    ) -> DeepseekV41CausalLMOutputWithPast:
        """attention_mask is ignored: use right padding only, no packed documents. cache_position[0] is
        the start position (None means 0). With labels the loss is computed in chunks and no logits are
        returned. Labels are aligned with input_ids and shifted here; pass num_items_in_batch to
        normalize by a global token count.

        When train_indexer=True, the indexer distillation loss is added to the language-model loss with
        config.indexer_loss_weight and also returned as outputs.indexer_loss. Pass
        return_indexer_loss=True to obtain it without labels.
        """
        if input_ids is None:
            raise ValueError("input_ids is required")
        start_pos = 0 if cache_position is None else int(cache_position[0])
        if use_cache is None:
            use_cache = labels is None and not self.training
        need_indexer_loss = bool(
            self.model.indexer_training
            and self.training
            and torch.is_grad_enabled()
            and (labels is not None or return_indexer_loss)
        )
        model_output = self.model(
            input_ids, start_pos=start_pos, use_cache=use_cache, return_indexer_loss=need_indexer_loss
        )
        if need_indexer_loss:
            hidden, indexer_loss = model_output
        else:
            hidden, indexer_loss = model_output, None
        if labels is not None:
            loss = self._lm_loss(hidden, labels, kwargs.get("num_items_in_batch"))
            if indexer_loss is not None:
                loss = loss + self.config.indexer_loss_weight * indexer_loss
            return DeepseekV41CausalLMOutputWithPast(loss=loss, indexer_loss=indexer_loss)
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = F.linear(hidden[:, keep].float(), self.head.weight.float())
        return DeepseekV41CausalLMOutputWithPast(logits=logits, indexer_loss=indexer_loss)