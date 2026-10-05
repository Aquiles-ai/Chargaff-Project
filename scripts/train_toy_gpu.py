"""Toy Model training for Evo2 (dummy DNA data).

Usage:
    python scripts/train_toy_gpu.py

What it does:
    - Builds a small Evo2ForCausalLM from scratch (default ~few M params).
    - Moves it to CUDA when available, optional AMP + gradient checkpointing.
    - Trains on synthetic DNA-like token ids with a plain PyTorch loop.
    - Prints loss/perplexity and saves a checkpoint via save_pretrained.
"""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evo2 import Evo2Config, Evo2ForCausalLM

# Global config
STEPS = 100
BATCH_SIZE = 4
SEQ_LEN = 256
LR = 3e-4
HIDDEN_SIZE = 256
DEVICE = "auto"  # "auto" | "cuda" | "cpu"
USE_AMP = False
AMP_DTYPE = "bf16"  # "bf16" | "fp16"
USE_GRAD_CHECKPOINT = False
GRAD_ACCUM = 1
SAVE_DIR = "checkpoints/evo2-toy-gpu"
FULL_1B = False  # True trains the full ~1.1B default config
SEED = 0

# ACGT ids in Aquiles-ai/Chargaff-Tokenizer (see tokenizer/test_tokenizer.py).
ACGT_IDS = [32, 34, 38, 51]


def build_config() -> Evo2Config:
    if FULL_1B:
        # Reference evo2-1b-8k defaults. Needs ~40GB+ for training.
        return Evo2Config(
            vocab_size=512,
            pad_token_id=257,
            bos_token_id=256,
            eos_token_id=256,
        )
    # toy model: 8 layers covering HCS/HCM/HCL/attention twice.
    return Evo2Config(
        vocab_size=512,
        hidden_size=HIDDEN_SIZE,
        num_layers=8,
        num_hidden_layers=8,
        attn_layer_idxs=[3, 7],
        hcl_layer_idxs=[2, 6],
        hcm_layer_idxs=[1, 5],
        hcs_layer_idxs=[0, 4],
        num_attention_heads=8,
        proj_groups=1,
        hcm_filter_length=32,
        hcs_filter_length=7,
        hcl_filter_groups=HIDDEN_SIZE,
        hcm_filter_groups=32,
        hcs_filter_groups=32,
        short_filter_length=3,
        inner_mlp_size=HIDDEN_SIZE * 4,
        max_position_embeddings=1024,
        use_cache=False,
        initializer_range=0.02,
        pad_token_id=257,
        bos_token_id=256,
        eos_token_id=256,
    )


def dummy_dna_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    # 90% ACGT ids, 10% uniform noise: closer to DNA than pure uniform.
    acgt = torch.tensor(ACGT_IDS, device=device)
    ids = acgt[torch.randint(0, len(ACGT_IDS), (batch_size, seq_len), device=device)]
    noise_mask = torch.rand(batch_size, seq_len, device=device) < 0.1
    ids[noise_mask] = torch.randint(0, vocab_size, (noise_mask.sum(),), device=device)
    return ids.long()


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def resolve_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def main() -> None:
    torch.manual_seed(SEED)
    device = resolve_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    config = build_config()
    model = Evo2ForCausalLM(config)
    if USE_GRAD_CHECKPOINT and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    print(f"params: {count_params(model) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    use_amp = USE_AMP and device.type == "cuda"
    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and amp_dtype == torch.float16) else None

    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(GRAD_ACCUM):
            input_ids = dummy_dna_batch(BATCH_SIZE, SEQ_LEN, config.vocab_size, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                loss = out.loss / GRAD_ACCUM
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if step == 1 or step % 10 == 0 or step == STEPS:
            ppl = math.exp(min(loss_accum, 20.0))
            print(f"step {step}/{STEPS} loss={loss_accum:.4f} ppl={ppl:.1f}")

    model.eval()
    with torch.no_grad():
        probe = dummy_dna_batch(1, min(SEQ_LEN, 64), config.vocab_size, device)
        logits = model(input_ids=probe, use_cache=False).logits
    assert logits.shape == (1, probe.shape[1], config.vocab_size)
    print(f"inference: {tuple(logits.shape)}")

    save_dir = ROOT / SAVE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    config.save_pretrained(str(save_dir))
    print(f"saved to {save_dir}")


if __name__ == "__main__":
    main()
