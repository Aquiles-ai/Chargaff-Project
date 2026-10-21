"""Tiny CPU smoke tests for the text-only port.

Run from the repository root with:

    python -m deepseek4_1.testmodel

The test is deliberately small: it exercises the real forward/backward path, the indexer freeze,
the gate-bias update, the cache path and a minimal engram table. It is not a numerical parity test.
"""

import torch

try:
    from .configuration_deepseek_4_1 import DeepseekV41Config
    from .modeling_deepseek_4_1 import DeepseekV41ForCausalLM, DeepseekV41CausalLMOutputWithPast
except ImportError:  # allow `python testmodel.py` as well
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deepseek4_1.configuration_deepseek_4_1 import DeepseekV41Config
    from deepseek4_1.modeling_deepseek_4_1 import DeepseekV41ForCausalLM, DeepseekV41CausalLMOutputWithPast


def _tiny_text_config(**overrides):
    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        qk_rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        hidden_act="silu",
        swiglu_limit=10.0,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_cache=True,
        max_position_embeddings=128,
        max_seq_len=32,
        max_batch_size=2,
        rope_theta=10000,
        rope_scaling={"rope_type": "default"},
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        sliding_window=4,
        compress_ratios=[0, 2, 2, 1],
        compress_rope_theta=160000,
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 2, 3],
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
        candidate_source_layer_id=3,
        candidate_topk_blocks=8,
        candidate_block_size=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1e-6,
        engram_layer_ids=[],
        engram_num_embeddings=[],
        engram_max_ngram_size=1,
        engram_vocab_size=0,
        engram_n_heads=0,
        engram_head_dim=0,
        engram_pad_token_id=0,
        engram_compressed_vocab_size=0,
        engram_sparse_grad=True,
        # Keep the smoke test on plain bf16/fp32 matmuls; fake fp8/fp4 is exercised by the
        # real model configuration, not by this CPU plumbing test.
        fake_quant=False,
    )
    kwargs.update(overrides)
    return DeepseekV41Config(**kwargs)


def _indexers(model):
    return [module for module in model.modules() if module.__class__.__name__ == "Indexer"]


def test_text_only_forward_backward():
    torch.manual_seed(0)
    config = _tiny_text_config()
    model = DeepseekV41ForCausalLM(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()
    labels[:, :2] = -100

    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert isinstance(out, DeepseekV41CausalLMOutputWithPast)
    assert out.indexer_loss is None
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    out.loss.backward()

    assert model.model.embed.weight.grad is not None
    assert model.head.weight.grad is not None
    gate = model.model.layers[0].ffn.gate
    assert gate.weight.grad is not None
    assert gate.bias.grad is None
    assert gate.bias.requires_grad is False
    assert gate.load is not None

    indexers = _indexers(model)
    assert indexers, "the tiny config should contain indexer layers"
    for indexer in indexers:
        assert all(not parameter.requires_grad for parameter in indexer.parameters())
        assert all(parameter.grad is None for parameter in indexer.parameters())

    # The bias update must push an overloaded expert down and an underloaded one up.
    gate.load = torch.tensor([8.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    before = gate.bias.detach().clone()
    updated = model.update_gate_bias(gamma=1e-2)
    assert updated >= 1
    assert gate.bias[0].item() < before[0].item()
    assert gate.bias[1].item() > before[1].item()
    assert gate.load is None

    # Cached prefill + one-token decode.
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        prefill = model(input_ids=input_ids[:, :6], use_cache=True)
        assert prefill.logits.shape == (2, 6, config.vocab_size)
        step = model(
            input_ids=input_ids[:, 6:7],
            cache_position=torch.tensor([6]),
            use_cache=True,
        )
        assert step.logits.shape == (2, 1, config.vocab_size)
    model.reset_cache()


def test_gate_load_not_double_counted_with_checkpointing():
    torch.manual_seed(0)
    config = _tiny_text_config()
    model = DeepseekV41ForCausalLM(config)
    model.train()
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()

    model.update_gate_bias(0.0)  # clear any counters
    model.gradient_checkpointing_enable()
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    out.loss.backward()

    gates = [module for module in model.modules() if module.__class__.__name__ == "Gate"]
    total = sum(gate.load.sum() for gate in gates if gate.load is not None)
    expected = input_ids.numel() * config.num_experts_per_tok * config.num_hidden_layers
    assert total.item() == expected, (
        "gradient-checkpoint recomputation double-counted gate load: "
        f"{total.item()} != {expected}"
    )


def test_indexer_training_loss():
    torch.manual_seed(0)
    config = _tiny_text_config(train_indexer=True, indexer_loss_weight=0.5)
    model = DeepseekV41ForCausalLM(config)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()
    labels[:, :2] = -100

    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.indexer_loss is not None, "train_indexer=True must expose the auxiliary loss"
    assert out.indexer_loss.requires_grad
    assert torch.isfinite(out.indexer_loss).item()
    assert torch.isfinite(out.loss).item()
    out.loss.backward()

    indexers = _indexers(model)
    assert len(indexers) == len(config.index_source_layer_ids)
    for indexer in indexers:
        assert indexer.wq_b.weight.requires_grad
        assert indexer.wq_b.weight.grad is not None
        assert indexer.weights_proj.weight.grad is not None
        if indexer.owns_k:
            assert indexer.wk.weight.grad is not None
            assert indexer.k_norm.weight.grad is not None
        else:
            # Layers that reuse a previous layer's index keys must not own wk/k_norm.
            assert not hasattr(indexer, "wk")
            assert not hasattr(indexer, "k_norm")

    # The language-model path still trains the backbone and the gate.
    assert model.model.embed.weight.grad is not None
    assert model.model.layers[0].ffn.gate.weight.grad is not None
    assert model.model.layers[0].ffn.gate.bias.grad is None


class _FakeBackend:
    def decode(self, ids, skip_special_tokens=False):
        return f"t{ids[0]}"

    def id_to_token(self, token_id):
        return f"t{token_id}"


class _FakeTokenizer:
    backend_tokenizer = _FakeBackend()

    def __len__(self):
        return 16


def test_engram_forward_backward_sparse():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        print("engram smoke test skipped: tokenizers is not installed")
        return

    torch.manual_seed(0)
    config = _tiny_text_config(
        vocab_size=16,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=1,
        max_position_embeddings=32,
        max_seq_len=16,
        n_routed_experts=2,
        num_experts_per_tok=1,
        sliding_window=4,
        compress_ratios=[0, 0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        index_n_heads=1,
        index_head_dim=32,
        index_topk=1,
        candidate_source_layer_id=-1,
        engram_layer_ids=[0],
        engram_num_embeddings=[256],
        engram_max_ngram_size=2,
        engram_vocab_size=100,
        engram_n_heads=1,
        engram_head_dim=8,
        engram_pad_token_id=0,
        engram_compressed_vocab_size=16,
        engram_sparse_grad=True,
    )
    model = DeepseekV41ForCausalLM(config)
    model.set_tokenizer(_FakeTokenizer())
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert torch.isfinite(out.loss).item()
    out.loss.backward()

    engram = model.model.layers[0].engram
    assert engram is not None
    grad = engram.embed.weight.grad
    assert grad is not None
    assert grad.is_sparse, "engram_sparse_grad=True must produce a sparse embedding gradient"

    # SGD accepts sparse gradients; AdamW intentionally does not.
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    test_text_only_forward_backward()
    test_indexer_training_loss()
    test_gate_load_not_double_counted_with_checkpointing()
    test_engram_forward_backward_sparse()
    print("ok: text-only smoke tests passed")
