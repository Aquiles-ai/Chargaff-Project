from transformers.configuration_utils import PretrainedConfig

# Based from https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json
# Text-only model. MTP/DSpark and vision are intentionally unsupported: their keys are dropped
# instead of being carried around as dead configuration.

_UNSUPPORTED_REFERENCE_KEYS = (
    "num_nextn_predict_layers",
    "dspark_block_size",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "dspark_num_experts_per_tok",
    "vision_config",
    "image_token_id",
    "quantization_config",
)


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Flatten the released checkpoint's nested `text_config` before building the config.

        The reference config keeps the language model under `text_config` and has top-level
        `vision_config`/`quantization_config` keys. This text-only port takes the text values and drops
        the rest, so `from_pretrained` does not silently fall back to the class defaults.
        """
        config_dict = dict(config_dict)
        text_config = config_dict.pop("text_config", None)
        if text_config is not None:
            if not isinstance(text_config, dict):
                raise TypeError("text_config must be a dict when loading the reference config")
            # Top-level tokenizer/architecture keys win over the nested text values.
            config_dict = {**text_config, **config_dict}
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=5120,
        moe_intermediate_size=2304,
        num_hidden_layers=40,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        hidden_act="silu",
        swiglu_limit=10.0,
        rms_norm_eps=1e-20,
        attention_bias=False,
        attention_dropout=0.0,
        initializer_range=0.02,
        use_cache=True,
        max_position_embeddings=1048576,
        max_seq_len=4096,
        max_batch_size=4,
        fake_quant=True,
        rope_theta=10000,
        rope_scaling=None,
        n_routed_experts=384,
        n_shared_experts=1,
        num_experts_per_tok=6,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        sliding_window=128,
        compress_ratios=None,
        compress_rope_theta=160000,
        kv_source_layer_ids=None,
        index_source_layer_ids=None,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-06,
        engram_layer_ids=None,
        engram_num_embeddings=None,
        engram_max_ngram_size=4,
        engram_vocab_size=16000000,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=99092,
        engram_sparse_grad=True,
        train_indexer=False,
        indexer_loss_weight=0.1,
        indexer_detach_inputs=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if "text_config" in kwargs:
            raise ValueError(
                "text_config must be flattened by DeepseekV41Config.from_dict/from_pretrained, "
                "not passed directly to the constructor"
            )
        # Keep the class model_type instead of letting the reference config shadow it.
        kwargs.pop("model_type", None)
        # The reference checkpoint config also contains MTP/DSpark and vision entries. They are not
        # part of this implementation, so do not let PretrainedConfig keep them as attributes.
        for key in _UNSUPPORTED_REFERENCE_KEYS:
            kwargs.pop(key, None)

        if rope_scaling is None:
            rope_scaling = {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            }
        if compress_ratios is None:
            # The released 40-layer model uses this exact pattern. Other depths default to pure
            # sliding-window attention; a caller that wants compression must provide its own
            # ratios plus matching kv/index source layers.
            compress_ratios = (
                [0] * 2 + [2] * 18 + [1] * 20
                if num_hidden_layers == 40
                else [0] * num_hidden_layers
            )
        compress_ratios = list(compress_ratios)
        if len(compress_ratios) < num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(compress_ratios)} entries for {num_hidden_layers} layers"
            )
        if len(compress_ratios) > num_hidden_layers:
            # Reference configs append one entry per MTP layer. Keep only the main model.
            compress_ratios = compress_ratios[:num_hidden_layers]
        if any(ratio < 0 for ratio in compress_ratios):
            raise ValueError("compress_ratios must be >= 0")

        if kv_source_layer_ids is None:
            kv_source_layer_ids = [2, 8, 14, 20]
        if index_source_layer_ids is None:
            index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
        if engram_layer_ids is None:
            engram_layer_ids = [1, 14]
        if engram_num_embeddings is None:
            # The released checkpoint has exactly these two engram tables. An explicitly empty layer
            # list disables engram; any other layout must provide its own table sizes so a small test
            # model cannot accidentally allocate the released ~384M-row tables.
            if list(engram_layer_ids) == [1, 14]:
                engram_num_embeddings = [384006168, 384016682]
            elif not engram_layer_ids:
                engram_num_embeddings = []
            else:
                raise ValueError("provide engram_num_embeddings for a custom engram_layer_ids layout")

        kv_source_layer_ids = list(kv_source_layer_ids)
        index_source_layer_ids = list(index_source_layer_ids)
        engram_layer_ids = list(engram_layer_ids)
        engram_num_embeddings = list(engram_num_embeddings)

        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        for name, layer_ids in (
            ("kv_source_layer_ids", kv_source_layer_ids),
            ("index_source_layer_ids", index_source_layer_ids),
            ("engram_layer_ids", engram_layer_ids),
        ):
            bad = [layer_id for layer_id in layer_ids if not 0 <= layer_id < num_hidden_layers]
            if bad:
                raise ValueError(f"{name} contains out-of-range layer ids: {bad}")
        if len(set(kv_source_layer_ids)) != len(kv_source_layer_ids):
            raise ValueError("kv_source_layer_ids contains duplicates")
        if len(set(index_source_layer_ids)) != len(index_source_layer_ids):
            raise ValueError("index_source_layer_ids contains duplicates")
        if len(set(engram_layer_ids)) != len(engram_layer_ids):
            raise ValueError("engram_layer_ids contains duplicates")

        for layer_id in kv_source_layer_ids:
            if compress_ratios[layer_id] <= 0:
                raise ValueError(f"kv source layer {layer_id} must have compress_ratios > 0")
        for layer_id in index_source_layer_ids:
            if compress_ratios[layer_id] <= 0:
                raise ValueError(f"index source layer {layer_id} must have compress_ratios > 0")

        if candidate_source_layer_id >= num_hidden_layers:
            raise ValueError("candidate_source_layer_id is out of range")
        if candidate_source_layer_id >= 0 and candidate_source_layer_id not in index_source_layer_ids:
            raise ValueError("candidate_source_layer_id must be an index source layer")
        if indexer_loss_weight < 0:
            raise ValueError("indexer_loss_weight must be >= 0")

        # A compressed layer reuses the nearest earlier (or its own) KV source with the same ratio.
        # This is exactly how the released layers 24/28/32/36 reuse layer 20's compressed KV/index K.
        for layer_id, ratio in enumerate(compress_ratios):
            if ratio > 0 and not any(
                source <= layer_id and source in kv_source_layer_ids and compress_ratios[source] == ratio
                for source in kv_source_layer_ids
            ):
                raise ValueError(
                    f"layer {layer_id} needs a kv source at or before it with compress_ratio {ratio}"
                )
            if ratio > 0 and not any(
                source <= layer_id and source in index_source_layer_ids and compress_ratios[source] == ratio
                for source in index_source_layer_ids
            ):
                raise ValueError(
                    f"layer {layer_id} needs an index source at or before it with compress_ratio {ratio}"
                )
        for layer_id in index_source_layer_ids:
            ratio = compress_ratios[layer_id]
            if not any(
                source <= layer_id
                and source in index_source_layer_ids
                and source in kv_source_layer_ids
                and compress_ratios[source] == ratio
                for source in index_source_layer_ids
            ):
                raise ValueError(
                    f"index source layer {layer_id} needs an earlier index source that owns K "
                    f"with compress_ratio {ratio}"
                )

        if len(engram_layer_ids) != len(engram_num_embeddings):
            raise ValueError(
                "engram_layer_ids and engram_num_embeddings must have the same length: "
                f"{len(engram_layer_ids)} != {len(engram_num_embeddings)}"
            )
        if any(rows <= 0 for rows in engram_num_embeddings):
            raise ValueError("engram_num_embeddings entries must be > 0")
        if engram_layer_ids:
            if engram_max_ngram_size < 2:
                raise ValueError("engram_max_ngram_size must be >= 2 when engram is enabled")
            if engram_n_heads < 1 or engram_head_dim < 1:
                raise ValueError("engram_n_heads and engram_head_dim must be >= 1")
            if engram_vocab_size < 2:
                raise ValueError("engram_vocab_size must be >= 2 when engram is enabled")
            if engram_compressed_vocab_size < 1:
                raise ValueError("engram_compressed_vocab_size must be >= 1 when engram is enabled")
            if not 0 <= engram_pad_token_id < vocab_size:
                raise ValueError("engram_pad_token_id is outside the vocabulary")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.fake_quant = fake_quant
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        self.compress_ratios = compress_ratios
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = kv_source_layer_ids
        self.index_source_layer_ids = index_source_layer_ids
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.engram_layer_ids = engram_layer_ids
        self.engram_num_embeddings = engram_num_embeddings
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.engram_sparse_grad = engram_sparse_grad
        self.train_indexer = train_indexer
        self.indexer_loss_weight = indexer_loss_weight
        self.indexer_detach_inputs = indexer_detach_inputs

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
