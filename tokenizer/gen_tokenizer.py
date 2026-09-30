"""
We generate the tokenizer for the DNA prediction model.

Usage:
    python gen_tokenizer.py
"""
from tokenizers import Tokenizer, AddedToken, decoders, pre_tokenizers
from tokenizers.models import BPE

OUT_PATH = "tokenizer.json"
HF_DIR = "./tokenizer"
VOCAB_SIZE = 512
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<eos>"
MODEL_MAX_LENGTH = 1048576
PADDING_SIDE = "right"
ADD_PREFIX_SPACE = False
USE_REGEX = False


def main():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    assert len(alphabet) == 256, len(alphabet)
    vocab = {tok: i for i, tok in enumerate(alphabet)}  # 0..255

    vocab[EOS_TOKEN] = 256
    vocab[PAD_TOKEN] = 257
    for i in range(258, VOCAB_SIZE):
        vocab[f"<unused_{i}>"] = i
    assert len(vocab) == VOCAB_SIZE

    tok = Tokenizer(BPE(vocab=vocab, merges=[], unk_token=None))

    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=ADD_PREFIX_SPACE, use_regex=USE_REGEX)
    tok.decoder = decoders.ByteLevel()

    tok.add_special_tokens([
        AddedToken(EOS_TOKEN, special=True, normalized=False),
        AddedToken(PAD_TOKEN, special=True, normalized=False),
    ])

    for s in ["hello", "café ñ €", "你好世界", "🎉🚀💩", "ACGT", "ACGTN"]:
        enc = tok.encode(s)
        dec = tok.decode(enc.ids)
        print(f"{s!r} -> {enc.ids} roundtrip OK={dec == s}")
    print("ACGT ids:", tok.encode("ACGT").ids)
    print("<eos>ACGT<pad>:", tok.encode("<eos>ACGT<pad>").ids)

    tok.save(OUT_PATH, pretty=True)
    print(f"guardado en {OUT_PATH}, vocab_size={tok.get_vocab_size()}")
    tok2 = Tokenizer.from_file(OUT_PATH)
    print("reload:", tok2.encode("hola 🌍").tokens, "eos/pad:", tok2.token_to_id(EOS_TOKEN), tok2.token_to_id(PAD_TOKEN))

    from transformers import PreTrainedTokenizerFast

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok2,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        model_max_length=MODEL_MAX_LENGTH,
        padding_side=PADDING_SIDE,
    )
    fast.save_pretrained(HF_DIR)
    print(f"Guardado en {HF_DIR}")


if __name__ == "__main__":
    main()
