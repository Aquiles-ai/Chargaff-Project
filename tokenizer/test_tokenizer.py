from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Aquiles-ai/Chargaff-Tokenizer")
print(tok.encode("ACGT"))          # [32, 34, 38, 51]
print(tok.decode([32, 34, 38, 51]))  # "ACGT"
print(tok(["ACGT", "ACGTN"], padding=True))
# {'input_ids': [[32, 34, 38, 51, 257], [32, 34, 38, 51, 45]]}
