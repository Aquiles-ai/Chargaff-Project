# Chargaff-Project

Project to train DNA models from genomic sequences.

## Why Chargaff?

Named for Erwin Chargaff (1905-2002), Austrian chemist at Columbia University.

In the late 1940s he used paper chromatography to measure DNA bases across species and found his rules: A equals T and G equals C, with purines equal to pyrimidines. He also showed composition varies by species and disproved the old tetranucleotide hypothesis.

Those ratios gave Watson and Crick key evidence for base pairing and the double helix. This project keeps the same focus: DNA sequence composition as the basis for modeling.

## Layout

- `tokenizer/`: byte-level BPE tokenizer, vocab 512, no training merges
- `tokenizer/gen_tokenizer.py`: generates `tokenizer.json`
- `tokenizer/test_tokenizer.py`: quick encode and padding check

Hub: `Aquiles-ai/Chargaff-Tokenizer`
