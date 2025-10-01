import pathlib
from tokenizer_bpe import BPETokenizer

files = list(pathlib.Path("data/clean").glob("*.txt"))
texts = [f.read_text(encoding="utf-8") for f in files]
bpe = BPETokenizer()
bpe.train(texts, vocab_size=12000, min_freq=2)
bpe.save("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")
print("ok")
