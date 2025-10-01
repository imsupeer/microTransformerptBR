import torch, torch.nn as nn, pathlib
from torch.utils.data import Dataset, DataLoader
from tokenizer_bpe import BPETokenizer
from transformer import TinyGPT


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=256):
        self.ids = []
        for t in texts:
            ids = tokenizer.encode(t)
            for i in range(0, max(0, len(ids) - seq_len - 1), seq_len):
                self.ids.append(ids[i : i + seq_len + 1])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = torch.tensor(self.ids[i][:-1], dtype=torch.long)
        y = torch.tensor(self.ids[i][1:], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = list(pathlib.Path("data/clean").glob("*.txt"))
    texts = [f.read_text(encoding="utf-8") for f in files]
    tok = BPETokenizer.load("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")
    ds = TextDataset(texts, tok, seq_len=256)
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
    m = TinyGPT(vocab_size=len(tok.token_to_id)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    steps = 2000
    it = iter(dl)
    m.train()
    for s in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(dl)
            xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        logits = m(xb)
        loss = loss_fn(logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 200 == 0:
            print(s, float(loss.item()))
    torch.save(m.state_dict(), "model/tinygpt.pt")
    print("done")
