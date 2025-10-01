import torch, numpy as np
from transformer import TinyGPT
from tokenizer_bpe import BPETokenizer

ckpt = torch.load("model/tinygpt.pt", map_location="cpu")
tok = BPETokenizer.load("data/tokenizer/vocab.json", "data/tokenizer/merges.txt")
m = TinyGPT(vocab_size=len(tok.token_to_id))
m.load_state_dict(ckpt)
m.eval()
q = {}
scales = {}
with torch.no_grad():
    for name, w in m.state_dict().items():
        if w.dtype == torch.float32 and (w.ndim == 2 or w.ndim == 1):
            maxv = w.abs().max().item() + 1e-8
            scale = maxv / 127.0
            wi8 = (w / scale).round().clamp(-127, 127).to(torch.int8).cpu().numpy()
            q[name] = wi8
            scales[name] = np.array([scale], dtype=np.float32)
        else:
            q[name] = w.cpu().numpy()
np.savez(
    "web/angular/public/weights.npz",
    **q,
    **{k + "__scale": v for k, v in scales.items()}
)
print("ok")
