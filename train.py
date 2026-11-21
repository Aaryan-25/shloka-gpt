"""
Train ShlokaGPT on the Bhagavad Gita.

    python train.py            # full run (GPU recommended, a few minutes)
    python train.py --small    # quick CPU run to check everything works
"""

import argparse
import json
import torch
from model import ShlokaGPT

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/gita.txt")
p.add_argument("--small", action="store_true", help="tiny model for a quick CPU test")
p.add_argument("--steps", type=int, default=None)
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.small:
    cfg = dict(block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.1)
    batch_size, lr, steps = 32, 1e-3, args.steps or 500
else:
    cfg = dict(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2)
    batch_size, lr, steps = 64, 3e-4, args.steps or 1500

# ---- data: every distinct character becomes a token -------------------------
text = open(args.data, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"{len(text):,} characters, {len(chars)} unique symbols, device={device}")


def get_batch(src):
    ix = torch.randint(len(src) - cfg["block_size"] - 1, (batch_size,))
    x = torch.stack([src[i:i + cfg["block_size"]] for i in ix])
    y = torch.stack([src[i + 1:i + cfg["block_size"] + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, iters=50):
    model.eval()
    out = {}
    for name, src in [("train", train_data), ("val", val_data)]:
        losses = torch.stack([model(*get_batch(src))[1] for _ in range(iters)])
        out[name] = losses.mean().item()
    model.train()
    return out


# ---- train -------------------------------------------------------------------
model = ShlokaGPT(len(chars), **cfg).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
opt = torch.optim.AdamW(model.parameters(), lr=lr)

best_val = float("inf")
for step in range(steps + 1):
    if step % max(steps // 20, 1) == 0:
        l = estimate_loss(model)
        note = ""
        # keep the version that does best on verses it has NOT studied;
        # after that point the model starts memorising instead of learning
        if l["val"] < best_val:
            best_val = l["val"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "chars": chars}, "shloka_gpt.pt")
            note = "  <- best so far, saved"
        print(f"step {step:5d} | train loss {l['train']:.3f} | val loss {l['val']:.3f}{note}")
    xb, yb = get_batch(train_data)
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

print(f"done. best val loss {best_val:.3f}, model in shloka_gpt.pt")
