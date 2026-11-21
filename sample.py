"""
Ask the trained model to compose new verses.

    python sample.py
    python sample.py --start "धर्म" --length 600 --temperature 0.8
"""

import argparse
import sys
import torch
from model import ShlokaGPT

p = argparse.ArgumentParser()
p.add_argument("--start", default="\n")
p.add_argument("--length", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load("shloka_gpt.pt", map_location=device)
chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}

model = ShlokaGPT(len(chars), **ckpt["cfg"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

start = [stoi[c] for c in args.start if c in stoi] or [0]
idx = torch.tensor([start], dtype=torch.long, device=device)
out = model.generate(idx, args.length, args.temperature)[0].tolist()
text = "".join(chars[i] for i in out)
with open("samples.txt", "w", encoding="utf-8") as f:
    f.write(text)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
except Exception:
    pass
print("\nsaved to samples.txt")
