# 🪔 shloka-gpt

**A tiny GPT-style language model trained from scratch on the Bhagavad Gita, to study how an AI picks up Sanskrit metre and word patterns.**

No pre-trained model, no shortcuts. The model starts knowing nothing, not even what a letter is, and learns Sanskrit purely by reading all 700 verses of the Gita, one Devanagari character at a time.

---

## Why

Most AI models are trained on English. Sanskrit is different: its words fuse together through *sandhi*, and its verses follow strict metres like the *anuṣṭubh* (four quarters of eight syllables). This project asks a simple question:

> Can a small neural network, given only the raw text, discover these rules on its own?

## How it works

| Step | What happens |
|---|---|
| 1. Read | The full Gita (~66,000 characters, 73 unique Devanagari symbols) is the only training data |
| 2. Learn | A 6-layer Transformer (~10M parameters) repeatedly guesses the next character and corrects itself |
| 3. Compose | The trained model writes brand-new verses, character by character |

The code is deliberately small and readable: the whole model is in [`model.py`](model.py) (~100 lines), written in plain PyTorch so every step can be followed.

```
model.py     the Transformer, built from scratch
train.py     teaches the model on the Gita
sample.py    asks the model to compose new verses
data/        the Bhagavad Gita in Devanagari
```

## Run it

```bash
pip install torch
python train.py            # full training (NVIDIA GPU recommended)
python train.py --small    # quick test on any laptop CPU
python sample.py           # generate new verses
```

## What to look for in the output

The generated verses are not meaningful Sanskrit, and they aren't meant to be. The interesting part is *what structure appears anyway*:

- valid Devanagari syllables and conjunct consonants
- verse-ending marks (`।` and `।।`) in the right places
- line lengths drifting towards the 8-syllable *anuṣṭubh* rhythm
- speaker lines like `श्रीभगवानुवाच` ("the Lord said")

## Roadmap

- [ ] Compare outputs from early vs late training to see *when* metre emerges
- [ ] Syllable-level tokens instead of characters
- [ ] Train on more classical texts (Upanishads, Ramayana)

## Credits

- Verse text from the public-domain [gita/gita](https://github.com/gita/gita) dataset
- Model design inspired by Andrej Karpathy's educational GPT implementations
