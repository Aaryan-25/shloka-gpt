<div align="center">

# 🪔 shloka-gpt

### *I built a small AI and taught it Sanskrit using only the Bhagavad Gita*

**धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः।**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-from%20scratch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Sanskrit](https://img.shields.io/badge/Language-Sanskrit-FF9933?style=for-the-badge)
![Verses](https://img.shields.io/badge/Training%20data-700%20verses-138808?style=for-the-badge)

**Built by Aaryan Ahuja**, Master of Hospital Administration, TISS Mumbai

</div>

---

## ⏱️ In one minute

**ChatGPT** and similar tools are built on a type of AI called a **GPT**. They learn language by reading enormous amounts of text and practising one simple skill: *guessing what comes next*.

I wanted to understand how that really works, so I **built a small GPT myself, from scratch**, and gave it just one book to learn from: the **Bhagavad Gita**, in its original Sanskrit.

The AI started knowing **nothing**: no words, no grammar, not even the alphabet. After reading the Gita's 700 verses many times over, it learned to **write new verses of its own** that follow the look and rhythm of the Gita.

> 💡 **The question behind it:** can a machine discover the rules of an ancient language just by reading, without anyone teaching it grammar?

---

## 🪔 Verses written by the AI

None of these verses exist in the Gita. The AI composed them itself, one letter at a time:

```text
अर्जुन उवाच
क्ष्यमुक्षाणं पात्मनसंशयोर्तगां संजगयम्।
सृहस्त्वभूता स्वभूतात्ममनः कौन्तेय श्रीप्रलयात्मुपः।।2.28।।

देवादिह्य वाद्योगेश च सर्वति मच्छन्ति चैव योगी त्।
सच्छामार्थं ते तद्वेषु योगस्वभूयः प्रकृतिष्ठताः।।2.29।।

सोऽहमावीनुप्ति मां तामां निव्यस्य कर्मिदः।
निर्माणि निराणीन्निर्ते जन्मन्ति मे न प्रलियम्।।2.67।।
```

Most of the words are invented, so the verses don't have a meaning. That's expected for a small AI trained on one book. What's remarkable is **what it figured out by itself**:

| What the AI learned on its own | Example from its verses |
|---|---|
| 📜 A verse has two lines and ends with `।।` and a verse number | `...प्रकृतिष्ठताः।।2.29।।` |
| 🗣️ Verses can begin with who is speaking | `अर्जुन उवाच`: "Arjuna said" |
| 📖 Real words from the Gita | `कौन्तेय` (Kaunteya), `योगी` (yogi), `कर्म` (karma), `सोऽहम्` (so'ham) |
| ✍️ How Sanskrit letters combine into syllables | joined letters like `ष्ठ`, `न्ति`, `त्म` |

Nobody told it any of these rules. It discovered them purely from patterns in the text.

---

## 🧠 How it works, in plain words

```mermaid
flowchart LR
    A["📜 Read the Gita<br/>700 verses"] --> B["🤔 Guess the<br/>next letter"]
    B --> C["✅ Check the answer<br/>and improve"]
    C --> B
    C --> D["🪔 Write new verses"]
```

1. **Reading:** the AI sees the Gita as a long string of Sanskrit letters, about 66,000 of them.
2. **Guessing:** again and again, it looks at some text and guesses the next letter.
3. **Improving:** each time it is wrong, it adjusts itself slightly. Over thousands of rounds its guesses get better.
4. **Writing:** once trained, it keeps guessing "the next letter" on its own, which produces brand-new verses.

This is the same basic idea behind ChatGPT, just millions of times smaller.

---

## 📈 The most interesting finding: learning vs memorising

With only one book to read, there was a real risk the AI would simply **learn the Gita by heart** instead of learning how Sanskrit works, like a student who memorises answers without understanding the subject.

To check, I kept some verses hidden from the AI during training and tested it on them (a lower score means better):

| Training run | Score on verses it studied | Score on verses it never saw | Result |
|---|---|---|---|
| Long (5000 rounds) | 0.05 | 4.54 | ❌ Memorised the book; got worse at new verses |
| Short (600 rounds) | 1.58 | **1.91** | ✅ Learned real patterns |

The longer run looked "perfect" on what it had studied, but it was only reciting. **Stopping at the right moment** gave an AI that actually learned. The training code now detects this automatically and keeps the best version.

> 🏥 **Why this matters beyond Sanskrit:** the same check, *does the AI work on cases it has never seen?*, is how AI tools in healthcare should be judged before anyone relies on them, whether they read X-rays or predict bed demand.

---

## 🛠️ For technical readers

<details>
<summary><b>Model details, code and how to run it</b></summary>

<br/>

| | |
|---|---|
| **Model** | GPT-style decoder-only Transformer, written from scratch in plain PyTorch (~100 lines in [`model.py`](model.py)) |
| **Size** | ~10.8M parameters: 6 layers, 6 attention heads, 384-dim embeddings, 256-character context |
| **Tokens** | Character-level: 73 unique Devanagari symbols |
| **Data** | Full Bhagavad Gita, ~66,000 characters, 90/10 train/validation split |
| **Training** | AdamW, lr 3e-4, batch 64, dropout 0.2; best-validation checkpointing |
| **Hardware** | Single NVIDIA GPU, ~15 minutes |

```bash
pip install torch
python train.py            # train (NVIDIA GPU recommended)
python train.py --small    # quick test on any laptop CPU
python sample.py           # compose new verses → samples.txt
```

```
shloka-gpt/
├── model.py     ← the Transformer, built from scratch
├── train.py     ← teaches the model on the Gita
├── sample.py    ← asks the model to compose new verses
└── data/
    └── gita.txt ← all 700 verses in Devanagari
```

</details>

---

## 🗺️ What's next

- [x] Build a GPT from scratch
- [x] Train it on the full Bhagavad Gita
- [x] Detect memorisation and keep the best version automatically
- [ ] Track *when* during training the verse rhythm appears
- [ ] Teach it whole syllables instead of single letters
- [ ] Train on more classical texts: Upanishads, Ramayana

---

## 🙏 Credits

- Gita text from the public-domain [gita/gita](https://github.com/gita/gita) dataset
- Model design inspired by Andrej Karpathy's educational GPT implementations

<div align="center">

**कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ॥**
*"Your right is to the work alone, never to its fruits." (Gita 2.47)*

</div>
