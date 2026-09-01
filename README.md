# KTRWS

Joint Khmer text recognition and word segmentation — [arXiv:2608.30213](https://arxiv.org/abs/2608.30213).

One CTC model reads Khmer text and, when given `b = 1`, also marks word boundaries with `U+200B`.

![the KTRWS framework](viz.png)

## Data

```
data/vocab.txt                       one KCC token per line
data/{train,dev,test}/labels.jsonl
data/{train,dev,test}/images/*.png
```

```json
{"image": "images/000000.png", "text": "ព្រមជាមួយ", "segmented": "ព្រម​ជាមួយ"}
```

Each image is used twice: plain text as the `b = 0` target, segmented as `b = 1`.

## Weights

Trained 5 epochs on 16k synthetic textlines. On 2,000 held-out lines: **4.05% CER**
(4.01% at `b = 0`, 4.09% at `b = 1`) and **77.5%** segmentation F1.

```sh
curl -LO https://github.com/seanghay/KTRWS/releases/download/v0.1.0/ktrws.pt
```

Then run it — the checkpoint carries its own vocabulary and config:

```sh
uv run python -m ktrws.infer line.png --sep "|"
uv run python -m ktrws.infer line.png --plain      # b = 0, no boundaries
```

```python
from ktrws.infer import load, recognize

model, vocab = load("ktrws.pt")
print(recognize(model, vocab, ["line.png"])[0])   # words separated by U+200B
```

## Usage

```sh
uv run python -m ktrws.train --data data --epochs 5 --device cuda --workers 8
uv run pytest -q
```

`--gating` swaps FiLM fusion (Eq. 10) for the paper's gated alternative (Eq. 13-15).
On Apple Silicon use `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
