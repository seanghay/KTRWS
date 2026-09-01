# KTRWS

Joint Khmer text recognition and word segmentation, per *Towards a Joint Khmer Text Recognition and Word Segmentation* (arXiv:2608.30213).

A single CTC model recognizes Khmer text and, when conditioned on `b = 1`, emits `U+200B` word-boundary tokens.

![the KTRWS framework](viz.png)

The KTRWS framework (Figure 2 of the paper, Kong et al.). A binary flag `b` is embedded and
projected into the FiLM parameters that modulate the fused visual-temporal features, so the same
CTC decoder emits Khmer text with (`b = 1`) or without (`b = 0`) word boundaries.

## Layout

- `ktrws/data.py` — textline dataset yielding both `b=0` and `b=1` targets per image
- `ktrws/tokenizer.py` — Khmer character cluster (KCC) tokenizer, vocab, boundary insertion via `khmercut.nn` (`deep=True`)
- `ktrws/model.py` — ResNet CNN → Transformer encoder → height pooling → MAFS (router + adapters + FiLM from the boundary projector) → CTC head
- `ktrws/metrics.py` — CER (boundary-excluded) and segmentation F1 (offset-based, word-multiset fallback)
- `ktrws/train.py` — CTC training, cyclic LR, greedy decode

## Data

Training expects a directory of splits, each with a `labels.jsonl` and an `images/` folder,
plus a shared `vocab.txt` (one KCC token per line):

```
data/vocab.txt
data/{train,dev,test}/labels.jsonl
data/{train,dev,test}/images/*.png
```

Each `labels.jsonl` row carries the plain text and the same text segmented with `U+200B`:

```json
{"image": "images/000000.png", "text": "\u1796\u17d2\u179a\u1798...", "segmented": "\u1796\u17d2\u179a\u1798\u200b..."}
```

`LineDataset` yields each image twice, once as a `b = 0` target (plain) and once as `b = 1`
(segmented), so a single model learns both behaviours.

## Usage

```sh
uv run python -m ktrws.train --data data --epochs 5 --device mps
uv run pytest -q
```

`--gating` swaps the FiLM boundary fusion (Eq. 10) for the paper's gated alternative (Eq. 13-15).

On Apple Silicon set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
