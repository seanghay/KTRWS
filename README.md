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

## Usage

```sh
uv run python -m ktrws.train --data data --epochs 5 --device cuda --workers 8
uv run pytest -q
```

`--gating` swaps FiLM fusion (Eq. 10) for the paper's gated alternative (Eq. 13-15).
On Apple Silicon use `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
