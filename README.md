# KTRWS

Joint Khmer text recognition and word segmentation — [arXiv:2608.30213](https://arxiv.org/abs/2608.30213).

One CTC model reads Khmer text and, with `b = 1`, marks word boundaries with `U+200B`.

![the KTRWS framework](viz.png)

## Run

```sh
curl -LO https://github.com/seanghay/KTRWS/releases/download/v0.2.0/ktrws.pt
uv run python -m ktrws.infer line.png --sep "|"
```

2.49% CER and 86.5% segmentation F1 on 2,000 held-out lines.

## Train

```
data/vocab.txt                       one KCC token per line
data/{train,dev,test}/images/*.png
data/{train,dev,test}/labels.jsonl   {"image": …, "text": …, "segmented": …}
```

```sh
uv run python -m ktrws.train --data data --epochs 5 --device cuda --workers 8
```

Each image trains twice: plain text as the `b = 0` target, segmented as `b = 1`.
On Apple Silicon, `--device mps` needs `PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Citation

This is an independent implementation. Please cite the original paper:

> Kong, M., Buoy, R., Chenda, S., Taing, N., Iwamura, M., Kise, K.
> *Towards a Joint Khmer Text Recognition and Word Segmentation.*
> arXiv:2608.30213, 2026.

```bibtex
@misc{kong2026ktrws,
  title         = {Towards a Joint Khmer Text Recognition and Word Segmentation},
  author        = {Kong, Marry and Buoy, Rina and Chenda, Sovisal and
                   Taing, Nguonly and Iwamura, Masakazu and Kise, Koichi},
  year          = {2026},
  eprint        = {2608.30213},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.30213}
}
```

Word segmentation targets come from [khmercut](https://github.com/seanghay/khmercut).
