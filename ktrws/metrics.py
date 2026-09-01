from collections import Counter

from ktrws.tokenizer import ZWSP, kcc_split


def cer(pred, gold):
  pred, gold = kcc_split(pred), kcc_split(gold)
  d = list(range(len(gold) + 1))
  for i, p in enumerate(pred, 1):
    prev, d[0] = d[0], i
    for j, g in enumerate(gold, 1):
      prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (p != g))
  return d[-1] / max(1, len(gold))


def words(text):
  return [w for w in text.split(ZWSP) if w]


def offsets(text):
  cum, out = 0, []
  for w in words(text)[:-1]:
    cum += len(kcc_split(w))
    out.append(cum)
  return set(out)


def seg_counts(pred, gold):
  if pred.replace(ZWSP, "") == gold.replace(ZWSP, ""):
    bp, bg = offsets(pred), offsets(gold)
    return len(bp & bg), len(bp - bg), len(bg - bp)
  cp, cg = Counter(words(pred)), Counter(words(gold))
  tp = sum((cp & cg).values())
  return tp, sum(cp.values()) - tp, sum(cg.values()) - tp


def f1(counts):
  tp, fp, fn = counts
  return 2 * tp / max(1, 2 * tp + fp + fn)


def boundary_positions(ids, blank_id, boundary_id, image_width, downsample=4):
  positions = []
  prev = None
  for i, token in enumerate(ids):
    if token == boundary_id and token != prev:
      positions.append(min(image_width, i * downsample))
    prev = token
  return positions
