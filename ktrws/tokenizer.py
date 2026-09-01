import re

from khmercut.nn import tokenize as khmercut_tokenize

ZWSP = "\u200b"
BLANK = "∅"

BASE = "\u1780-\u17b3"
COENG = "\u17d2"
SIGN = "\u17b4-\u17d1\u17dd"

KCC_RE = re.compile(
  f"[{BASE}](?:{COENG}[{BASE}])*[{SIGN}]*|[\u17d4-\u17dc\u17e0-\u17e9\u17f0-\u17f9\u19e0-\u19ff]|."
)


def kcc_split(text):
  return KCC_RE.findall(text)


def insert_boundaries(text):
  words = ["".join(g) for g in khmercut_tokenize(text, deep=True)]
  return ZWSP.join(w for w in words if w.strip())


def encode_text(text, with_boundary):
  if with_boundary:
    text = insert_boundaries(text)
  else:
    text = text.replace(ZWSP, "")
  return kcc_split(text)


class Vocab:
  def __init__(self, tokens):
    tokens = [t for t in tokens if t not in (BLANK, ZWSP)]
    self.itos = [BLANK] + sorted(set(tokens)) + [ZWSP]
    self.stoi = {t: i for i, t in enumerate(self.itos)}

  def __len__(self):
    return len(self.itos)

  @classmethod
  def from_corpus(cls, texts):
    tokens = set()
    for t in texts:
      tokens.update(kcc_split(t))
    return cls(tokens)

  def encode(self, text, with_boundary=False):
    return [self.stoi[t] for t in encode_text(text, with_boundary) if t in self.stoi]

  def decode(self, ids, keep_boundary=True):
    out = []
    prev = None
    for i in ids:
      if i != prev and i != 0:
        out.append(self.itos[i])
      prev = i
    text = "".join(out)
    return text if keep_boundary else text.replace(ZWSP, "")

  def decode_targets(self, ids):
    return "".join(self.itos[i] for i in ids)

  def save(self, path):
    with open(path, "w", encoding="utf-8") as f:
      f.write("\n".join(self.itos[1:-1]))

  @classmethod
  def load(cls, path):
    with open(path, encoding="utf-8") as f:
      return cls(f.read().split("\n"))
