import argparse

import torch
from PIL import Image

from ktrws.data import resize, to_tensor
from ktrws.model import KTRWS
from ktrws.tokenizer import ZWSP, Vocab


def load(weights, device="cpu"):
  ck = torch.load(weights, map_location=device)
  vocab = Vocab(ck["vocab"])
  model = KTRWS(len(vocab), **ck.get("config", {}))
  model.load_state_dict(ck["model"])
  return model.to(device).eval(), vocab


@torch.no_grad()
def recognize(model, vocab, paths, boundaries=True, device="cpu", batch_size=16):
  out = []
  for i in range(0, len(paths), batch_size):
    chunk = paths[i : i + batch_size]
    x = torch.stack([to_tensor(resize(Image.open(p).convert("RGB"))) for p in chunk]).to(device)
    flags = torch.full((len(chunk),), int(boundaries), dtype=torch.long, device=device)
    for row in model(x, flags).argmax(-1).cpu():
      out.append(vocab.decode(row.tolist(), keep_boundary=boundaries))
  return out


def main():
  p = argparse.ArgumentParser(description="Recognize Khmer textline images")
  p.add_argument("images", nargs="+")
  p.add_argument("--weights", default="ktrws.pt")
  p.add_argument("--device", default="cpu")
  p.add_argument("--plain", action="store_true", help="b=0: no word boundaries")
  p.add_argument("--sep", default=ZWSP, help="what to print between words")
  args = p.parse_args()

  model, vocab = load(args.weights, args.device)
  texts = recognize(model, vocab, args.images, not args.plain, args.device)
  for path, text in zip(args.images, texts, strict=True):
    print(f"{path}\t{text.replace(ZWSP, args.sep)}")


if __name__ == "__main__":
  main()
