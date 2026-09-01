import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from ktrws.tokenizer import kcc_split

IMG_H = 32
IMG_W = 320


def resize(img, height=IMG_H, max_width=IMG_W):
  w = max(1, min(max_width, round(img.width * height / img.height)))
  return img.resize((w, height), Image.BILINEAR)


def render(text, font, height=IMG_H, max_width=IMG_W, pad=4):
  bbox = font.getbbox(text)
  w, h = bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2
  img = Image.new("RGB", (max(w, 8), max(h, 8)), (255, 255, 255))
  ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=(0, 0, 0))
  return resize(img, height, max_width)


def to_tensor(img, width=IMG_W):
  canvas = Image.new("RGB", (width, IMG_H), (255, 255, 255))
  canvas.paste(img, (0, 0))
  x = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8).float().div_(255.0)
  return x.view(IMG_H, width, 3).permute(2, 0, 1).sub_(0.5).div_(0.5)


class LineDataset(Dataset):
  def __init__(self, root, vocab, width=IMG_W):
    self.root = Path(root)
    self.vocab = vocab
    self.width = width
    lines = (self.root / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    self.rows = [json.loads(line) for line in lines if line]

  def __len__(self):
    return len(self.rows) * 2

  def __getitem__(self, i):
    row, b = self.rows[i // 2], i % 2
    img = resize(Image.open(self.root / row["image"]).convert("RGB"), IMG_H, self.width)
    text = row["segmented"] if b else row["text"]
    ids = [self.vocab.stoi[t] for t in kcc_split(text) if t in self.vocab.stoi]
    return to_tensor(img, self.width), b, torch.tensor(ids, dtype=torch.long)


def collate(batch):
  images = torch.stack([x[0] for x in batch])
  flags = torch.tensor([x[1] for x in batch])
  targets = [x[2] for x in batch]
  lengths = torch.tensor([len(t) for t in targets])
  return images, flags, torch.cat(targets), lengths
