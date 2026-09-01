import argparse

import torch
from PIL import Image, ImageDraw

from ktrws.data import IMG_W, resize, to_tensor
from ktrws.infer import load


@torch.no_grad()
def ground(model, vocab, path, device="cpu"):
  img = resize(Image.open(path).convert("RGB"))
  logits = model(to_tensor(img).unsqueeze(0).to(device), torch.ones(1, dtype=torch.long, device=device))
  ids = logits.argmax(-1)[0].tolist()

  boundary = len(vocab) - 1
  runs, prev = [], None
  for frame, token in enumerate(ids):
    if token == boundary:
      if token == prev:
        runs[-1].append(frame)
      else:
        runs.append([frame])
    prev = token

  stride = IMG_W / len(ids)
  xs = [(sum(run) / len(run) + 0.5) * stride for run in runs]
  return img, [x for x in xs if x < img.width], vocab.decode(ids)


def draw(img, xs, scale=3, color=(214, 40, 40)):
  out = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
  pen = ImageDraw.Draw(out)
  for x in xs:
    pen.line([(x * scale, 0), (x * scale, out.height)], fill=color, width=2)
  return out


def sheet(panels, pad=10, gap=8):
  width = max(p.width for p in panels) + pad * 2
  height = sum(p.height for p in panels) + gap * (len(panels) - 1) + pad * 2
  canvas = Image.new("RGB", (width, height), (255, 255, 255))
  y = pad
  for p in panels:
    canvas.paste(p, (pad, y))
    y += p.height + gap
  return canvas


def main():
  p = argparse.ArgumentParser(description="Project predicted word boundaries onto the input image")
  p.add_argument("images", nargs="+")
  p.add_argument("--weights", default="ktrws.pt")
  p.add_argument("--device", default="cpu")
  p.add_argument("--out", default="grounding.png")
  args = p.parse_args()

  model, vocab = load(args.weights, args.device)
  panels = []
  for path in args.images:
    img, xs, text = ground(model, vocab, path, args.device)
    panels.append(draw(img, xs))
    print(f"{path}\t{len(xs)} boundaries\t{text}")
  sheet(panels).save(args.out)
  print(args.out)


if __name__ == "__main__":
  main()
