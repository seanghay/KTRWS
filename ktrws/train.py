import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ktrws.data import LineDataset, collate
from ktrws.metrics import cer, f1, seg_counts
from ktrws.model import KTRWS
from ktrws.tokenizer import ZWSP, Vocab


def ctc_loss(logits, targets, target_lengths):
  logp = logits.float().log_softmax(-1).transpose(0, 1)
  if logp.device.type == "mps":
    logp = logp.cpu()
  input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logp.device)
  targets, target_lengths = targets.to(logp.device), target_lengths.to(logp.device)
  return F.ctc_loss(logp, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)


@torch.no_grad()
def evaluate(model, vocab, loader, device):
  model.eval()
  total_cer, n, counts = 0.0, 0, [0, 0, 0]
  for images, flags, targets, lengths in loader:
    ids = model(images.to(device), flags.to(device)).argmax(-1).cpu()
    offset = 0
    for row, flag, length in zip(ids, flags.tolist(), lengths, strict=True):
      gold = vocab.decode_targets(targets[offset : offset + length].tolist())
      offset += length
      pred = vocab.decode(row.tolist())
      total_cer += cer(pred.replace(ZWSP, ""), gold.replace(ZWSP, ""))
      n += 1
      if flag:
        counts = [a + b for a, b in zip(seg_counts(pred, gold), counts, strict=True)]
  return total_cer / max(1, n), f1(counts)


def train(
  data_dir, epochs=5, batch_size=32, lr=1e-4, device="cpu", steps=None, out=None, gating=False, workers=0
):
  root = Path(data_dir)
  vocab = Vocab.load(root / "vocab.txt")
  train_loader = DataLoader(
    LineDataset(root / "train", vocab),
    batch_size=batch_size,
    collate_fn=collate,
    shuffle=True,
    num_workers=workers,
  )
  dev_loader = DataLoader(
    LineDataset(root / "dev", vocab), batch_size=batch_size, collate_fn=collate, num_workers=workers
  )

  config = {"gating": gating}
  model = KTRWS(len(vocab), **config).to(device)
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  sched = torch.optim.lr_scheduler.CyclicLR(
    opt, base_lr=lr / 10, max_lr=lr, step_size_up=max(1, len(train_loader) // 2), cycle_momentum=False
  )

  for epoch in range(epochs):
    model.train()
    bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
    for i, (images, flags, targets, lengths) in enumerate(bar):
      loss = ctc_loss(model(images.to(device), flags.to(device)), targets, lengths)
      opt.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
      opt.step()
      sched.step()
      bar.set_postfix(loss=f"{loss.item():.3f}")
      if steps and i + 1 >= steps:
        break
    dev_cer, dev_f1 = evaluate(model, vocab, dev_loader, device)
    print(f"dev CER {dev_cer:.4f} | seg F1 {dev_f1:.4f}")
    if out:
      torch.save({"model": model.state_dict(), "vocab": vocab.itos, "config": config}, out)
  return model, vocab


@torch.no_grad()
def predict(model, vocab, images, b, device="cpu"):
  model.eval()
  flags = torch.full((images.size(0),), int(b), dtype=torch.long, device=device)
  ids = model(images.to(device), flags).argmax(-1).cpu()
  return [vocab.decode(row.tolist()) for row in ids]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="data")
  p.add_argument("--epochs", type=int, default=5)
  p.add_argument("--batch-size", type=int, default=32)
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--device", default="cpu")
  p.add_argument("--out", default="ktrws.pt")
  p.add_argument("--workers", type=int, default=0)
  p.add_argument("--gating", action="store_true", help="Eq. 13-15 gated fusion instead of FiLM")
  args = p.parse_args()
  train(
    args.data,
    args.epochs,
    args.batch_size,
    args.lr,
    args.device,
    out=args.out,
    gating=args.gating,
    workers=args.workers,
  )


if __name__ == "__main__":
  main()
