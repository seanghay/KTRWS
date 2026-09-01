import json

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from ktrws.data import LineDataset, collate
from ktrws.infer import load, recognize
from ktrws.model import KTRWS
from ktrws.tokenizer import ZWSP, Vocab, insert_boundaries, kcc_split
from ktrws.metrics import boundary_positions, cer, f1, seg_counts
from ktrws.train import ctc_loss, evaluate, predict, train

TEXTS = ["ខ្ញុំទៅសាលារៀន", "មេដឹកនាំអាមេរិកនិងអ៊ីស្រាអែល", "ការរីករាលដាលជំងឺឆ្លង ២០២៥"]


def test_kcc_split():
  assert "".join(kcc_split(TEXTS[2])) == TEXTS[2]
  assert kcc_split("ខ្ញុំ") == ["ខ្ញុំ"]


def test_boundaries():
  seg = insert_boundaries(TEXTS[0])
  assert ZWSP in seg
  assert seg.replace(ZWSP, "") == TEXTS[0]


def test_vocab_roundtrip():
  vocab = Vocab.from_corpus(TEXTS)
  ids = vocab.encode(TEXTS[0], with_boundary=True)
  assert vocab.decode(ids, keep_boundary=False) == TEXTS[0]
  assert ZWSP in vocab.decode(ids)


def test_cer():
  assert cer("abc", "abc") == 0
  assert cer(f"ខ្ញុំ{ZWSP}ទៅ".replace(ZWSP, ""), "ខ្ញុំទៅ") == 0


def test_segmentation_f1():
  gold = f"ខ្ញុំ{ZWSP}ទៅ{ZWSP}សាលា"
  assert f1(seg_counts(gold, gold)) == 1.0
  assert f1(seg_counts(f"ខ្ញុំទៅ{ZWSP}សាលា", gold)) == pytest.approx(2 / 3)
  assert f1(seg_counts(f"ខ្ញុំ{ZWSP}ទៅ{ZWSP}សាល", gold)) == pytest.approx(2 / 3)


def test_boundary_positions():
  assert boundary_positions([1, 9, 9, 2, 9], blank_id=0, boundary_id=9, image_width=320) == [4, 16]


def test_gating_variant():
  model = KTRWS(16, d=64, layers=1, heads=2, ffn=128, gating=True).eval()
  images = torch.randn(1, 3, 32, 320)
  with torch.no_grad():
    assert not torch.allclose(model(images, torch.tensor([0])), model(images, torch.tensor([1])))


def test_model_forward_and_loss():
  vocab = Vocab.from_corpus(TEXTS)
  model = KTRWS(len(vocab), d=64, layers=1, heads=2, ffn=128)
  logits = model(torch.randn(2, 3, 32, 320), torch.tensor([0, 1]))
  assert logits.shape[0] == 2
  assert logits.shape[2] == len(vocab)
  assert torch.isfinite(ctc_loss(logits, torch.tensor([1, 2, 3, 4]), torch.tensor([2, 2])))


def test_flag_changes_output():
  model = KTRWS(16, d=64, layers=1, heads=2, ffn=128).eval()
  images = torch.randn(1, 3, 32, 320)
  with torch.no_grad():
    assert not torch.allclose(model(images, torch.tensor([0])), model(images, torch.tensor([1])))


def make_dataset(root, vocab, n=4):
  images = root / "images"
  images.mkdir(parents=True, exist_ok=True)
  rows = []
  for i in range(n):
    text = TEXTS[i % len(TEXTS)]
    Image.new("RGB", (240, 32), (255, 255, 255)).save(images / f"{i}.png")
    rows.append({"image": f"images/{i}.png", "text": text, "segmented": insert_boundaries(text)})
  (root / "labels.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
  )


def test_infer_roundtrip(tmp_path):
  vocab = Vocab.from_corpus(TEXTS)
  config = {"d": 64, "layers": 1, "heads": 2, "ffn": 128}
  model = KTRWS(len(vocab), **config)
  torch.save({"model": model.state_dict(), "vocab": vocab.itos, "config": config}, tmp_path / "w.pt")

  loaded, loaded_vocab = load(tmp_path / "w.pt")
  assert list(loaded_vocab.itos) == list(vocab.itos)

  img = tmp_path / "line.png"
  Image.new("RGB", (240, 32), (255, 255, 255)).save(img)
  assert len(recognize(loaded, loaded_vocab, [img, img])) == 2
  assert ZWSP not in recognize(loaded, loaded_vocab, [img], boundaries=False)[0]


def test_dataset_and_train(tmp_path):
  vocab = Vocab.from_corpus(TEXTS)
  vocab.save(tmp_path / "vocab.txt")
  for split in ("train", "dev"):
    make_dataset(tmp_path / split, vocab)

  ds = LineDataset(tmp_path / "train", vocab)
  images, flags, targets, lengths = collate([ds[0], ds[1]])
  assert images.shape == (2, 3, 32, 320)
  assert flags.tolist() == [0, 1]
  assert targets.numel() == lengths.sum()

  model, vocab = train(tmp_path, epochs=1, batch_size=2, steps=1)
  assert len(predict(model, vocab, torch.randn(2, 3, 32, 320), b=1)) == 2

  loader = DataLoader(LineDataset(tmp_path / "dev", vocab), batch_size=2, collate_fn=collate)
  dev_cer, dev_f1 = evaluate(model, vocab, loader, "cpu")
  assert dev_cer >= 0.0
  assert 0.0 <= dev_f1 <= 1.0
