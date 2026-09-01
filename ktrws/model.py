import math

import torch
from torch import nn


class ResBlock(nn.Module):
  def __init__(self, cin, cout, stride=1):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(cin, cout, 1, bias=False),
      nn.BatchNorm2d(cout),
      nn.ReLU(inplace=True),
      nn.Conv2d(cout, cout, 3, stride, 1, bias=False),
      nn.BatchNorm2d(cout),
      nn.ReLU(inplace=True),
      nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
      nn.BatchNorm2d(cout),
    )
    self.skip = (
      nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
      if stride != 1 or cin != cout
      else nn.Identity()
    )
    self.act = nn.ReLU(inplace=True)

  def forward(self, x):
    return self.act(self.conv(x) + self.skip(x))


class CNN(nn.Module):
  CHANNELS = [32, 64, 128, 256, 384, 512]
  STRIDES = [(2, 2), 1, 1, (2, 2), 1, 1]
  REPEATS = [1, 1, 3, 1, 2, 1]

  def __init__(self, d=512):
    super().__init__()
    channels = [*self.CHANNELS[:-1], d]
    layers, cin = [], 3
    for cout, stride, repeat in zip(channels, self.STRIDES, self.REPEATS, strict=True):
      layers.append(ResBlock(cin, cout, stride))
      layers += [ResBlock(cout, cout) for _ in range(repeat - 1)]
      cin = cout
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


def sinusoidal(length, d, device):
  pos = torch.arange(length, device=device)[:, None]
  freq = torch.exp(torch.arange(0, d, 2, device=device) * (-math.log(10000.0) / d))
  pe = torch.zeros(length, d, device=device)
  pe[:, 0::2] = torch.sin(pos * freq)
  pe[:, 1::2] = torch.cos(pos * freq)
  return pe


class BoundaryProjector(nn.Module):
  def __init__(self, d, gating=False):
    super().__init__()
    self.gating = gating
    self.emb = nn.Embedding(2, d)
    self.left = nn.Linear(d, d)
    self.right = nn.Linear(d, d)

  def forward(self, b):
    e = self.emb(b)
    if self.gating:
      return torch.sigmoid(self.left(e)), self.right(e)
    return self.left(e), self.right(e)


class MAFS(nn.Module):
  def __init__(self, d, n=5, bottleneck=4):
    super().__init__()
    hidden = d // bottleneck
    self.router = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n))
    self.adapters = nn.ModuleList(
      nn.Sequential(nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, d)) for _ in range(n)
    )

  def forward(self, g):
    r = self.router(g.mean(dim=1)).softmax(-1)
    h = torch.stack([a(g) for a in self.adapters], dim=-1)
    return (h * r[:, None, None, :]).sum(-1)


class KTRWS(nn.Module):
  def __init__(self, vocab_size, d=512, layers=3, heads=8, ffn=2048, dropout=0.1, n_modality=5, gating=False):
    super().__init__()
    self.gating = gating
    self.cnn = CNN(d)
    enc = nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True)
    self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
    self.boundary = BoundaryProjector(d, gating)
    self.mafs = MAFS(d, n_modality)
    self.head = nn.Linear(d, vocab_size)

  def forward(self, images, b):
    f = self.cnn(images)
    n, d, h, w = f.shape
    g = f.flatten(2).transpose(1, 2)
    g = self.encoder(g + sinusoidal(g.size(1), d, g.device))
    g = g.transpose(1, 2).reshape(n, d, h, w).mean(dim=2).transpose(1, 2)

    fused = self.mafs(g)
    left, right = self.boundary(b)
    if self.gating:
      u = left[:, None, :] * fused + (1 - left[:, None, :]) * right[:, None, :]
    else:
      u = right[:, None, :] * fused + left[:, None, :]
    return self.head(u)
