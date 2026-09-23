"""オプション課題「何枚あれば学習できるか」の学習をまとめた関数．

    from common.datasize import run
    result = run("MLP", n_train=1000, seed=GROUP_ID, device=device)

GW3 試行①（MLP）・試行③（シンプルな CNN）と同じ構造のモデルを，FashionMNIST の訓練データを
n_train 枚に減らして学習し，テストデータ（1 万枚）の正解率を返す．
"""
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from . import ASSETS

N_TRAIN_CHOICES = (100, 300, 1000, 3000, 10000, 48000)


def _model(kind, c, h, w, n_class=10):
    if kind == "MLP":            # GW3 試行①と同じ
        return nn.Sequential(nn.Flatten(), nn.Linear(c * h * w, 512), nn.ReLU(),
                             nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_class))
    if kind == "CNN":            # GW3 試行③と同じ（畳み込み 1 層）
        return nn.Sequential(nn.Conv2d(c, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(16 * (h // 2) * (w // 2), 128), nn.ReLU(),
                             nn.Linear(128, n_class))
    raise ValueError('MODEL は "MLP" か "CNN" にする')


def run(kind, n_train, seed=0, device="cpu", epochs=20, batch_size=64):
    if n_train not in N_TRAIN_CHOICES:
        raise ValueError(f"N_TRAIN は {N_TRAIN_CHOICES} のどれかにする")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    tf = transforms.Compose([transforms.ToTensor()])
    root = str(ASSETS / "data")
    full = datasets.FashionMNIST(root=root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tf)
    train, _ = random_split(full, [48000, 12000])                 # GW3 と同じ 8:2 の分け方
    train = Subset(train, range(n_train))                        # そのうち n_train 枚だけ使う
    tl = DataLoader(train, batch_size=batch_size, shuffle=True)
    model = _model(kind, 1, 28, 28).to(device)
    lossf, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=0.001)
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in DataLoader(test, batch_size=1000):
            p = model(x.to(device)).argmax(1).cpu()
            correct += (p == y).sum().item(); total += len(y)
    return {"test_accuracy": correct / total, "train_sec": time.perf_counter() - t0, "n_train": n_train}
