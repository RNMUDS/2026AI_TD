"""第4回：自分で撮った画像（assets/myphotos/）を学習に使える形（assets/mydata/）にそろえ，ex2 で 5 シードを回す．

    python setup/prepare_mydata.py            # myphotos/ → mydata/（HEIC を JPEG に，長い辺を 256 画素に縮める）
    from common.mydata import image_dir, run
    result = run("CNN", seed=0)               # ex1 で自分で書いた①〜⑫と同じ手順を 1 回実行する

元の写真は書き換えない．mydata/ は何度作り直してもよい．
"""
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import ASSETS

PHOTOS = ASSETS / "myphotos"      # 自分で撮った写真（クラスごとのフォルダ）
MYDATA = ASSETS / "mydata"        # 学習に使う形にそろえたもの
CABLE = ASSETS / "cable1"         # 写真が無いときの代わり（第1回のケーブル画像）
LONG_SIDE = 256
MIN_PER_CLASS, GOOD_PER_CLASS, MAX_PER_CLASS = 20, 50, 200
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HEIC_EXT = {".heic", ".heif"}


def _open(path):
    """写真を RGB の PIL 画像で開く．HEIC は Mac の sips で JPEG にしてから開く．"""
    if path.suffix.lower() in HEIC_EXT:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.jpg"
            subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(out)],
                           check=True, capture_output=True)
            image = Image.open(out)
            image.load()
    else:
        image = Image.open(path)
    return ImageOps.exif_transpose(image).convert("RGB")   # スマホの縦横の向きを直す


def _classes(src):
    return sorted(p for p in Path(src).iterdir() if p.is_dir() and not p.name.startswith("."))


def prepare(src=PHOTOS, dst=MYDATA, min_per_class=MIN_PER_CLASS, good_per_class=GOOD_PER_CLASS):
    """src/<クラス名>/ の写真を dst/<クラス名>/ に JPEG（長い辺 256 画素）でそろえる．問題があれば理由を出して止める．"""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise SystemExit(f"{src} がありません．クラスごとのフォルダを作り，写真を入れてください")
    classes = _classes(src)
    if len(classes) < 2:
        raise SystemExit(f"{src} の中のフォルダ（クラス）が {len(classes)} 個です．2 つ作ってください")
    non_ascii = [c.name for c in classes if not c.name.isascii()]
    if non_ascii:
        raise SystemExit(f"フォルダ名 {non_ascii} を半角英数字に変えてください（例：pen_mine）")
    if dst.exists():
        shutil.rmtree(dst)
    counts, skipped = {}, []
    for c in classes:
        files = sorted(p for p in c.iterdir() if p.suffix.lower() in IMAGE_EXT | HEIC_EXT)
        skipped += [p.name for p in c.iterdir() if p.is_file() and not p.name.startswith(".")
                    and p.suffix.lower() not in IMAGE_EXT | HEIC_EXT]
        (dst / c.name).mkdir(parents=True)
        n = 0
        for p in files[:MAX_PER_CLASS]:
            try:
                image = _open(p)
            except Exception as e:  # 壊れた画像は飛ばして知らせる
                skipped.append(f"{p.name}（読めない：{type(e).__name__}）")
                continue
            image.thumbnail((LONG_SIDE, LONG_SIDE))
            out = dst / c.name / f"{p.stem}.jpg"
            if out.exists():   # IMG_1.HEIC と IMG_1.JPG のように名前が重なるときは上書きしない
                out = dst / c.name / f"{p.stem}_{p.suffix[1:].lower()}.jpg"
            image.save(out, quality=90)
            n += 1
        counts[c.name] = n
    for name, n in counts.items():
        mark = "OK" if n >= good_per_class else (f"少ない（{good_per_class} 枚以上を勧める）" if n >= min_per_class else "足りない")
        print(f"  {name:20s} {n:4d} 枚  {mark}")
    if skipped:
        print("  使わなかったファイル:", ", ".join(skipped[:10]), "…" if len(skipped) > 10 else "")
    short = [k for k, n in counts.items() if n < min_per_class]
    if short:
        shutil.rmtree(dst)   # 足りないまま mydata/ が残ると，ノートブックがそれで進んでしまう
        raise SystemExit(f"{', '.join(short)} が {min_per_class} 枚未満です．写真を足してからもう一度実行してください")
    print(f"→ {dst} にそろえました（{len(classes)} クラス・計 {sum(counts.values())} 枚）")
    return counts


def image_dir():
    """自分のデータ（mydata/）があればそれを，無ければケーブル画像を返す．"""
    if MYDATA.exists() and len(_classes(MYDATA)) >= 2:
        return MYDATA
    print("assets/mydata/ が無いので，ケーブル画像（assets/cable1/）で進めます")
    return CABLE


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


_CACHE = {}


def _load(root, size):
    """画像を最初に 1 回だけ読み，テンソルにしてメモリに置く（5 シードを速く回すため）．"""
    key = (str(root), size)
    if key not in _CACHE:
        full = ImageFolder(root=str(root))
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        _CACHE[key] = ([(tf(image), label) for image, label in full], full.classes)
    return _CACHE[key]


def loaders(seed=0, size=64, batch_size=8, root=None):
    """②③④：64×64 にそろえ，70% 訓練・15% 検証・15% テストに分ける（GW3 カード N・P・Q と同じ）．"""
    cached, classes = _load(root or image_dir(), size)
    _set_seed(seed)
    n_train, n_val = int(0.7 * len(cached)), int(0.15 * len(cached))
    parts = random_split(cached, [n_train, n_val, len(cached) - n_train - n_val])
    return (DataLoader(parts[0], batch_size=batch_size, shuffle=True),
            DataLoader(parts[1], batch_size=batch_size, shuffle=False),
            DataLoader(parts[2], batch_size=batch_size, shuffle=False),
            classes)


def make_model(kind, n_class, c=3, h=64, w=64):
    """⑦：GW3 カード M（MLP）・R（CNN）と同じ形．kind に関数を渡すと kind(n_class) でモデルを作る（オプション課題の ViT）．"""
    if callable(kind):
        return kind(n_class)
    if kind == "MLP":
        return nn.Sequential(nn.Flatten(), nn.Linear(c * h * w, 512), nn.ReLU(),
                             nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_class))
    if kind == "CNN":
        return nn.Sequential(nn.Conv2d(c, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Flatten(), nn.Linear(16 * (h // 2) * (w // 2), 128), nn.ReLU(),
                             nn.Linear(128, n_class))
    raise ValueError('kind は "MLP" か "CNN"')


def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run(kind, seed=0, epochs=20, device=None, root=None):
    """③〜⑫を 1 回実行し，テスト正解率などを返す．seed で分け方・重みの初期値・並び順が変わる．"""
    from .device import get_device
    device = device or get_device()
    train_loader, val_loader, test_loader, classes = loaders(seed, root=root)
    model = make_model(kind, len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    t0 = time.perf_counter()
    val_accs = []
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        val_accs.append(_accuracy(model, val_loader, device))
    return {"test_acc": _accuracy(model, test_loader, device), "val_acc_best": max(val_accs),
            "val_acc_last": val_accs[-1], "train_sec": time.perf_counter() - t0,
            "n_test": len(test_loader.dataset)}


def diff_ratio(a, b):
    """2 つの条件の値（シードごと）から，平均の差 ÷ 標準偏差 を返す．標準偏差は 2 つの標準偏差を 2 乗平均したもの．"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sd = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    diff = b.mean() - a.mean()
    if sd == 0:   # 5 回とも同じ値（例：両方 100%）
        return diff, sd, (0.0 if diff == 0 else float("inf"))
    return diff, sd, abs(diff) / sd


def verdict(ratio):
    """判定の目安：2 以上なら「差がある」，1 未満なら「差があるとは言えない」．"""
    if ratio >= 2:
        return "差がある"
    if ratio < 1:
        return "差があるとは言えない"
    return "どちらとも言えない（回数を増やして確かめる）"
