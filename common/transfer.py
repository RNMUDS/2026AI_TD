"""第5回：学習済みモデル（ImageNet）を自分の写真に合わせて学習し直す（転移学習）．

    from common.transfer import run
    run("ResNet18", "freeze", seed=0)     # 最後の層だけ学習（凍結）
    run("ResNet18", "finetune", seed=0)   # 全部の層を学習（微調整）
    run("ViT", "freeze", seed=0)          # ViT-B/16 の最後の層だけ学習

データの分け方は第4回の mydata.run と同じ（同じシードなら同じ写真がテストになる）．
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

from .mydata import _accuracy, _set_seed, image_dir, loaders

SIZE = 224                                                     # ImageNet で学習したときの大きさ
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]       # ImageNet の画像の平均と標準偏差
LR = {"freeze": 0.001, "finetune": 0.0001}                     # 微調整は学習済みの重みを壊さないよう小さく
EPOCHS = {"freeze": 20, "finetune": 10}


def build(kind, mode, n_class):
    """⑦：学習済みの重みを読み，最後の層を n_class 個の出力に替える．freeze なら最後の層以外を凍結する．"""
    if kind == "ResNet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif kind == "ViT":
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError('kind は "ResNet18" か "ViT"')
    if mode == "freeze":
        for p in net.parameters():
            p.requires_grad = False
    elif mode != "finetune":
        raise ValueError('mode は "freeze" か "finetune"')
    if kind == "ResNet18":
        net.fc = nn.Linear(net.fc.in_features, n_class)              # 512 → n_class
    else:
        net.heads.head = nn.Linear(net.heads.head.in_features, n_class)   # 768 → n_class
    return net


def count_params(net):
    """（全体のパラメータ数, 学習するパラメータ数）"""
    return (sum(p.numel() for p in net.parameters()),
            sum(p.numel() for p in net.parameters() if p.requires_grad))


def _features(backbone, loader, device):
    """凍結した本体を 1 回だけ通し，最後の層に入る値（特徴量）を集める．"""
    xs, ys = [], []
    backbone.eval()
    with torch.no_grad():
        for images, labels in loader:
            xs.append(backbone(images.to(device)).cpu())
            ys.append(labels)
    return TensorDataset(torch.cat(xs), torch.cat(ys))


def _vit_freeze_loaders(net, parts, device):
    """ViT の凍結：本体の出力はエポックごとに同じなので先に 1 回だけ計算する（結果は毎回通すのと同じで，速い）．
    ViT-B/16 には BatchNorm も Dropout も無いので，train() と eval() で本体の出力は変わらない．"""
    vit = net[1]                                   # net は (Normalize, ViT)
    fc = vit.heads.head
    vit.heads.head = nn.Identity()
    feats = [_features(net, ld, device) for ld in parts]
    vit.heads.head = fc
    return fc, [DataLoader(f, batch_size=ld.batch_size, shuffle=(i == 0)) for i, (f, ld) in enumerate(zip(feats, parts))]


def run(kind, mode, seed=0, epochs=None, device=None, root=None):
    """③〜⑫を 1 回実行し，テスト正解率などを返す．"""
    from .device import get_device
    device = device or get_device()
    epochs = epochs or EPOCHS[mode]
    train_loader, val_loader, test_loader, classes = loaders(seed, size=SIZE, root=root or image_dir())
    _set_seed(seed)                                      # 最後の層の初期値とミニバッチの順をシードで決める
    net = build(kind, mode, len(classes)).to(device)
    norm = transforms.Normalize(MEAN, STD)               # ②の Normalize と同じことをモデルの入口で行う
    model = nn.Sequential(norm, net)
    parts = [train_loader, val_loader, test_loader]
    t0 = time.perf_counter()
    trainable = model
    if kind == "ViT" and mode == "freeze":
        trainable, parts = _vit_freeze_loaders(model, parts, device)
    train_loader, val_loader, test_loader = parts
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in trainable.parameters() if p.requires_grad], lr=LR[mode])
    val_accs = []
    for _ in range(epochs):
        trainable.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(trainable(images), labels)
            loss.backward()
            optimizer.step()
        val_accs.append(_accuracy(trainable, val_loader, device))
    total, n_train = count_params(net)
    return {"test_acc": _accuracy(trainable, test_loader, device), "val_acc_best": max(val_accs),
            "val_acc_last": val_accs[-1], "train_sec": time.perf_counter() - t0,
            "n_test": len(test_loader.dataset), "params_total": total, "params_trained": n_train}
