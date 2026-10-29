#!/usr/bin/env python3
"""第5回：学習済みの重み（ResNet18 45MB・ViT-B/16 330MB）を先にダウンロードして確かめる．

  python setup/download_weights.py

2 回目からは ~/.cache/torch/hub/checkpoints/ にあるものを使う（ダウンロードしない）．
"""
import time

from torchvision import models

if __name__ == "__main__":
    for name, fn, weights in [("ResNet18", models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
                              ("ViT-B/16", models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1)]:
        t0 = time.perf_counter()
        net = fn(weights=weights)
        n = sum(p.numel() for p in net.parameters())
        print(f"  {name:9s} OK｜パラメータ {n:,}｜{time.perf_counter() - t0:.0f} 秒")
    print("→ 学習済みの重みがそろいました")
