#!/usr/bin/env python3
"""GPU の速さが時間とともにどう変わるかを記録する（教員の確認用．授業では使わない）．

    python setup/gpu_timeline.py            # 60 秒（ファンレス機の熱による低下を見るなら 120 秒以上）
    python setup/gpu_timeline.py --sec 180

行列積（2048×2048）を回し続け，時間帯ごとの GFLOPS の中央値を表示する．
「最初だけ速い」「数十秒後に遅くなる」が本当に起きるかを，測り方を変えずに確かめる．
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.device import synchronize  # noqa: E402

WINDOWS = [(0, 0.3), (0.3, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 60), (60, 120), (120, 180), (180, 300)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(a.n, a.n, device=dev)
    y = torch.randn(a.n, a.n, device=dev)
    flops = 2 * a.n ** 3
    x @ y
    synchronize(dev)
    rec, t_start = [], time.perf_counter()
    while time.perf_counter() - t_start < a.sec:
        t0 = time.perf_counter()
        x @ y
        synchronize(dev)
        rec.append((t0 - t_start, flops / (time.perf_counter() - t0) / 1e9))
    print(f"デバイス: {dev}  行列 {a.n}×{a.n}  回数 {len(rec)}  最初の 1 回: {rec[0][1]:.0f} GFLOPS")
    print("| 時間帯（秒） | GFLOPS（中央値） | 回数 |")
    print("|---|---|---|")
    for lo, hi in WINDOWS:
        v = [g for t, g in rec if lo <= t < hi]
        if v:
            print(f"| {lo}–{hi} | {statistics.median(v):.0f} | {len(v)} |")


if __name__ == "__main__":
    main()
