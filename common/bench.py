"""E1-1 環境ベンチマーク（各項目 数秒）．verify/micro.py と同じ 3 種を短縮したもの．

B1 メモリ帯域: 大きなベクトルの加算（読み 2 + 書き 1 → 3×N×4 byte）
B2 演算:       行列積 (N×N)·(N×N) → 2N^3 FLOP
B3 学習:       小型 A3 の学習ステップ時間
"""
import time

import torch
import torch.nn as nn

from .device import synchronize
from .models import build_model


def _timeit(fn, device, iters, warmup=3):
    for _ in range(warmup):
        fn()
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - t0) / iters


def timed_loop(fn, device, min_sec=2.0, min_iters=5, warmup=2):
    """fn を最低 min_sec 秒回し，1 回あたりの時間を (中央値, 初回) で返す．

    GPU は最初の実行に準備（カーネルの読み込みなど）の時間が入るため，warmup 回は捨てる．
    1 回だけの値はぶれるので，代表値は 2 秒以上回した中央値で取る．
    （M4 Max で 20 秒連続実行して確認：中央値は最初から一定で，初回の 1 回はむしろ約半分の速さ．
    ファンレス機の熱による低下は数十秒〜数分の連続負荷で起こるもので，2 秒の計測にはほぼ効かない）
    """
    for _ in range(warmup):
        fn()
    synchronize(device)
    times = []
    t_start = time.perf_counter()
    while len(times) < min_iters or time.perf_counter() - t_start < min_sec:
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    import statistics
    return statistics.median(times), times[0], len(times)


def warm_up(device, sec=2.0):
    """GPU を sec 秒間回して最初の遅い実行を済ませる．時間を比較する測定の直前に呼ぶ．"""
    a = torch.randn(1024, 1024, device=device)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        a = a @ a * 1e-3
    synchronize(device)


def bandwidth_gbps(device, n=64 * 1024 * 1024, min_sec=2.0):
    """戻り値: (中央値の GB/s, 初回 1 回の GB/s, 1回の秒数)"""
    a = torch.ones(n, device=device)
    b = torch.ones(n, device=device)
    med, first, _ = timed_loop(lambda: torch.add(a, b), device, min_sec)
    byt = 3 * n * 4
    return byt / med / 1e9, byt / first / 1e9, med


def matmul_gflops(device, n=2048, min_sec=2.0):
    """戻り値: (中央値の GFLOPS, 初回 1 回の GFLOPS, 1回の秒数)"""
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    med, first, _ = timed_loop(lambda: a @ b, device, min_sec)
    fl = 2 * n ** 3
    return fl / med / 1e9, fl / first / 1e9, med


def train_step_ms(device, vocab_size=5000, max_len=32, target_params=300000, batch=128, iters=20):
    torch.manual_seed(0)
    model, info = build_model("A3", target_params, vocab_size, max_len=max_len)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    x = torch.randint(1, vocab_size, (batch, max_len), device=device)
    y = torch.randint(0, 2, (batch,), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        lossf(model(x), y).backward()
        opt.step()

    sec = _timeit(step, device, iters)
    return sec * 1000, info["params"]
