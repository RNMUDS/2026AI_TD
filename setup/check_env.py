#!/usr/bin/env python3
"""環境診断．前週に各自実行する．results/env_check.json を出力する．

進行状況を逐次表示し（各項目 1〜3 秒），最後に結果を JSON で表示する．
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEPS = 6


def step(i, msg):
    """進行状況を 1 行で表示する（バッファせず即時出力）．"""
    print(f"[{i}/{STEPS}] {msg} ...", end=" ", flush=True)


def done(t0, extra=""):
    print(f"完了 ({time.perf_counter() - t0:.1f} 秒){'  ' + extra if extra else ''}", flush=True)


def main():
    print("環境診断を開始します（30 秒ほどかかります）", flush=True)
    rep = {"ok": True, "problems": []}

    t0 = time.perf_counter()
    step(1, "ライブラリの読み込みを確認中（PyTorch は初回 5 秒ほどかかる）")
    try:
        import torch, numpy, pandas, yaml, matplotlib, psutil, pyarrow, huggingface_hub, nbformat  # noqa
        rep["versions"] = {"torch": torch.__version__, "numpy": numpy.__version__,
                           "pandas": pandas.__version__, "python": sys.version.split()[0]}
        done(t0, f"torch {torch.__version__}")
    except ImportError as e:
        print("失敗", flush=True)
        rep["ok"] = False
        rep["problems"].append(f"import error: {e}  -> uv pip install -r requirements.txt")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 1

    from common.device import get_device, machine_info
    from common import bench, RESULTS

    t0 = time.perf_counter()
    step(2, "機種情報を取得中")
    rep["machine"] = machine_info()
    done(t0, f"{rep['machine']['chip']} / {rep['machine']['memory_gb']}GB")

    t0 = time.perf_counter()
    step(3, "使用デバイスを確認中")
    dev = get_device()
    rep["device"] = str(dev)
    if dev.type != "mps":
        rep["problems"].append("MPS が使えない．CPU で動くが時間制約は免除される")
    done(t0, f"device = {dev}")

    t0 = time.perf_counter()
    step(4, "メモリ帯域を計測中（2 秒）")
    gb, _, _ = bench.bandwidth_gbps(dev)
    done(t0, f"{gb:.1f} GB/s")

    t0 = time.perf_counter()
    step(5, "演算性能を計測中（2 秒）")
    gf, _, _ = bench.matmul_gflops(dev)
    done(t0, f"{gf:.0f} GFLOPS")

    t0 = time.perf_counter()
    step(6, "学習ステップ時間を計測中")
    ms, _ = bench.train_step_ms(dev)
    done(t0, f"{ms:.2f} ms/step")

    rep["bench"] = {"bandwidth_gbps": round(gb, 1), "matmul_gflops": round(gf, 1), "train_step_ms": round(ms, 2)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "env_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    print("\n===== 診断結果 =====", flush=True)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("\n保存先:", RESULTS / "env_check.json")
    if rep["problems"]:
        print("注意:", " / ".join(rep["problems"]))
    else:
        print("問題なし．演習の準備ができています．")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
