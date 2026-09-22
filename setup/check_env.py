#!/usr/bin/env python3
"""環境診断．前週に各自実行する．results/env_check.json を出力する．"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    rep = {"ok": True, "problems": []}
    try:
        import torch, numpy, pandas, yaml, matplotlib, psutil, pyarrow, huggingface_hub, nbformat  # noqa
        rep["versions"] = {"torch": torch.__version__, "numpy": numpy.__version__,
                           "pandas": pandas.__version__, "python": sys.version.split()[0]}
    except ImportError as e:
        rep["ok"] = False
        rep["problems"].append(f"import error: {e}  -> pip install -r requirements.txt")
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 1
    from common.device import get_device, machine_info
    from common import bench, RESULTS
    dev = get_device()
    rep["machine"] = machine_info()
    rep["device"] = str(dev)
    if dev.type != "mps":
        rep["problems"].append("MPS が使えない．CPU で動くが時間制約は免除される")
    gb, _, _ = bench.bandwidth_gbps(dev)
    gf, _, _ = bench.matmul_gflops(dev)
    ms, _ = bench.train_step_ms(dev)
    rep["bench"] = {"bandwidth_gbps": round(gb, 1), "matmul_gflops": round(gf, 1), "train_step_ms": round(ms, 2)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "env_check.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
