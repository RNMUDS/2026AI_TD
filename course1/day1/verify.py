#!/usr/bin/env python3
"""提出物の形式検証（Ⅰ第1回）．  python course1/day1/verify.py --group 3"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import RESULTS  # noqa: E402
from common.logger import COLUMNS  # noqa: E402

REQ = {
    "ex0": {"test_accuracy", "epochs", "train_sec"},
    "ex1": {"bandwidth_gbps", "matmul_gflops", "train_step_ms", "memory_gb"},
    "ex2": {"accuracy", "loss", "infer_ms_per_1000", "peak_mem_mb", "params"},
}


def check(group, root=RESULTS):
    g = f"{int(group):02d}"
    problems, info = [], {}
    for ex, need in REQ.items():
        p = root / f"c1_d1_{ex}_{g}.csv"
        if not p.exists():
            problems.append(f"{p.name} がない")
            continue
        df = pd.read_csv(p)
        if list(df.columns) != COLUMNS:
            problems.append(f"{p.name}: 列が共通形式と違う {list(df.columns)}")
            continue
        missing = need - set(df["metric_name"])
        if missing:
            problems.append(f"{p.name}: 指標が足りない {sorted(missing)}")
        info[ex] = len(df)
        if ex == "ex2":
            models = set(df.loc[df["metric_name"] == "accuracy", "condition"])
            if models != {"A1", "A2", "A3"}:
                problems.append(f"{p.name}: 3 モデル分の accuracy がない {sorted(models)}")
    tbl = root / f"table_c1_d1_ex2_{g}.csv"
    if not tbl.exists():
        problems.append(f"{tbl.name} がない")
    return problems, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    problems, info = check(a.group, Path(a.root) if a.root else RESULTS)
    print("行数:", info)
    if problems:
        print("NG:"); [print(" -", p) for p in problems]
        return 1
    print("OK: 提出物の形式に問題なし")
    return 0


if __name__ == "__main__":
    sys.exit(main())
