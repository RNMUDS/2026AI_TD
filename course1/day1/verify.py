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
    "ex1": {"bandwidth_gbps", "matmul_gflops", "train_step_ms", "memory_gb"},
    "ex2": {"accuracy", "loss", "infer_ms_per_1000", "peak_mem_mb", "params"},
    "ex3": {"rank", "score", "weight_accuracy", "weight_loss", "weight_infer_ms_per_1000", "weight_peak_mem_mb"},
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
        if ex == "ex3":
            ranks = df[df["metric_name"] == "rank"].sort_values("timestamp").groupby("condition").tail(1)
            if set(ranks["condition"]) != {"A1", "A2", "A3"}:
                problems.append(f"{p.name}: 3 モデル分の rank がない")
            w = df[df["metric_name"].str.startswith("weight_")].sort_values("timestamp").groupby("metric_name").tail(1)
            if abs(w["metric_value"].sum() - 1.0) > 1e-6:
                problems.append(f"{p.name}: 重みの合計が 1 でない ({w['metric_value'].sum():.3f})")
            d = root / f"c1_d1_ex3_definition_{g}.txt"
            if not d.exists() or not d.read_text(encoding="utf-8").strip():
                problems.append("班の定義 (c1_d1_ex3_definition_<班>.txt) が空")
            q = root / f"c1_d1_ex3_questions_{g}.txt"
            if not q.exists():
                problems.append("質問カード (c1_d1_ex3_questions_<班>.txt) がない")
            elif "____" in q.read_text(encoding="utf-8"):
                problems.append("質問カードに未記入 (____) が残っている")
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
