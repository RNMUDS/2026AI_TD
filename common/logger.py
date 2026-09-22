"""全演習共通の CSV ロガー．集計 (aggregate/) はこの形式のみを前提とする．

出力先: results/{course}_{day}_{exercise}_{group_id}.csv
"""
import csv
import datetime as _dt
from pathlib import Path

import pandas as pd

from . import RESULTS
from .device import machine_string

COLUMNS = ["timestamp", "group_id", "member_role", "course", "day", "exercise",
           "condition", "seed", "metric_name", "metric_value", "elapsed_sec", "device", "machine"]
ROLES = ("implementer", "verifier", "recorder", "auditor", "collector")   # collector は 5 人班の 5 つ目の役割


class ResultLogger:
    def __init__(self, group_id, member_role, course, day, exercise, device="cpu", root=None):
        assert member_role in ROLES, f"member_role は {ROLES} のいずれか"
        self.group_id = f"{int(group_id):02d}" if str(group_id).isdigit() else str(group_id)
        self.member_role = member_role
        self.course, self.day, self.exercise = course, day, exercise
        self.device = str(device)
        self.machine = machine_string()
        root = Path(root) if root else RESULTS
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{course}_{day}_{exercise}_{self.group_id}.csv"
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def log(self, metric_name, metric_value, condition="", seed=-1, elapsed_sec=None):
        row = {"timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
               "group_id": self.group_id, "member_role": self.member_role,
               "course": self.course, "day": self.day, "exercise": self.exercise,
               "condition": condition, "seed": seed, "metric_name": metric_name,
               "metric_value": float(metric_value),
               "elapsed_sec": "" if elapsed_sec is None else round(float(elapsed_sec), 3),
               "device": self.device, "machine": self.machine}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
        return row

    def log_many(self, metrics: dict, **kw):
        return [self.log(k, v, **kw) for k, v in metrics.items()]


def read_results(course=None, day=None, exercise=None, root=None):
    """results/ 以下の CSV を結合して DataFrame で返す（班横断集計用）．"""
    root = Path(root) if root else RESULTS
    import re
    c, d, e = course or "c[0-9]", day or "d[0-9]", exercise or "ex[0-9]"
    rx = re.compile("^" + c + "_" + d + "_" + e + "_([0-9]+)[.]csv$")
    frames = [pd.read_csv(p) for p in sorted(root.glob("*.csv")) if rx.match(p.name)]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat(frames, ignore_index=True)
    df["group_id"] = df["group_id"].astype(str).str.zfill(2)
    return df
