"""dl-course 共通基盤．各演習ノートブックは
    from common import device, seed, logger, memory, models, data, train, runner, bench
の形で利用する．"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config" / "course.yaml"


def load_config():
    import yaml
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)
