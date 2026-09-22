#!/usr/bin/env python3
"""前週に各自実行．コーパスと配布重みを assets/ に取得する．

  python setup/download_assets.py
コーパスは config/course.yaml の指定に従い HuggingFace Hub から取得し，語彙とテンソルを決定的に生成する．
重みは day1.weights_url が空なら，教員から配布された assets/weights/ をそのまま使う．
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ASSETS, load_config, data  # noqa: E402


def main():
    cfg = load_config()
    npz, vj = data.prepare_corpus(cfg)
    print("corpus :", npz, vj)
    w = ASSETS / "weights"
    url = cfg["day1"].get("weights_url") or ""
    need = [w / f"day1_{a}.pt" for a in ("A1", "A2", "A3")] + [w / "day1_meta.json"]
    if url:
        import io, zipfile, urllib.request
        w.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(w)
    missing = [p.name for p in need if not p.exists()]
    if missing:
        print("重みが見つからない:", missing, "\n教員配布の assets/weights/ を配置するか，config の day1.weights_url を設定すること")
        return 1
    print("weights:", [p.name for p in need])
    return 0


if __name__ == "__main__":
    sys.exit(main())
