#!/usr/bin/env python3
"""データを assets/ に取得する（第1回は FashionMNIST．第2回からはコーパスと学習済み重みも）．

  python setup/download_assets.py

コーパスは config/course.yaml の指定に従い HuggingFace Hub から取得し，語彙とテンソルを決定的に生成する．
重みは day1.weights_url が空なら，リポジトリに同梱された assets/weights/ をそのまま使う．
進行状況を逐次表示し，最後に「完了」または「失敗」と理由を明示する．
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ASSETS, load_config, data  # noqa: E402


def mb(p: Path) -> str:
    return f"{p.stat().st_size / 1e6:.1f} MB"


def main():
    print("データの準備を開始します", flush=True)
    cfg = load_config()
    c = cfg["corpus"]
    out = data.corpus_dir(cfg)
    npz, vj = out / "encoded.npz", out / "vocab.json"

    week2 = (ASSETS / "weights").exists()      # 第2回の配布（学習済み重み）があるときだけコーパスと重みを扱う
    # ---- 1) コーパス（第2回の演習2 で使う）
    t0 = time.perf_counter()
    if not week2:
        print("[1/3] コーパス … 第2回で取得する（今回はスキップ）", flush=True)
    elif npz.exists() and vj.exists():
        print(f"[1/3] コーパス（{c['name']}）は取得済み ... スキップ", flush=True)
    else:
        print(f"[1/3] コーパス（{c['name']}）を HuggingFace Hub から取得中（約 5MB，1〜2 分） ...", flush=True)
        try:
            data.prepare_corpus(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"      失敗: {type(e).__name__}: {str(e)[:200]}", flush=True)
            print("\n===== 失敗 =====")
            print("コーパスを取得できませんでした．ネットワーク（学内の制限，Hugging Face Hub への接続）を確認し，"
                  "自宅回線でやり直してください．解決しない場合は授業当日に教員が配布します．")
            return 1
        print(f"      完了 ({time.perf_counter() - t0:.1f} 秒)", flush=True)
    if week2:
        d = dict(__import__("numpy").load(npz))
        vocab_n = len(json.loads(vj.read_text()))
        print(f"      訓練 {len(d['X_train']):,} 文 / 検証 {len(d['X_dev']):,} 文 / テスト {len(d['X_test']):,} 文 "
              f"/ 語彙 {vocab_n:,} 語 / 系列長 {d['X_test'].shape[1]}  （{npz.name} {mb(npz)}）", flush=True)

    # ---- 2) FashionMNIST（GW3 の教科書コード用．torchvision が assets/data に取得する，約 30MB）
    t0 = time.perf_counter()
    fm = ASSETS / "data" / "FashionMNIST" / "raw"
    if fm.exists() and any(fm.glob("*-ubyte")):
        print("[2/3] FashionMNIST は取得済み ... スキップ", flush=True)
    else:
        print("[2/3] FashionMNIST（画像 7 万枚，約 30MB）を取得中（1〜2 分） ...", flush=True)
        try:
            from torchvision import datasets
            datasets.FashionMNIST(root=str(ASSETS / "data"), train=True, download=True)
            datasets.FashionMNIST(root=str(ASSETS / "data"), train=False, download=True)
        except Exception as e:  # noqa: BLE001
            print(f"      失敗: {type(e).__name__}: {str(e)[:200]}", flush=True)
            print("\n===== 失敗 =====")
            print("FashionMNIST を取得できませんでした．ネットワークを確認して再実行してください．")
            return 1
        print(f"      完了 ({time.perf_counter() - t0:.1f} 秒)", flush=True)
    print(f"      訓練 60,000 枚 / テスト 10,000 枚（28×28 グレースケール，10 クラス）", flush=True)

    cable = ASSETS / "cable1"
    n_img = len(list(cable.glob("class*/*.jpg")))
    print(f"      オリジナル画像（assets/cable1）: {n_img} 枚" + ("" if n_img else "  ← 見つからない．git pull する"), flush=True)
    if not week2:
        print("[3/3] 学習済み重み … 第2回で配布する（今回はスキップ）", flush=True)
        print("\n===== 完了 =====")
        print("GW3（FashionMNIST とオリジナル画像）の準備ができています．")
        return 0

    # ---- 3) 学習済み重み
    w = ASSETS / "weights"
    url = cfg["day1"].get("weights_url") or ""
    need = [w / f"day1_{a}.pt" for a in ("A1", "A2", "A3")] + [w / "day1_meta.json"]
    if url:
        print("[3/3] 学習済み重みをダウンロード中 ...", flush=True)
        import io, zipfile, urllib.request
        w.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(w)
    else:
        print("[3/3] 学習済み重み（リポジトリに同梱）を確認中 ...", flush=True)
    missing = [p.name for p in need if not p.exists()]
    if missing:
        print(f"      失敗: 見つからないファイル {missing}", flush=True)
        print("\n===== 失敗 =====")
        print("学習済み重みがありません．git pull で最新のリポジトリに更新するか，教員配布の assets/weights/ を配置してください．")
        return 1
    meta = json.loads((w / "day1_meta.json").read_text())
    for a in ("A1", "A2", "A3"):
        s = meta.get("summary", {}).get(a, {})
        acc = s.get("distributed_test_acc")
        print(f"      day1_{a}.pt  {mb(w / f'day1_{a}.pt')}" + (f"  （テスト正解率 {acc:.4f}）" if acc else ""), flush=True)

    print("\n===== 完了 =====")
    print(f"コーパス: {out}")
    print(f"重み    : {w}")
    print("GW3（FashionMNIST）と演習2（学習済み 3 モデルの測定）の準備ができています．")
    return 0


if __name__ == "__main__":
    sys.exit(main())
