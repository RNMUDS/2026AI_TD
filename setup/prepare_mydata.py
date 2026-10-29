#!/usr/bin/env python3
"""第4回：自分で撮った写真を学習に使える形にそろえる（元の写真は書き換えない）．

  python setup/prepare_mydata.py          # assets/myphotos/<クラス名>/ → assets/mydata/（演習1・2）
  python setup/prepare_mydata.py --new    # assets/mynew_photos/<クラス名>/ → assets/mynew/（演習3 の新しい写真）
  python setup/prepare_mydata.py --vit    # assets/myphotos_vit/<クラス名>/ → assets/mydata_vit/（第5回 オプション課題）

HEIC（iPhone）は JPEG に変換し，長い辺を 256 画素に縮める．枚数が足りなければ理由を出して止まる．
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ASSETS, mydata  # noqa: E402

if __name__ == "__main__":
    if "--vit" in sys.argv:
        print("ViT 用に集めた写真をそろえます")
        mydata.prepare(ASSETS / "myphotos_vit", ASSETS / "mydata_vit")
    elif "--new" in sys.argv:
        print("演習3 の新しい写真をそろえます")
        mydata.prepare(ASSETS / "mynew_photos", ASSETS / "mynew", min_per_class=1, good_per_class=5)
    else:
        print("自分の写真をそろえます")
        mydata.prepare()
