# Ⅰ第1回：AIは何を学習し，どう組み立て，どう測るか

授業ページ：https://rnmuds.github.io/2026AI_TD/aitd1_week_1.html

## 環境構築

```bash
brew install uv
uv python install 3.12
cd && git clone https://github.com/RNMUDS/2026AI_TD.git AI_TD   # 初回のみ（2回目以降は cd ~/AI_TD && git pull）
cd AI_TD
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python setup/check_env.py        # 環境診断
python setup/download_assets.py  # データの取得（FashionMNIST）
code .                           # カーネルに .venv の Python 3.12 を選ぶ
```

## ファイル

| ファイル | 使う場面 |
|---|---|
| `ex0_1_mlp_fashion.ipynb` | GW3 試行①：MLP × FashionMNIST（カード A〜L を①〜⑫に貼る） |
| `ex0_2_mlp_original.ipynb` | GW3 試行②：MLP × オリジナル画像（①と同じカード＋M〜Q） |
| `ex0_3_cnn_original.ipynb` | GW3 試行③：シンプルな CNN × オリジナル画像（②と同じカード＋R）．最後に MLP と CNN の説明課題 |
| `ex1.ipynb` | 演習1：環境ベンチマーク |
| `verify.py` | 提出前の形式チェック |

オリジナル画像は `assets/cable1/`（2 クラス × 100 枚）．

## 提出（全員が個人で）

1. グループで共有する：演習1 の CSV（各自の行），穴埋めシート，失敗記録・討議メモ
2. 自分の `results/` に集める
3. `python course1/day1/verify.py --group <グループ番号>` で OK を確認
4. Open-LMS の「第1回」へ zip にせずアップロード：ex0_1〜ex0_3 の3ファイル，`c1_d1_ex0_<グループ>.csv`，`c1_d1_ex1_<グループ>.csv`，穴埋めシート（PDF）

期限：次回の授業開始まで
