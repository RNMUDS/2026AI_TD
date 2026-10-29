# Ⅰ第5回：転移学習 ― 学習済みのモデルを自分の写真に合わせる

授業ページ：https://rnmuds.github.io/2026AI_TD/aitd1_week_5.html

## 準備（授業の最初に）

```bash
cd ~/AI_TD
git fetch origin
git checkout origin/main -- course1/day5 common setup
source .venv/bin/activate
python setup/download_weights.py   # 学習済みの重み（ResNet18 45MB・ViT-B/16 330MB）
code .
```

写真は第4回の `assets/mydata/` をそのまま使う（無ければケーブル画像で進む）．

## ファイル

| ファイル | 使う場面 |
|---|---|
| `review.ipynb` | 成果発表：第4回の MLP と CNN をグループで発表し，7 つのキーポイントでレビューし合う |
| `ex1.ipynb` | 演習1：第4回の①〜⑫をコピーして直し，ResNet18 を凍結と微調整で学習する（LLM を使わずにやってみる） |
| `ex2.ipynb` | 演習2：CNN（ゼロから）・ResNet18 凍結・微調整・ViT 凍結をシード 0〜4 で回し，差 ÷ 標準偏差 で判定する |
| `summary.ipynb` | 宿題：転移学習を自分で調べて整理し，Google スライドで図解する（PDF にして `c1_d5_summary.pdf`） |
| `opt_vit_transfer.ipynb` | オプション課題（任意）：ViT が得意な画像の仮説を立てて写真を集め（`assets/myphotos_vit/` → `python setup/prepare_mydata.py --vit`），ResNet18 凍結と ViT 凍結を比べる |

## 提出（全員が個人で）

Open-LMS の「第5回 課題提出」へ zip にせずアップロード：`review.ipynb`，`ex1.ipynb`，`ex2.ipynb`，`c1_d5_ex1.csv`，`c1_d5_ex2.csv`，宿題の `summary.ipynb`，`c1_d5_summary.pdf`．写真は出さない．

オプション課題（任意）は `opt_vit_transfer.ipynb` と `c1_d5_opt5.csv` を同じ所へ．

期限：次回（11/5）の授業開始まで
