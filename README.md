# 2026AI_TD ― 人工知能(AI)テクノロジデザインⅠ／Ⅱ 授業用フォルダ

授業ページ: https://rnmuds.github.io/2026AI_TD/ （`docs/` がその元ファイル）

```bash
cd && git clone https://github.com/RNMUDS/2026AI_TD.git AI_TD   # 初回のみ（2回目以降は cd ~/AI_TD && git pull）
cd ~/AI_TD
uv venv --python 3.12              # 初回のみ
source .venv/bin/activate          # ターミナルを開き直すたびに
uv pip install -r requirements.txt # 初回のみ
python setup/check_env.py          # 環境診断
python setup/download_assets.py    # データの取得（FashionMNIST）
code .                             # VS Code で course1/day1/ のノートブックを開き，カーネルに .venv を選ぶ
```

- `common/` … 全14回で使い回す共通基盤（使用デバイスの選択・乱数固定・CSV 記録・モデル定義）
- `course1/dayN/` … 各回のノートブック（ex1.ipynb …）と README
- `results/` … 自分の CSV がここに出る（提出物）．git には含めない
- `docs/` … 授業ページ（GitHub Pages）の元ファイル
