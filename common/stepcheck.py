"""第4回・第5回 演習1：自分で書いたコードテンプレート①〜⑫を 1 つずつ確かめる．

    check("③", globals())
    check("②", globals(), size=224)   # 第5回（学習済みモデルに合わせて 224×224）

足りない名前や形の違いがあれば，どのカード（第1回 GW3）を見直せばよいかを添えて止める．
"""
import torch

HINT = {
    "①": "カード L・O（ライブラリのインポート）",
    "②": "カード Q（64×64 にそろえる前処理）．第5回は 224×224 と Normalize",
    "③": "カード P（フォルダから読み，3 つに分ける）．IMAGE_DIR は (1) で決めたものを使う",
    "④": "カード N（データローダ．テスト用も作る）",
    "⑤": "カード D（使用デバイスの選択）",
    "⑥": "カード C（ミニバッチの形の確認）",
    "⑦": "MLP はカード M，CNN はカード R．第5回は ex1 の「新しく使う書き方」．OUTPUT_SIZE（第5回は NUM_CLASSES）はクラスの数",
    "⑧⑨": "カード A（損失関数）と J（最適化手法）",
    "⑩": "カード E（学習）．EPOCHS = 20（第5回の微調整は 10）",
    "⑫": "カード H（テストデータでの評価）",
}


def _need(ns, names, step):
    missing = [n for n in names if n not in ns]
    if missing:
        raise RuntimeError(f"{step}：{', '.join(missing)} がまだ作られていない．見直す所：{HINT[step]}")


def _fail(step, msg):
    raise RuntimeError(f"{step}：{msg}．見直す所：{HINT[step]}")


def _check_transform(ns, size):
    tf = ns.get("data_transforms", {}).get("val") if isinstance(ns.get("data_transforms"), dict) else None
    if tf is None:
        _fail("②", "data_transforms = {'train': ..., 'val': ...} の形で書く")
    from PIL import Image
    x = tf(Image.new("RGB", (300, 200)))
    if tuple(x.shape) != (3, size, size):
        _fail("②", f"前処理のあとの形が {tuple(x.shape)}．(3, {size}, {size}) になるようにする")
    if size == 224 and not any(type(t).__name__ == "Normalize" for t in getattr(tf, "transforms", [])):
        _fail("②", "ToTensor() のあとに transforms.Normalize(...) を足す（学習済みモデルの入力に合わせる）")


def _check_split(ns):
    full = ns["full_dataset"]
    if str(full.root) != str(ns.get("IMAGE_DIR")):
        _fail("③", f"読み込んだフォルダが {full.root}．(1) の IMAGE_DIR を使う")
    sizes = [len(ns[n]) for n in ("train_dataset", "val_dataset", "test_dataset")]
    if sum(sizes) != len(full):
        _fail("③", f"3 つの合計 {sum(sizes)} 枚が全体 {len(full)} 枚と合わない")
    print(f"クラス {full.classes}｜訓練 {sizes[0]}・検証 {sizes[1]}・テスト {sizes[2]} 枚")


def _check_model(ns, size):
    model, device = ns["model"], ns["device"]
    n_class = len(ns["full_dataset"].classes)
    try:
        with torch.no_grad():
            out = model(torch.randn(1, 3, size, size).to(device))
    except RuntimeError as e:
        _fail("⑦", f"(1, 3, {size}, {size}) の画像が通らない（{str(e).splitlines()[0][:80]}）．INPUT_SIZE・FLATTEN_SIZE は c・h・w から計算する")
    if tuple(out.shape) != (1, n_class):
        _fail("⑦", f"出力の形が {tuple(out.shape)}．(1, {n_class}) になるようにする（クラスは {n_class} 個）")
    kind = ("ResNet" if hasattr(model, "fc") and hasattr(model, "layer1") else
            "CNN" if any(isinstance(m, torch.nn.Conv2d) for m in model.modules()) else "MLP")
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trained = "" if n_train == n_params else f"（うち学習する数 {n_train:,}）"
    print(f"{kind}｜出力の形 {tuple(out.shape)}｜パラメータ数 {n_params:,}{trained}")


# 第5回（size=224）で止まったときに足すヒント（ほぼ答え）
HINT_W5 = {
    "①": "①の最後に from torchvision import models を足す．ほかは第4回 ex1 の①をコピーする",
    "②": "'train' と 'val' の両方を transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), "
         "transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) にする",
    "③": "第4回 ex1 の③をそのままコピーする（IMAGE_DIR の行は書かない）",
    "④": "第4回 ex1 の④をそのままコピーする",
    "⑤": "第4回 ex1 の⑤をそのままコピーする",
    "⑥": "第4回 ex1 の⑥をそのままコピーする．②を実行し直してから④⑥を実行し直す",
    "⑦": "NUM_CLASSES = len(full_dataset.classes)，model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) で読み，"
         "num_ftrs = model.fc.in_features，model.fc = nn.Linear(num_ftrs, NUM_CLASSES) で最終層を替え，model = model.to(device)",
    "⑧⑨": "凍結は optimizer = optim.Adam(model.fc.parameters(), lr=0.001)，微調整は optimizer = optim.Adam(model.parameters(), lr=0.0001)．"
           "⑦を実行し直したら⑧⑨も実行し直す",
    "⑩": "第4回 ex1 の⑩をコピーし，凍結は EPOCHS = 20，微調整は EPOCHS = 10 にする",
    "⑫": "第4回 ex1 の⑫をそのままコピーする",
}


def check(step, ns, size=64):
    """ns には globals() を渡す．size は②でそろえる画像の大きさ（第5回は 224．止まったときにヒントを足す）．"""
    try:
        _check(step, ns, size)
    except RuntimeError as e:
        if size != 224:
            raise
        raise RuntimeError(f"{e}\nヒント：{HINT_W5[step]}．このテンプレートの上の「ヒント」も開いてみる") from None


def _check(step, ns, size):
    if step == "①":
        _need(ns, ["torch", "nn", "optim", "transforms", "DataLoader", "random_split", "ImageFolder"]
              + (["models"] if size == 224 else []), step)   # 第5回は学習済みモデル用の models も要る
    elif step == "②":
        _check_transform(ns, size)
    elif step == "③":
        _need(ns, ["full_dataset", "train_dataset", "val_dataset", "test_dataset"], step)
        _check_split(ns)
    elif step == "④":
        _need(ns, ["train_loader", "val_loader", "test_loader"], step)
        if len(ns["test_loader"].dataset) != len(ns["test_dataset"]):
            _fail(step, "test_loader に test_dataset を渡す")
    elif step == "⑤":
        _need(ns, ["device"], step)
    elif step == "⑥":
        _need(ns, ["c", "h", "w"], step)
        if (ns["c"], ns["h"], ns["w"]) != (3, size, size):
            _fail(step, f"形が {(ns['c'], ns['h'], ns['w'])}．(3, {size}, {size}) のはず．②の前処理を見直す")
    elif step == "⑦":
        _need(ns, ["model", "device", "full_dataset"], step)
        _check_model(ns, size)
    elif step == "⑧⑨":
        _need(ns, ["criterion", "optimizer"], step)
        mine = {id(p) for p in ns["model"].parameters()}
        given = [p for g in ns["optimizer"].param_groups for p in g["params"]]
        if not all(id(p) in mine for p in given):
            _fail(step, "optimizer に今の model のパラメータを渡す（⑦を実行し直したら⑨も実行し直す）")
    elif step == "⑩":
        _need(ns, ["train_loss_list", "val_loss_list", "train_acc_list", "val_acc_list", "train_sec"], step)
        if len(ns["val_acc_list"]) != ns.get("EPOCHS", -1):
            _fail(step, f"記録が {len(ns['val_acc_list'])} エポック分．EPOCHS 回分そろうように⑩を実行し直す")
    elif step == "⑫":
        _need(ns, ["correct", "total"], step)
        if ns["total"] != len(ns["test_dataset"]):
            _fail(step, "test_loader で数える（total がテストの枚数と合わない）")
    else:
        raise ValueError(f"step は {list(HINT)} のどれか")
    print(f"{step} OK")
