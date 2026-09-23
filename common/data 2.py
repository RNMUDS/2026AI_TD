"""コーパスの取得・トークン化・テンソル化．

調達先は config/course.yaml の corpus 節で差し替える（未確定事項 1）．
既定は SST-2 (stanfordnlp/sst2, HuggingFace Hub, parquet)．
分割:
  train : GLUE 版 train 67,349 件（句レベルを含む）→ 学習用
  dev   : GLUE 版 validation 872 件               → 学習中の検証・エポック選択用
  test  : 原 SST の文レベル test 1,821 件（SetFit/sst2 test.jsonl，ラベル付き）→ 最終評価用
GLUE 版の test はラベル非公開のため，test だけ別ソースから取る（config の corpus.test_source）．
train と test の文の完全一致は 1 件のみ（確認済み）．
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from . import ASSETS, load_config

PAD, UNK = 0, 1
_TOK = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")


def tokenize(text: str):
    return _TOK.findall(text.lower())


def corpus_dir(cfg=None):
    cfg = cfg or load_config()
    return ASSETS / "corpus" / cfg["corpus"]["name"]


def download_corpus(cfg=None, force=False):
    """HF Hub から parquet を取得し assets/corpus/<name>/raw_{split}.parquet に置く．"""
    from huggingface_hub import hf_hub_download
    cfg = cfg or load_config()
    c = cfg["corpus"]
    out = corpus_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, fn in c["hf_files"].items():
        dst = out / f"raw_{split}.parquet"
        if dst.exists() and not force:
            paths[split] = dst
            continue
        src = hf_hub_download(repo_id=c["hf_repo"], filename=fn, repo_type="dataset")
        dst.write_bytes(Path(src).read_bytes())
        paths[split] = dst
    ts = c["test_source"]
    dst = out / ("raw_test" + Path(ts["file"]).suffix)
    if not dst.exists() or force:
        src = hf_hub_download(repo_id=ts["hf_repo"], filename=ts["file"], repo_type="dataset")
        dst.write_bytes(Path(src).read_bytes())
    paths["test"] = dst
    return paths


def _read_any(p):
    import pandas as pd
    return pd.read_json(p, lines=True) if p.suffix == ".jsonl" else pd.read_parquet(p)


def build_vocab(texts, vocab_size):
    cnt = Counter(t for s in texts for t in tokenize(s))
    words = [w for w, _ in cnt.most_common(vocab_size - 2)]
    return {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(words)}}


def encode(texts, vocab, max_len):
    X = np.full((len(texts), max_len), PAD, dtype=np.int64)
    for i, s in enumerate(texts):
        ids = [vocab.get(t, UNK) for t in tokenize(s)][:max_len]
        X[i, :len(ids)] = ids
    return X


def prepare_corpus(cfg=None, force=False):
    """raw parquet → 語彙 (vocab.json) と固定長テンソル (encoded.npz)．決定的に生成する．"""
    import pandas as pd
    cfg = cfg or load_config()
    c = cfg["corpus"]
    out = corpus_dir(cfg)
    npz, vj = out / "encoded.npz", out / "vocab.json"
    if npz.exists() and vj.exists() and not force:
        return npz, vj
    paths = download_corpus(cfg, force=force)
    tr = _read_any(paths["train"])
    dv = _read_any(paths["validation"])
    te = _read_any(paths["test"])
    tcol, lcol = c["text_column"], c["label_column"]
    ts = c["test_source"]
    tr_txt = tr[tcol].tolist()
    vocab = build_vocab(tr_txt, int(c["vocab_size"]))
    L = int(c["max_len"])
    data = {
        "X_train": encode(tr_txt, vocab, L), "y_train": tr[lcol].to_numpy().astype(np.int64),
        "X_dev": encode(dv[tcol].tolist(), vocab, L), "y_dev": dv[lcol].to_numpy().astype(np.int64),
        "X_test": encode(te[ts["text_column"]].tolist(), vocab, L),
        "y_test": te[ts["label_column"]].to_numpy().astype(np.int64),
    }
    np.savez_compressed(npz, **data)
    vj.write_text(json.dumps(vocab, ensure_ascii=False))
    # 生テキストも残す（第3回のデータ拡張・第6回のショートカット混入で使う）
    (out / "texts.json").write_text(json.dumps(
        {"train": tr_txt, "dev": dv[tcol].tolist(), "test": te[ts["text_column"]].tolist()}, ensure_ascii=False))
    return npz, vj


def load_corpus(cfg=None):
    """dict(X_train, y_train, X_dev, y_dev, X_test, y_test, vocab, vocab_size, max_len, num_classes)."""
    cfg = cfg or load_config()
    npz, vj = prepare_corpus(cfg)
    d = dict(np.load(npz))
    d["vocab"] = json.loads(vj.read_text())
    d["vocab_size"] = len(d["vocab"])
    d["max_len"] = int(cfg["corpus"]["max_len"])
    d["num_classes"] = int(cfg["corpus"]["num_classes"])
    return d


def decode(ids, vocab):
    inv = {v: k for k, v in vocab.items()}
    return " ".join(inv.get(int(i), "?") for i in ids if int(i) != PAD)
