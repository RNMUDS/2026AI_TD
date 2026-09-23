"""A1 / A2 / A3 の定義と，パラメータ総数を揃える build_model．

A1: 埋め込み平均 -> MLP   （位置情報を持たない．語順を捨てる）
A2: 1次元畳み込み          （局所の順序は見えるが受容野が有限）
A3: 小型 Transformer       （位置埋め込み + 自己注意で全域を参照）

PAD = 0 は埋め込みをゼロに固定し（padding_idx），平均・注意から除外する．
"""
import torch
import torch.nn as nn

PAD = 0


class A1(nn.Module):
    def __init__(self, vocab_size, d, h, num_classes=2, max_len=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, num_classes))

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1).to(self.emb.weight.dtype)   # (B,L,1)
        z = (self.emb(x) * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # マスク付き平均
        return self.net(z)


class A2(nn.Module):
    def __init__(self, vocab_size, d, h, num_classes=2, k=5, max_len=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.c1 = nn.Conv1d(d, h, k, padding=k // 2)
        self.c2 = nn.Conv1d(h, h, k, padding=k // 2)
        self.head = nn.Linear(h, num_classes)

    def forward(self, x):
        z = self.emb(x).transpose(1, 2)              # (B,d,L)
        z = torch.relu(self.c1(z))
        z = torch.relu(self.c2(z))
        return self.head(z.max(dim=2).values)        # 位置方向の最大値プーリング


class A3(nn.Module):
    def __init__(self, vocab_size, d, h, num_classes=2, max_len=64, nhead=4, nlayer=2):
        super().__init__()
        d = max(nhead, (d // nhead) * nhead)
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=h,
                                           batch_first=True, dropout=0.0, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayer, enable_nested_tensor=False)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):
        pad = x == PAD                                   # (B,L) True=pad
        p = torch.arange(x.shape[1], device=x.device)
        z = self.emb(x) + self.pos(p)[None]
        z = self.enc(z, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(z.dtype)
        z = (z * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.head(z)


ARCHS = {"A1": A1, "A2": A2, "A3": A3}


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(arch, target_params, vocab_size, num_classes=2, max_len=64, d=32, tol_pct=5.0):
    """隠れ幅 h を二分探索し，総パラメータ数を target_params に ±tol_pct 以内で合わせる．

    戻り値: (model, info)  info = {"arch","hidden","params","target","dev_pct"}
    """
    def make(h):
        return ARCHS[arch](vocab_size, d, h, num_classes=num_classes, max_len=max_len)

    lo, hi, best = 2, 8192, None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_params(make(mid))
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (mid, n)
        if n < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    h, n = best
    dev = 100.0 * (n - target_params) / target_params
    assert abs(dev) <= tol_pct, f"{arch}: {n} params は目標 {target_params} の ±{tol_pct}% を外れる ({dev:.2f}%)"
    model = make(h)
    return model, {"arch": arch, "hidden": h, "params": n, "target": target_params,
                   "dev_pct": round(dev, 3), "embed_dim": d}
