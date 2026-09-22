"""共通学習ループ．DataLoader を使わず，デバイス上のテンソルを直接スライスする
（num_workers=0 / pin_memory=False の制約下で MPS 上ではこれが最速）．

進捗は log_path に '[progress] epoch=i/N ...' の形で追記し，runner.check_progress が読む．
"""
import math
import time

import torch
import torch.nn as nn

from .device import synchronize


@torch.no_grad()
def evaluate(model, X, y, batch=512):
    """(accuracy, mean_loss) を返す．X, y はデバイス上のテンソル．"""
    model.eval()
    lossf = nn.CrossEntropyLoss(reduction="sum")
    correct, total_loss = 0, 0.0
    for b in range(0, len(X), batch):
        out = model(X[b:b + batch])
        total_loss += lossf(out, y[b:b + batch]).item()
        correct += (out.argmax(1) == y[b:b + batch]).sum().item()
    return correct / len(X), total_loss / len(X)


def train_model(model, Xtr, ytr, Xva, yva, *, epochs, lr, batch=128, weight_decay=0.0,
                device="cpu", log_path=None, max_sec=None, grad_hook=None, epoch_hook=None, verbose=True):
    """戻り値: history = [{"epoch","train_loss","train_acc","val_loss","val_acc","sec"}...]"""
    model.to(device)
    Xtr, ytr, Xva, yva = Xtr.to(device), ytr.to(device), Xva.to(device), yva.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    nb = max(1, math.ceil(n / batch))
    hist, t0 = [], time.perf_counter()

    def emit(msg):
        if verbose:
            print(msg, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        run_loss, run_correct = 0.0, 0
        for b in range(nb):
            idx = perm[b * batch:(b + 1) * batch]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            if grad_hook is not None:
                grad_hook(model, ep, b)
            opt.step()
            run_loss += loss.item() * len(idx)
            run_correct += (out.argmax(1) == yb).sum().item()
        synchronize(device)
        va_acc, va_loss = evaluate(model, Xva, yva)
        el = time.perf_counter() - t0
        rec = {"epoch": ep, "train_loss": run_loss / n, "train_acc": run_correct / n,
               "val_loss": va_loss, "val_acc": va_acc, "sec": el}
        hist.append(rec)
        if epoch_hook is not None:
            epoch_hook(hist)
        emit(f"[progress] epoch={ep}/{epochs} train_loss={rec['train_loss']:.4f} "
             f"train_acc={rec['train_acc']:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f} elapsed={el:.1f}")
        if max_sec is not None and el > max_sec:
            emit(f"[progress] stopped: time_limit at epoch {ep}")
            break
    emit("[done]")
    return hist
