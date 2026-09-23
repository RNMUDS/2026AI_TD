"""メモリ上限の強制とピーク使用量の記録．

API はインストール済み torch 2.8.0 で実物確認済み:
  torch.mps.set_per_process_memory_fraction(fraction) -> None   (ハード上限)
  torch.mps.recommended_max_memory() -> int
  torch.mps.current_allocated_memory() -> int   (テンソルが占める量)
  torch.mps.driver_allocated_memory() -> int    (ドライバが確保した量．キャッシュ込み)
  ※ torch.mps に max_memory_allocated 相当は存在しない → サンプリングでピークを採る
"""
import threading

LIMIT_GB = 11.0
LIMIT_BYTES = int(LIMIT_GB * 1024 ** 3)


def cap_torch(limit_bytes=LIMIT_BYTES):
    """torch MPS を limit_bytes 相当に制限する．16GB 機では recommended_max が約 10.6GB のため実質無変更．"""
    import torch
    if not torch.backends.mps.is_available():
        return {"capped": False, "reason": "mps unavailable"}
    rec = int(torch.mps.recommended_max_memory())
    frac = min(1.0, limit_bytes / rec)
    torch.mps.set_per_process_memory_fraction(frac)
    return {"capped": True, "recommended_max_gb": round(rec / 1024 ** 3, 2),
            "fraction": round(frac, 4), "effective_limit_gb": round(rec * frac / 1024 ** 3, 2)}


def _mps_allocated():
    import torch
    return int(torch.mps.current_allocated_memory())


def _mps_driver():
    import torch
    return int(torch.mps.driver_allocated_memory())


def _rss():
    import psutil, os
    return int(psutil.Process(os.getpid()).memory_info().rss)


class PeakSampler:
    """with ブロック内のピークメモリを採取する．

    kind='allocated' : MPS 上でテンソルが占める量（モデル比較に向く）
    kind='driver'    : MPS ドライバ確保量（キャッシュ込み．上限判定に向く）
    kind='rss'       : プロセスの常駐メモリ（CPU フォールバック時）
    """

    def __init__(self, kind="allocated", interval=0.02):
        import torch
        if kind in ("allocated", "driver") and not torch.backends.mps.is_available():
            kind = "rss"
        self.kind = kind
        self.fn = {"allocated": _mps_allocated, "driver": _mps_driver, "rss": _rss}[kind]
        self.interval = interval
        self.start = 0
        self.peak = 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            v = self.fn()
            if v > self.peak:
                self.peak = v
            self._stop.wait(self.interval)

    def __enter__(self):
        self.start = self.peak = self.fn()
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)
        self.peak = max(self.peak, self.fn())
        return False

    @property
    def peak_mb(self):
        return round(self.peak / 1024 ** 2, 2)

    @property
    def delta_mb(self):
        """ブロック開始時点からの増分．"""
        return round((self.peak - self.start) / 1024 ** 2, 2)

    @property
    def peak_gb(self):
        return round(self.peak / 1024 ** 3, 3)


def over_limit(peak_bytes, limit_bytes=LIMIT_BYTES):
    return peak_bytes >= limit_bytes


def forward_peak_mb(model, run):
    """run() を実行し，順伝播の各モジュール出力直後に読んだ使用量の最大値 − 開始時 を MB で返す．

    バックグラウンドのサンプリング（PeakSampler）は数十 ms の処理では取りこぼして実行ごとにぶれるが，
    この方式はフックで同期的に読むため決定的（実測: 3 回実行で完全一致）．
    MPS では torch.mps.current_allocated_memory()，CPU では RSS を読む．
    """
    import torch
    on_mps = any(p.device.type == "mps" for p in model.parameters())
    fn = _mps_allocated if on_mps else _rss
    peak = [0]

    def hook(mod, inp, out):
        v = fn()
        if v > peak[0]:
            peak[0] = v

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    base = fn()
    try:
        run()
        if on_mps:
            torch.mps.synchronize()
        peak[0] = max(peak[0], fn())
    finally:
        for h in handles:
            h.remove()
    return round((peak[0] - base) / 1024 ** 2, 2)
