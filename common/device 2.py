"""MPS / CPU の選択．環境変数 DLCOURSE_DEVICE=cpu で CPU を強制できる（フォールバック検証用）．"""
import os
import platform
import subprocess

import torch

from . import memory


def get_device(force=None):
    force = force or os.environ.get("DLCOURSE_DEVICE")
    if force:
        dev = torch.device(force)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if dev.type == "mps":
        memory.cap_torch()
    return dev


def synchronize(device):
    """MPS は非同期実行のため，時間計測の前後で必ず同期する．"""
    if torch.device(device).type == "mps":
        torch.mps.synchronize()


def _sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def machine_string():
    """ログ用の機種文字列．例: 'Apple M3/Mac15,13/24GB'."""
    chip = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    model = _sysctl("hw.model") or platform.node()
    mem = _sysctl("hw.memsize")
    gb = f"{int(mem) // 1024 ** 3}GB" if mem.isdigit() else "?GB"
    return f"{chip}/{model}/{gb}"


def machine_info():
    mem = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string"),
        "hw_model": _sysctl("hw.model"),
        "memory_gb": int(mem) // 1024 ** 3 if mem.isdigit() else None,
        "cpu_cores": int(_sysctl("hw.ncpu") or 0),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }
