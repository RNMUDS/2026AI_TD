"""バックグラウンド実行と進捗確認．

    job = run_background("python instructor/pretrain_day1.py", name="pretrain")
    check_progress(job)   -> {"status","current_epoch","total_epochs","elapsed","eta","last_line","log"}

学習スクリプトは log に '[progress] epoch=i/N ... elapsed=SEC' を追記する（common.train が行う）．
ジョブ情報は results/jobs/<job_id>.json に置き，別プロセス（別ノートブック）からも確認できる．
"""
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import RESULTS, ROOT

JOBS = RESULTS / "jobs"
_RE = re.compile(r"\[progress\] epoch=(\d+)/(\d+).*?elapsed=([\d.]+)")


def run_background(cmd, name="job", cwd=None, env=None):
    """cmd（文字列）を別プロセスで起動し job_id を返す．標準出力・標準エラーは log に追記される．"""
    JOBS.mkdir(parents=True, exist_ok=True)
    job_id = f"{name}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    log = JOBS / f"{job_id}.log"
    exit_file = JOBS / f"{job_id}.exit"
    shell = f"({cmd}) >> '{log}' 2>&1; echo $? > '{exit_file}'"
    e = dict(os.environ, PYTHONUNBUFFERED="1", DLCOURSE_LOG=str(log))
    if env:
        e.update(env)
    p = subprocess.Popen(["/bin/sh", "-c", shell], cwd=str(cwd or ROOT), env=e,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    meta = {"job_id": job_id, "cmd": cmd, "pid": p.pid, "log": str(log), "exit_file": str(exit_file),
            "start": time.time(), "cwd": str(cwd or ROOT)}
    (JOBS / f"{job_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return job_id


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_progress(job_id):
    meta = json.loads((JOBS / f"{job_id}.json").read_text())
    log = Path(meta["log"])
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    cur = tot = None
    elapsed = time.time() - meta["start"]
    for ln in reversed(lines):
        m = _RE.search(ln)
        if m:
            cur, tot, elapsed = int(m.group(1)), int(m.group(2)), float(m.group(3))
            break
    exit_file = Path(meta["exit_file"])
    if exit_file.exists():
        code = int(exit_file.read_text().strip() or -1)
        status = "done" if code == 0 else f"failed(exit={code})"
    elif _alive(meta["pid"]):
        status = "running"
    else:
        status = "unknown"
    eta = None
    if cur and tot and cur < tot and status == "running":
        eta = round(elapsed / cur * (tot - cur), 1)
    return {"job_id": job_id, "status": status, "current_epoch": cur, "total_epochs": tot,
            "elapsed": round(time.time() - meta["start"], 1), "eta": eta,
            "last_line": lines[-1] if lines else "", "log": str(log)}


def wait(job_id, poll=2.0, timeout=None):
    t0 = time.time()
    while True:
        st = check_progress(job_id)
        if st["status"] != "running":
            return st
        if timeout and time.time() - t0 > timeout:
            return st
        time.sleep(poll)


def list_jobs():
    return sorted(p.stem for p in JOBS.glob("*.json")) if JOBS.exists() else []
