#!/usr/bin/env python3
"""Kaggle runner for the flow 5000-trial study.

Paste into a Kaggle notebook cell (or run as a script) from the repo root:

    import os
    os.environ["KAGGLE_REMOTE"] = "gdrive:sol_optunal_hpc"   # your Drive folder
    os.environ["RCLONE_CONF"] = "... contents of ~/.config/rclone/rclone.conf ..."
    %run scripts/kaggle_flow_run.py

Env knobs (all optional):
    KAGGLE_REMOTE      rclone remote:path of your Drive folder
                       (default gdrive:sol_optuna_hpc)
    KAGGLE_TRIALS      trials to run (default 50 -- small for the Kaggle test)
    KAGGLE_WORKERS     worker processes (default os.cpu_count(); Kaggle free ~4)
    KAGGLE_BUNDLE      bundle (default flow)
    KAGGLE_SAMPLE      sample_fraction (default 1.0)
    KAGGLE_FRESH       "1" archives any existing optuna.db and starts clean
                       (default 1, recommended: the old db is 0.3-sample ablation)
    KAGGLE_UPLOAD      "1" uploads results back to KAGGLE_REMOTE/<prefix> (default 1)
    KAGGLE_PREFIX      upload folder prefix (default kaggle_run)
    RCLONE_CONF        contents of rclone.conf; written to /root/.config/rclone/rclone.conf

Flow: preflight -> rclone download (features.zip, optuna.db, selected_features.json)
      -> fresh study -> optimize (streaming progress) -> upload back -> verify.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REMOTE = os.environ.get("KAGGLE_REMOTE", "gdrive:sol_optuna_hpc")
TRIALS = int(os.environ.get("KAGGLE_TRIALS", "50"))
WORKERS = int(os.environ.get("KAGGLE_WORKERS", str(max(1, os.cpu_count() or 1))))
BUNDLE = os.environ.get("KAGGLE_BUNDLE", "flow")
SAMPLE = os.environ.get("KAGGLE_SAMPLE", "1.0")
FRESH = os.environ.get("KAGGLE_FRESH", "1") == "1"
UPLOAD = os.environ.get("KAGGLE_UPLOAD", "1") == "1"
PREFIX = os.environ.get("KAGGLE_PREFIX", "kaggle_run")

ROOT = Path.cwd()
CACHE = ROOT / "cache"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
DB = ROOT / "optuna.db"
STUDY = f"replay_optuna_{BUNDLE}"
SYNC = ROOT / "gdrive_sync"
DEST = f"{REMOTE}/{PREFIX}_{time.strftime('%Y%m%d-%H%M%S')}"


def sh(cmd: str, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("text", True)
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def ensure_rclone() -> None:
    if shutil_which("rclone"):
        print(f"rclone present at {shutil_which('rclone')}", flush=True)
        return
    sh("curl -sSLo /tmp/rclone.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip")
    sh("unzip -oq /tmp/rclone.zip -d /tmp/rclone_extract")
    sh("cp /tmp/rclone_extract/rclone-*-linux-amd64/rclone /usr/local/bin/")
    sh("rclone version")


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def configure_rclone() -> None:
    conf_dir = Path("/root/.config/rclone")
    conf = conf_dir / "rclone.conf"
    if conf.exists():
        return
    conf_body = os.environ.get("RCLONE_CONF")
    if conf_body:
        conf_dir.mkdir(parents=True, exist_ok=True)
        conf.write_text(conf_body)
        return
    if (ROOT / "rclone.conf").exists():
        conf_dir.mkdir(parents=True, exist_ok=True)
        sh(f"cp {ROOT / 'rclone.conf'} {conf}")
        return
    raise SystemExit(
        "rclone.conf not found. Set RCLONE_CONF to the contents of your "
        "~/.config/rclone/rclone.conf (from the VPS) or place it at ./rclone.conf"
    )


def download_data() -> None:
    SYNC.mkdir(exist_ok=True)
    listed = sh(f"rclone lsf {REMOTE}", capture_output=True).stdout
    print(f"Drive folder contains:\n{listed}", flush=True)
    if "cache_features.zip" not in listed:
        raise SystemExit(f"'cache_features.zip' not found in {REMOTE}")
    sh(f"rclone copy {REMOTE}/cache_features.zip {SYNC} --progress")
    if "optuna.db" in listed:
        sh(f"rclone copy {REMOTE}/optuna.db {SYNC}")
    if "selected_features.json" in listed:
        sh(f"rclone copy {REMOTE}/selected_features.json {CACHE}")


def extract_features() -> None:
    CACHE.mkdir(exist_ok=True)
    target = CACHE / "features.parquet"
    with zipfile.ZipFile(SYNC / "cache_features.zip") as zf:
        names = zf.namelist()
        candidate = next((n for n in names if n.endswith(".parquet")), None)
        if candidate is None:
            raise SystemExit("no .parquet found inside cache_features.zip")
        print(f"extracting {candidate} -> {target}", flush=True)
        with zf.open(candidate) as src, open(target, "wb") as dst:
            shutil_copyfileobj(src, dst)
    print(f"features.parquet: {target.stat().st_size / 2**30:.2f} GiB", flush=True)


def shutil_copyfileobj(src, dst, length: int = 64 * 1024 * 1024) -> None:
    import shutil

    shutil.copyfileobj(src, dst, length)


def prepare_study() -> None:
    if FRESH and DB.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        DB.rename(DB.with_name(f"{DB.name}.pre-final-{stamp}"))
        print(f"archived previous db -> optuna.db.pre-final-{stamp}", flush=True)
        stale = REPORT_DIR / f"best_strategy_{BUNDLE}.json"
        if stale.exists():
            stale.unlink()


def count_complete() -> int:
    try:
        con = sqlite3.connect(DB, timeout=30)
        row = con.execute(
            "SELECT COUNT(*) FROM trials t JOIN studies s ON t.study_id = s.study_id "
            "WHERE s.study_name = ? AND t.state = 'COMPLETE'",
            (STUDY,),
        ).fetchone()
        con.close()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def run_optimize() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    log = LOG_DIR / f"final_run_{BUNDLE}.log"
    cmd = [
        sys.executable, "hpc_replay.py", "optimize",
        "--trials", str(TRIALS), "--workers", str(WORKERS), "--resume",
        "--sample-fraction", SAMPLE, "--bundle", BUNDLE,
    ]
    print(" ".join(cmd), flush=True)
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    last = -1
    while proc.poll() is None:
        time.sleep(60)
        n = count_complete()
        if n != last:
            print(f"[{time.strftime('%H:%M:%S')}] progress: {n}/{TRIALS} trials ({n * 100 // TRIALS}%)", flush=True)
            last = n
    rc = proc.wait()
    final = count_complete()
    print(f"optimize exited rc={rc}, trials={final}/{TRIALS}", flush=True)
    print(f"full log: {log}", flush=True)
    return 0 if (rc == 0 and final >= TRIALS) else 1


def upload_back(status: str, rc: int) -> bool:
    if not UPLOAD:
        print("KAGGLE_UPLOAD=0, skipping upload", flush=True)
        return True
    REPORT_DIR.mkdir(exist_ok=True)
    status_file = LOG_DIR / f"final_status_{BUNDLE}.txt"
    status_file.write_text(
        f"study: {STUDY}\noutcome: {status}\ncompleted trials: {count_complete()}/{TRIALS}\n"
        f"rc: {rc}\nfinished: {time.strftime('%Y-%m-%d %H:%M:%S')}\nbundle: {BUNDLE}\n"
    )
    assets = [str(DB), str(REPORT_DIR), str(status_file), str(LOG_DIR / f"final_run_{BUNDLE}.log")]
    selected = CACHE / "selected_features.json"
    if selected.exists():
        assets.append(str(selected))
    ok = False
    for attempt in range(1, 6):
        print(f"upload attempt {attempt}/5 -> {DEST}", flush=True)
        try:
            sh("rclone copy --progress " + " ".join(f"'{a}'" for a in assets) + f" '{DEST}'")
        except subprocess.CalledProcessError:
            time.sleep(30)
            continue
        listed = sh(f"rclone lsf {DEST}", capture_output=True).stdout
        expect = [DB.name, f"final_status_{BUNDLE}.txt", f"final_run_{BUNDLE}.log"]
        if (REPORT_DIR / f"best_strategy_{BUNDLE}.json").exists():
            expect.append(f"best_strategy_{BUNDLE}.json")
        if (REPORT_DIR / f"final_summary_{BUNDLE}.txt").exists():
            expect.append(f"final_summary_{BUNDLE}.txt")
        if selected.exists():
            expect.append("selected_features.json")
        if all(e in listed for e in expect):
            ok = True
            break
        print("files missing on remote, retrying...", flush=True)
        time.sleep(30)
    print(f"upload {'verified' if ok else 'FAILED after 5 attempts'} -> {DEST}", flush=True)
    return ok


def main() -> int:
    if sys.version_info < (3, 11):
        raise SystemExit(f"Python >= 3.11 required (have {sys.version_info.major}.{sys.version_info.minor})")
    ensure_rclone()
    configure_rclone()
    download_data()
    extract_features()
    prepare_study()
    ok = run_optimize() == 0
    status = "COMPLETE" if ok else "FAILED"
    uploaded = upload_back(status, 0 if ok else 1)
    print(f"=== done: outcome={status} upload={'OK' if uploaded else 'FAILED'} dest={DEST} ===", flush=True)
    return 0 if (ok and uploaded) else 1


if __name__ == "__main__":
    import shutil

    try:
        sys.exit(main())
    except SystemExit as e:
        raise
    except KeyboardInterrupt:
        print("interrupted; partial results are in the local db", flush=True)
        sys.exit(130)
