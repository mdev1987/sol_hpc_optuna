#!/usr/bin/env python3
"""Kaggle runner for the flow 5000-trial study (gdown download).

Make the Drive files shareable ("Anyone with the link"), copy each file ID
from its share URL, then from the repo root in a notebook cell:

    import os
    os.environ["KAGGLE_FEATURES_ID"] = "<file id of cache_features.zip>"
    os.environ["KAGGLE_DB_ID"]       = "<file id of optuna.db>"   # optional
    %run scripts/kaggle_flow_run.py

Env knobs (all optional):
    KAGGLE_FEATURES_ID   Drive file id of cache_features.zip (required to download)
    KAGGLE_DB_ID         Drive file id of optuna.db
    KAGGLE_SELECTED_ID   Drive file id of selected_features.json
    KAGGLE_TRIALS        trials to run (default 50 -- small for the Kaggle test)
    KAGGLE_WORKERS       worker processes (default os.cpu_count(); Kaggle free ~4)
    KAGGLE_BUNDLE        bundle (default flow)
    KAGGLE_SAMPLE        sample_fraction (default 1.0)
    KAGGLE_FRESH         "1" archives any existing optuna.db and starts clean
                         (default 1: the old db is 0.3-sample ablation)
    KAGGLE_UPLOAD        "1" uploads results back via rclone (default 1; needs
                         a configured gdrive remote, else skipped with a note)
    KAGGLE_REMOTE        rclone remote:path used only for upload-back
                         (default gdrive:sol_optuna_hpc)
    RCLONE_CONF          contents of ~/.config/rclone/rclone.conf, written to
                         /root/.config/rclone/rclone.conf (only needed for upload)

Download uses gdown (no auth if files are link-shared). If gdown is missing it
falls back to rclone for files it can find in KAGGLE_REMOTE.

Flow: download features.zip + optuna.db -> extract features.parquet
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

FEATURES_ID = os.environ.get("KAGGLE_FEATURES_ID", "")
DB_ID = os.environ.get("KAGGLE_DB_ID", "")
SELECTED_ID = os.environ.get("KAGGLE_SELECTED_ID", "")
TRIALS = int(os.environ.get("KAGGLE_TRIALS", "50"))
WORKERS = int(os.environ.get("KAGGLE_WORKERS", str(max(1, os.cpu_count() or 1))))
BUNDLE = os.environ.get("KAGGLE_BUNDLE", "flow")
SAMPLE = os.environ.get("KAGGLE_SAMPLE", "1.0")
FRESH = os.environ.get("KAGGLE_FRESH", "1") == "1"
UPLOAD = os.environ.get("KAGGLE_UPLOAD", "1") == "1"
REMOTE = os.environ.get("KAGGLE_REMOTE", "gdrive:sol_optuna_hpc")

ROOT = Path.cwd()
CACHE = ROOT / "cache"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
DB = ROOT / "optuna.db"
STUDY = f"replay_optuna_{BUNDLE}"
SYNC = ROOT / "gdrive_sync"
DEST = f"{REMOTE}/{BUNDLE}_kaggle_{time.strftime('%Y%m%d-%H%M%S')}"


def sh(cmd: str, **kw) -> subprocess.CompletedProcess:
    kw.setdefault("text", True)
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


def rclone_ready() -> bool:
    import shutil

    return shutil.which("rclone") is not None


def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        sh("pip install -q gdown")
    import gdown  # noqa: F401

    return gdown


def gdown_get(gdown, file_id: str, output: Path) -> None:
    if not file_id:
        return False
    print(f"downloading id {file_id} -> {output}", flush=True)
    gdown.download(id=file_id, output=str(output), quiet=False)
    if not output.exists():
        raise SystemExit(f"gdown failed for id {file_id}; the file must be "
                         f"shared as 'Anyone with the link'")
    return True


def download_data() -> None:
    gdown = ensure_gdown()
    CACHE.mkdir(exist_ok=True)
    SYNC.mkdir(exist_ok=True)

    ok = gdown_get(gdown, FEATURES_ID, SYNC / "cache_features.zip")
    if not ok and rclone_ready():
        print("falling back to rclone download", flush=True)
        sh(f"rclone copy {REMOTE}/cache_features.zip {SYNC}")
        ok = (SYNC / "cache_features.zip").exists()
    if not ok:
        raise SystemExit("need cache_features.zip: set KAGGLE_FEATURES_ID "
                         "(or configure rclone and set KAGGLE_REMOTE)")

    db_target = SYNC / "optuna.db"
    if not gdown_get(gdown, DB_ID, db_target) and rclone_ready():
        sh(f"rclone copy {REMOTE}/optuna.db {SYNC}")
    if db_target.exists():
        db_target.rename(DB)

    sel_target = CACHE / "selected_features.json"
    if not gdown_get(gdown, SELECTED_ID, sel_target) and rclone_ready():
        sh(f"rclone copy {REMOTE}/selected_features.json {CACHE}")

    if DB_ID and not DB.exists():
        print("note: optuna.db not downloaded; starting a brand-new study", flush=True)


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
            import shutil

            shutil.copyfileobj(src, dst, 64 * 1024 * 1024)
    print(f"features.parquet: {target.stat().st_size / 2**30:.2f} GiB", flush=True)


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
            print(f"[{time.strftime('%H:%M:%S')}] progress: {n}/{TRIALS} trials "
                  f"({n * 100 // TRIALS}%)", flush=True)
            last = n
    rc = proc.wait()
    final = count_complete()
    print(f"optimize exited rc={rc}, trials={final}/{TRIALS}", flush=True)
    print(f"full log: {log}", flush=True)
    return 0 if (rc == 0 and final >= TRIALS) else 1


def upload_back(status: str, rc: int) -> bool:
    REPORT_DIR.mkdir(exist_ok=True)
    status_file = LOG_DIR / f"final_status_{BUNDLE}.txt"
    status_file.write_text(
        f"study: {STUDY}\noutcome: {status}\ncompleted trials: {count_complete()}/{TRIALS}\n"
        f"rc: {rc}\nfinished: {time.strftime('%Y-%m-%d %H:%M:%S')}\nbundle: {BUNDLE}\n"
    )
    if not UPLOAD:
        print("KAGGLE_UPLOAD=0, skipping upload", flush=True)
        return True
    if not rclone_ready():
        print(f"rclone not installed; results are in /kaggle/working: "
              f"{DB}, {REPORT_DIR}, {LOG_DIR} (download via the sidebar)", flush=True)
        return True

    conf = Path("/root/.config/rclone/rclone.conf")
    if not conf.exists():
        body = os.environ.get("RCLONE_CONF")
        if body:
            conf.parent.mkdir(parents=True, exist_ok=True)
            conf.write_text(body)
        else:
            print(f"rclone has no gdrive config; results are in /kaggle/working: "
                  f"{DB}, {REPORT_DIR}, {LOG_DIR}", flush=True)
            return True

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
        raise SystemExit(
            f"Python >= 3.11 required (have {sys.version_info.major}.{sys.version_info.minor})")
    download_data()
    extract_features()
    prepare_study()
    ok = run_optimize() == 0
    status = "COMPLETE" if ok else "FAILED"
    uploaded = upload_back(status, 0 if ok else 1)
    print(f"=== done: outcome={status} upload={'OK' if uploaded else 'FAILED'} ===", flush=True)
    return 0 if (ok and uploaded) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted; partial results are in the local db", flush=True)
        sys.exit(130)
