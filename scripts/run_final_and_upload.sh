#!/usr/bin/env bash
#
# run_final_and_upload.sh
#
# Run the final full-data Optuna sequence for a single bundle inside tmux
# (a dropped SSH session cannot kill it), write a live progress log, then
# upload all results to Google Drive via rclone when the run finishes.
#
# Designed to be launched on a VPS and left unattended overnight:
#   - preflight checks (rclone remote, data, tools) run BEFORE tmux starts,
#     so a misconfiguration fails immediately instead of next morning;
#   - the upload retries on failure and verifies the files landed on Drive;
#   - even a failed/partial run still uploads whatever results exist, tagged
#     with a COMPLETE/FAILED status file.
#
# Usage:
#   ./scripts/run_final_and_upload.sh [trials] [workers] [bundle] [sample_fraction] [rclone_dest]
#
#   trials           trials to run                (default 5000)
#   workers          worker processes             (default 16)
#   bundle           flow | structure | early_momentum | reduced_full (default flow)
#   sample_fraction  mints kept per trial         (default 1.0 = full data)
#   rclone_dest      rclone remote:path           (default gdrive:sol_optuna_results/)
#
#   Env:
#   FRESH_STUDY=0|1     archive optuna.db (+wal/shm) and the stale bundle
#                       report before starting so every trial uses consistent
#                       full data. Default 1 (recommended after an ablation);
#                       set 0 to resume the existing study. Archiving is a
#                       rename, never a delete.
#   SHUTDOWN_AFTER_UPLOAD=0|1
#                       power off the VPS after any run whose upload was
#                       verified (complete or partial; default 1). Requires
#                       root/passwordless sudo; if it fails the VPS stays on
#                       and the error is logged.
#   EXTRA_RCLONE_FLAGS  extra flags passed to rclone (e.g. --drive-chunk-size 128M)
#   STALL_MINUTES       kill the run if no new COMPLETE trial for this many
#                       minutes and upload whatever exists (default 45). This
#                       is the last-resort watchdog: even if a pathological
#                       trial hangs the optimizer, the VPS still uploads and
#                       powers off instead of waiting forever.
#
# Examples:
#   ./scripts/run_final_and_upload.sh
#   SHUTDOWN_AFTER_UPLOAD=0 ./scripts/run_final_and_upload.sh 5000 16 flow 1.0
#
# Watch progress:   tail -f logs/final_run_<bundle>.log
# Attach:           tmux attach -t optuna-final-<bundle>
# Stop it:          tmux kill-session -t optuna-final-<bundle>

set -euo pipefail

TRIALS="${1:-5000}"
WORKERS="${2:-16}"
BUNDLE="${3:-flow}"
SAMPLE_FRACTION="${4:-1.0}"
RCLONE_DEST="${5:-gdrive:sol_optuna_results/}"
FRESH_STUDY="${FRESH_STUDY:-1}"
SHUTDOWN_AFTER_UPLOAD="${SHUTDOWN_AFTER_UPLOAD:-1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
REPORT_DIR="${ROOT}/reports"
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"
LOG="${LOG_DIR}/final_run_${BUNDLE}.log"
STATUS="${LOG_DIR}/final_status_${BUNDLE}.txt"
SESSION="optuna-final-${BUNDLE}"
STUDY="replay_optuna_${BUNDLE}"
DB="${ROOT}/optuna.db"
REPORT="${REPORT_DIR}/best_strategy_${BUNDLE}.json"
SUMMARY="${REPORT_DIR}/final_summary_${BUNDLE}.txt"
REMOTE="${RCLONE_DEST%%:*}:"   # e.g. "gdrive:"

_log() { echo "[$(date '+%H:%M:%S')] $*"; }

# --- Outer phase: fail-fast preflight, then start the tmux session ----------
if [ -z "${RUN_FINAL_INNER:-}" ]; then

    missing=0
    for bin in tmux uv rclone; do
        if ! command -v "${bin}" >/dev/null 2>&1; then
            echo "ERROR: '${bin}' is not installed."
            missing=1
        fi
    done
    if command -v rclone >/dev/null 2>&1 && ! rclone listremotes 2>/dev/null | grep -qx "${REMOTE}"; then
        echo "ERROR: rclone remote '${REMOTE}' is not configured. Run: rclone config"
        missing=1
    fi
    if [ ! -f "${ROOT}/cache/features.parquet" ]; then
        echo "ERROR: ${ROOT}/cache/features.parquet not found (need the features dataset)."
        missing=1
    fi
    if [ "${missing}" = "1" ]; then
        echo "Preflight failed. Fix the above and re-run. Nothing was started."
        exit 1
    fi

    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "Session '${SESSION}' already running."
        echo "  tail -f ${LOG}"
        echo "  tmux attach -t ${SESSION}"
        exit 1
    fi

    SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
    echo "Preflight OK. Starting final run in tmux session '${SESSION}'..."
    echo "  trials=${TRIALS} workers=${WORKERS} bundle=${BUNDLE} sample_fraction=${SAMPLE_FRACTION}"
    echo "  study=${STUDY} fresh=${FRESH_STUDY}"
    echo "  upload -> ${RCLONE_DEST}"
    echo ""
    echo "The run takes several hours. You can close your SSH session now."
    if [ "${SHUTDOWN_AFTER_UPLOAD}" = "1" ]; then
        echo "After a verified upload the VPS will power off automatically."
    else
        echo "SHUTDOWN_AFTER_UPLOAD=0: the VPS will stay on after uploading."
    fi
    tmux new-session -d -s "${SESSION}" \
        "cd ${ROOT} && RUN_FINAL_INNER=1 FRESH_STUDY=${FRESH_STUDY} SHUTDOWN_AFTER_UPLOAD=${SHUTDOWN_AFTER_UPLOAD} bash ${SCRIPT} ${TRIALS} ${WORKERS} ${BUNDLE} ${SAMPLE_FRACTION} '${RCLONE_DEST}' >> ${LOG} 2>&1"
    echo "Started. Watch progress: tail -f ${LOG}"
    echo "Attach (live, detach with C-b d): tmux attach -t ${SESSION}"
    exit 0
fi

# --- Inner phase -------------------------------------------------------------

# Count COMPLETE trials for the study (stdlib sqlite3, no optuna import).
_count_complete() {
    python3 - "${DB}" "${STUDY}" <<'PY'
import sqlite3, sys
db, study = sys.argv[1], sys.argv[2]
try:
    con = sqlite3.connect(db, timeout=30)
    row = con.execute(
        "SELECT COUNT(*) FROM trials t JOIN studies s ON t.study_id = s.study_id "
        "WHERE s.study_name = ? AND t.state = 'COMPLETE'",
        (study,),
    ).fetchone()
    con.close()
    print(int(row[0]) if row else 0)
except Exception:
    print(0)
PY
}

if [ "${FRESH_STUDY}" = "1" ] && [ -f "${DB}" ]; then
    stamp="$(date +%Y%m%d-%H%M%S)"
    mv "${DB}" "${DB}.pre-final-${stamp}"
    [ -f "${DB}-wal" ] && mv "${DB}-wal" "${DB}-wal.pre-final-${stamp}"
    [ -f "${DB}-shm" ] && mv "${DB}-shm" "${DB}-shm.pre-final-${stamp}"
    rm -f "${REPORT}"
    _log "archived previous db -> optuna.db.pre-final-${stamp} (fresh study)"
fi

_log "=== final run: ${STUDY} ==="
_log "  trials=${TRIALS} workers=${WORKERS} sample_fraction=${SAMPLE_FRACTION}"
_log "  head=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
_log "  log: ${LOG}"
_log "  started $(date '+%Y-%m-%d %H:%M:%S')"

cd "${ROOT}"
uv run python hpc_replay.py optimize \
    --trials "${TRIALS}" --workers "${WORKERS}" --resume \
    --sample-fraction "${SAMPLE_FRACTION}" --bundle "${BUNDLE}" \
    >> "${LOG}" 2>&1 &
RUN_PID=$!

# Watchdog: if the COMPLETE trial count stops advancing for STALL_MINUTES
# (e.g. a pathological parameter set hangs a worker despite the in-code per-
# trial timeout), kill the run and fall through to the upload/shutdown path
# instead of waiting forever on an unattended VPS.
STALL_MINUTES="${STALL_MINUTES:-45}"
last="-1"
last_change="$(date +%s)"
while kill -0 "${RUN_PID}" 2>/dev/null; do
    sleep 60
    count="$(_count_complete)"
    if [ "${count}" != "${last}" ]; then
        pct=$(( count * 100 / TRIALS ))
        _log "[progress] ${count}/${TRIALS} trials (${pct}%)"
        last="${count}"
        last_change="$(date +%s)"
    elif [ $(( $(date +%s) - last_change )) -ge $(( STALL_MINUTES * 60 )) ]; then
        _log "WARNING: no new complete trial for ${STALL_MINUTES}m; killing run (${count}/${TRIALS})."
        kill "${RUN_PID}" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            sleep 2
            kill -0 "${RUN_PID}" 2>/dev/null || break
        done
        kill -9 "${RUN_PID}" 2>/dev/null || true
        sleep 2
        break
    fi
done
rc=0
wait "${RUN_PID}" || rc=$?

final_count="$(_count_complete)"
if [ "${rc}" -eq 0 ] && [ "${final_count}" -ge "${TRIALS}" ]; then
    _log "[done] ${final_count}/${TRIALS} trials completed successfully. Finishing $(date '+%H:%M:%S')"
    outcome="COMPLETE"
    if [ -f "${REPORT}" ]; then
        python3 - "${REPORT}" "${SUMMARY}" "${STUDY}" "${final_count}" <<'PY' || _log "warning: summary generation failed"
import json, sys, datetime
report, summary, study, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(report) as fh:
    data = json.load(fh)
lines = [
    f"Final run summary - {study}",
    f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
    f"completed trials: {n}",
    f"trial: {data.get('trial')}",
    f"score: {data.get('score')}",
]
for key in (
    "val_score", "val_profit_factor", "val_win_rate", "val_trades",
    "val_drawdown", "val_avg_roi", "profit_factor", "win_rate",
    "drawdown", "trades",
):
    m = data.get("metrics", {})
    if key in m:
        lines.append(f"{key}: {m[key]}")
lines.append("params:")
lines.append(json.dumps(data.get("params", {}), indent=2))
open(summary, "w").write("\n".join(lines) + "\n")
print("summary written to", summary)
PY
    fi
else
    _log "error: run did not complete (rc=${rc}, trials=${final_count}/${TRIALS}). Uploading partial results."
    outcome="FAILED"
fi

{
    echo "study: ${STUDY}"
    echo "outcome: ${outcome}"
    echo "completed trials: ${final_count}/${TRIALS}"
    echo "rc: ${rc}"
    echo "finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "bundle: ${BUNDLE}"
} > "${STATUS}"
_log "status file written: ${STATUS}"

# --- Upload (always, retried, verified) -------------------------------------
# rclone accepts a single source per `copy` invocation reliably (batched
# sources/dirs can error), so each file is uploaded with its own command and
# verified individually.

# Build an explicit list of "<local path> <remote name>" pairs.
declare -a PAIRS=()
_add() { # $1 local-path  $2 remote-name
    [ -f "$1" ] && PAIRS+=("$1|$2")
}
_add "${DB}" "$(basename "${DB}")"
[ -f "${DB}-wal" ] && _add "${DB}-wal" "$(basename "${DB}-wal")"
[ -f "${DB}-shm" ] && _add "${DB}-shm" "$(basename "${DB}-shm")"
_add "${STATUS}" "final_status_${BUNDLE}.txt"
_add "${LOG}" "final_run_${BUNDLE}.log"
_add "${REPORT}" "best_strategy_${BUNDLE}.json"
_add "${SUMMARY}" "final_summary_${BUNDLE}.txt"
_add "${ROOT}/cache/selected_features.json" "selected_features.json"

if [ "${#PAIRS[@]}" = "0" ]; then
    _log "ERROR: nothing to upload."
elif [ "${#PAIRS[@]}" -gt 100 ]; then
    _log "ERROR: upload list seems malformed (${#PAIRS[@]} entries)."
fi

uploaded=0
for attempt in 1 2 3 4 5; do
    failed=0
    for pair in "${PAIRS[@]}"; do
        src="${pair%%|*}"
        name="${pair#*|}"
        _log "  upload attempt ${attempt}/5: ${name}"
        if ! rclone copy --progress ${EXTRA_RCLONE_FLAGS:-} "${src}" "${RCLONE_DEST}" >/dev/null 2>&1; then
            _log "  ...rclone returned an error for ${name}"
            failed=1
            continue
        fi
        if ! rclone lsf "${RCLONE_DEST}" 2>/dev/null | grep -qx "${name}"; then
            _log "  ...${name} not verified on remote"
            failed=1
        fi
    done
    if [ "${failed}" = "0" ]; then
        uploaded=1
        break
    fi
    _log "warning: some files failed, retrying..."
    sleep 30
done

if [ "${uploaded}" = "1" ]; then
    _log "upload verified: all files present at ${RCLONE_DEST}"
else
    _log "ERROR: upload failed after 5 attempts."
    _log "  run manually, one file at a time, e.g.:"
    for pair in "${PAIRS[@]}"; do
        _log "    rclone copy ${pair%%|*} ${RCLONE_DEST}"
    done
fi

_log "=== done $(date '+%Y-%m-%d %H:%M:%S') ==="
if [ "${uploaded}" = "1" ]; then
    if [ "${SHUTDOWN_AFTER_UPLOAD}" = "1" ]; then
        _log "all results uploaded to ${RCLONE_DEST}; powering off the VPS in 30s..."
        sleep 30
        if [ "$(id -u)" = "0" ]; then
            shutdown -h now || _log "warning: shutdown command failed; power off the VPS manually."
        else
            sudo shutdown -h now || _log "warning: shutdown command failed; power off the VPS manually."
        fi
    fi
    if [ "${outcome}" = "COMPLETE" ]; then
        exit 0
    fi
fi
exit 1
