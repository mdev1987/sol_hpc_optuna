#!/usr/bin/env bash
#
# run_all_bundles.sh
#
# Run the bundle ablations SEQUENTIALLY (one at a time) so the shared
# optuna.db is never accessed by two studies at once. Each bundle runs via
# run_optuna_detached.sh in its own tmux session; this script monitors it
# with a live progress bar and aborts the whole sequence if a bundle fails.
#
# The entire sequence runs inside its own tmux session ("optuna-bundles"),
# so a dropped SSH/network session cannot kill it.
#
# Usage:
#   ./scripts/run_all_bundles.sh [trials] [workers] [bundle...]
#
#   trials     trials per bundle (default 300; ablation budget, not the
#              final 5000-trial run)
#   workers    worker processes per bundle (default 16)
#   bundle...  subset of bundles to run (default: structure flow
#              early_momentum reduced_full)
#
#   Env:
#   SAMPLE_FRACTION  fraction of mints to keep per trial (default 0.3).
#                    Sub-samples whole mints, cutting runtime and RAM ~3x.
#                    Set SAMPLE_FRACTION=1.0 for the full-data winner run.
#
# Examples:
#   ./scripts/run_all_bundles.sh                  # 300 trials, all 4 bundles
#   ./scripts/run_all_bundles.sh 1000 16          # 1000 trials, all 4
#   ./scripts/run_all_bundles.sh 1500 16 flow reduced_full
#   SAMPLE_FRACTION=1.0 ./scripts/run_all_bundles.sh 5000 16 flow
#
# Watch progress:   tail -f logs/optuna_bundles.log
# Live bar:         tmux attach -t optuna-bundles
# Stop it:          tmux kill-session -t optuna-bundles

set -euo pipefail

TRIALS="${1:-300}"
WORKERS="${2:-16}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.3}"
shift 2 || true
if [ "$#" -gt 0 ]; then
    BUNDLES=("$@")
else
    BUNDLES=(structure flow early_momentum reduced_full)
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

OUTER_SESSION="optuna-bundles"
OUTER_LOG="${LOG_DIR}/optuna_bundles.log"
DB="${ROOT}/optuna.db"

# --- Self-restart inside the outer tmux session ----------------------------
if [ -z "${RUN_ALL_BUNDLES_INNER:-}" ]; then
    if tmux has-session -t "${OUTER_SESSION}" 2>/dev/null; then
        echo "Outer session '${OUTER_SESSION}' already running."
        echo "  tail -f ${OUTER_LOG}"
        echo "  tmux attach -t ${OUTER_SESSION}"
        exit 1
    fi
    SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
    echo "Starting sequential bundle runs inside tmux session '${OUTER_SESSION}'..."
    tmux new-session -d -s "${OUTER_SESSION}" \
        "cd ${ROOT} && RUN_ALL_BUNDLES_INNER=1 SAMPLE_FRACTION=${SAMPLE_FRACTION} bash ${SCRIPT} ${TRIALS} ${WORKERS} ${BUNDLES[*]} 2>&1 | tee ${OUTER_LOG}"
    echo "Started. Watch: tail -f ${OUTER_LOG}"
    exit 0
fi

# --- Helpers ----------------------------------------------------------------

# Count COMPLETE trials for a study via stdlib sqlite3 (same query as optuna_engine).
_count_complete() {
    local study="$1"
    python3 - "${DB}" "${study}" <<'PY'
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

_elapsed() {
    echo "$(( $(date +%s) - $1 ))"
}

# Monitor one bundle until its session ends; returns 0 if complete, 1 if failed.
monitor_bundle() {
    local bundle="$1" idx="$2" total="$3"
    local session="optuna-detached-${bundle}"
    local study="replay_optuna_${bundle}"
    local start now elapsed
    start="$(date +%s)"
    local count prev=-1 width=20 filled bar k pct
    local line

    echo ""
    echo "[${idx}/${total}] ${bundle}: monitoring ${session} (target ${TRIALS} trials)..."
    while true; do
        count="$(_count_complete "${study}")"
        now="$(date +%s)"
        elapsed="$(( now - start ))"
        pct=$(( count * 100 / TRIALS ))
        filled=$(( count * width / TRIALS ))
        [ "$filled" -gt "$width" ] && filled="$width"
        bar=""
        for ((k = 0; k < width; k++)); do
            if ((k < filled)); then bar+="#"; else bar+="-"; fi
        done

        if ((count > 0)); then
            local eta=0
            eta=$(( elapsed * (TRIALS - count) / count ))
            line=$(printf "\r[%s/%s] %-16s [%s] %6d/%-6d (%3d%%)  elapsed %02dm%02ds  eta ~%02dm%02ds" \
                "${idx}" "${total}" "${bundle}" "${bar}" "${count}" "${TRIALS}" "${pct}" \
                "$((elapsed / 60))" "$((elapsed % 60))" "$((eta / 60))" "$((eta % 60))")
        else
            line=$(printf "\r[%s/%s] %-16s [%s] %6d/%-6d (%3d%%)  elapsed %02dm%02ds" \
                "${idx}" "${total}" "${bundle}" "${bar}" "${count}" "${TRIALS}" "${pct}" \
                "$((elapsed / 60))" "$((elapsed % 60))")
        fi
        printf "%s" "${line}"

        # Session gone -> bundle's command finished (complete) or died (failed).
        if ! tmux has-session -t "${session}" 2>/dev/null; then
            printf "\n"
            if ((count >= TRIALS)); then
                echo "[${idx}/${total}] ${bundle}: DONE (${count}/${TRIALS} trials)"
                return 0
            fi
            echo "[${idx}/${total}] ${bundle}: FAILED - session ended with ${count}/${TRIALS} trials"
            return 1
        fi
        sleep 10
    done
}

# --- Main loop --------------------------------------------------------------

echo "=== Sequential bundle ablations ==="
echo "  trials/bundle: ${TRIALS}   workers: ${WORKERS}   sample_fraction: ${SAMPLE_FRACTION}"
echo "  bundles: ${BUNDLES[*]}"
echo "  db: ${DB}"
echo ""

TOTAL="${#BUNDLES[@]}"
IDX=0
for bundle in "${BUNDLES[@]}"; do
    IDX=$((IDX + 1))
    study="replay_optuna_${bundle}"
    session="optuna-detached-${bundle}"
    done_count="$(_count_complete "${study}")"

    if ((done_count >= TRIALS)); then
        echo "[${IDX}/${TOTAL}] ${bundle}: already complete (${done_count}/${TRIALS}), skipping."
        continue
    fi

    if [ "${done_count}" -gt 0 ]; then
        echo "[${IDX}/${TOTAL}] ${bundle}: resuming from ${done_count}/${TRIALS} trials."
    fi

    # Launch only if the bundle session is not already running.
    if ! tmux has-session -t "${session}" 2>/dev/null; then
        bash "${SCRIPT_DIR}/run_optuna_detached.sh" "${TRIALS}" "${WORKERS}" "${bundle}" "${SAMPLE_FRACTION}"
    else
        echo "[${IDX}/${TOTAL}] ${bundle}: session '${session}' already running, monitoring."
    fi

    if ! monitor_bundle "${bundle}" "${IDX}" "${TOTAL}"; then
        echo ""
        echo "!!! ABORTING SEQUENCE: ${bundle} failed."
        echo "    Check logs/optuna_detached-${bundle}.log. Completed trials persist in optuna.db;"
        echo "    re-run this script to resume the remaining bundles."
        exit 1
    fi
done

# --- Summary ----------------------------------------------------------------

echo ""
echo "=== Summary ==="
python3 - "${ROOT}" "${BUNDLES[*]}" <<'PY'
import json, sys
root, bundles = sys.argv[1], sys.argv[2].split()
print(f"{'bundle':<16} {'val_score':>10} {'val_pf':>8} {'val_win':>9} {'val_trades':>11}")
for b in bundles:
    path = f"{root}/reports/best_strategy_{b}.json"
    try:
        with open(path) as fh:
            m = json.load(fh)["metrics"]
        vs = m.get("val_score")
        pf = m.get("val_profit_factor")
        wr = m.get("val_win_rate")
        tr = m.get("val_trades")
        print(f"{b:<16} {vs:>10.4f} {pf:>8.2f} {wr:>9.2%} {tr:>11}")
    except Exception:
        print(f"{b:<16} {'no result':>10}")
PY

echo ""
echo "All bundles finished. Compare val_score/val_profit_factor and run the full"
echo "5000-trial sequence on the winner."
