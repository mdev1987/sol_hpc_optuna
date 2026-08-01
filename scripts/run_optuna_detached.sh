#!/usr/bin/env bash
#
# run_optuna_detached.sh
#
# Run the Optuna optimization detached from the terminal so a dropped
# SSH/network session cannot kill it (the common "Terminated" cause on
# metered Wi-Fi VPS sessions). Output goes to a log file.
#
# The SQLite study (optuna.db) already persists every completed trial, so
# re-running after an interruption simply resumes where it left off.
#
# Usage:
#   ./scripts/run_optuna_detached.sh [trials] [workers] [bundle]
#
#   trials     number of trials per invocation (default 5000)
#   workers    worker processes (default 16)
#   bundle     feature bundle: structure | flow | early_momentum | reduced_full
#              (default: none, uses selected_features.json)
#
# Examples:
#   ./scripts/run_optuna_detached.sh           # 5000 trials, 16 workers
#   ./scripts/run_optuna_detached.sh 1000 16   # 1000 trials, 16 workers
#   ./scripts/run_optuna_detached.sh 1500 16 flow   # flow bundle ablation
#
# Watch progress:   tail -f logs/optuna_detached.log
# Stop it:          pkill -f hpc_replay.py  (or the tmux session name)

set -euo pipefail

TRIALS="${1:-5000}"
WORKERS="${2:-16}"
BUNDLE="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

SESSION="optuna-detached${BUNDLE:+-${BUNDLE}}"
LOG="${LOG_DIR}/optuna_detached${BUNDLE:+-${BUNDLE}}.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "A detached run already exists (session '${SESSION}')."
    echo "  tail -f ${LOG}"
    echo "  tmux attach -t ${SESSION}"
    echo "To restart, stop it first: tmux kill-session -t ${SESSION}"
    exit 1
fi

# Ensure the study schema exists before starting detached workers, so a
# schema race on first run cannot deadlock the pool.
STUDY_NAME="replay_optuna${BUNDLE:+_${BUNDLE}}"
if ! ls "${ROOT}"/optuna.db >/dev/null 2>&1; then
    echo "optuna.db not found; creating study schema first (sync)..."
    uv run --project "${ROOT}" python -c "
import sys
sys.path.insert(0, '${ROOT}')
from pathlib import Path
from optuna_engine import OptunaConfig, Optimizer
c = OptunaConfig(
    dataset=Path('${ROOT}/cache/features.parquet'),
    output_dir=Path('${ROOT}/reports'),
    study_name='${STUDY_NAME}',
    storage='sqlite:///${ROOT}/optuna.db',
    trials=1, jobs=1, seed=42,
    selected_features=[],
)
Optimizer(c).study()
print('study created')
"
fi

echo "Starting detached Optuna run (${TRIALS} trials, ${WORKERS} workers)"
echo "  study: ${STUDY_NAME}${BUNDLE:+ (bundle: ${BUNDLE})}"
echo "  log: ${LOG}"

BUNDLE_ARG=""
if [ -n "${BUNDLE}" ]; then
    BUNDLE_ARG="--bundle ${BUNDLE}"
fi

tmux new-session -d -s "${SESSION}" \
    "cd ${ROOT} && uv run python hpc_replay.py optimize --trials ${TRIALS} --workers ${WORKERS} --resume ${BUNDLE_ARG} 2>&1 | tee ${LOG}"

echo "Started. Progress: tail -f ${LOG}"
