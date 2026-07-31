#!/usr/bin/env bash
#
# download_aria2.sh
#
# Download PumpAPI replay archives with aria2c (much faster than the
# built-in asyncio downloader on high-bandwidth links).
#
# Files are stored per-day under downloads/YYYY/MM/DD/ so no filename
# collisions occur, matching what the pipeline's parser expects.
#
# Only missing hours are requested, so re-running skips files that are
# already downloaded (no aria2 resume needed).
#
# Usage:
#   ./scripts/download_aria2.sh [days] [concurrent]
#
#   days         how many UTC days ending yesterday (default 3)
#   concurrent   aria2c -j parallel files (default 16)
#
# Notes:
#   - -x 1 -s 1: the server (Cloudflare) rejects range requests, so we
#     must not split a single file across connections.

set -euo pipefail

DAYS="${1:-3}"
JOBS="${2:-16}"
BASE_URL="https://replay.pumpapi.io"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${ROOT}/downloads"

command -v aria2c >/dev/null 2>&1 || { echo "aria2c not found. Install: sudo apt install aria2"; exit 1; }

mkdir -p "${DOWNLOAD_DIR}"

# Remove stale control files from interrupted/pre-flag runs; the server
# cannot resume partial files, so resuming is never possible anyway.
find "${DOWNLOAD_DIR}" -name "*.aria2" -delete

# Drop empty/partial leftovers so they get re-downloaded.
find "${DOWNLOAD_DIR}" -name "*.jsonl.zst" -size 0 -delete

total_files=0
for d in $(seq 0 $((DAYS - 1))); do
    # Previous N UTC days ending yesterday.
    day=$(date -u -d "-$((d + 1)) day" +%F)
    yyyy=${day%%-*}
    mm=${day#*-}; mm=${mm%%-*}
    dd=${day##*-}

    dir="${DOWNLOAD_DIR}/${yyyy}/${mm}/${dd}"
    mkdir -p "${dir}"

    links=""
    existing=0
    for h in $(seq -w 0 23); do
        if [ -f "${dir}/${h}.jsonl.zst" ]; then
            existing=$((existing + 1))
        else
            links+="${BASE_URL}/${yyyy}/${mm}/${dd}/${h}.jsonl.zst"$'\n'
        fi
    done

    if [ -n "${links}" ]; then
        aria2c \
            -j "${JOBS}" \
            -x 1 -s 1 \
            --auto-file-renaming=false \
            --allow-overwrite=false \
            --console-log-level=warn \
            --summary-interval=0 \
            -d "${dir}" \
            -i <(printf "%s" "${links}")
    fi

    count=$(find "${dir}" -maxdepth 1 -name "*.jsonl.zst" | wc -l)
    total_files=$((total_files + count))
    echo "  day ${yyyy}-${mm}-${dd}: ${count}/24 files (${existing} already present)"
done

# Clean up renamed leftovers from older flat downloads.
find "${DOWNLOAD_DIR}" -name "*.jsonl.1.zst" -delete

echo "Downloaded ${total_files}/$((DAYS * 24)) hourly files into ${DOWNLOAD_DIR}"
