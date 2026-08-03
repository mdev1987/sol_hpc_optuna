"""One-time bootstrap: bake the training scaler into a self-contained
strategy file for the live paper-trading bot.

``best_strategy_<bundle>.json`` stores only parameters + train/val metrics.
But the weighted eligibility score standardizes each feature with the
mean/std computed over the TRAIN rows of the training window
(``SnapshotDataset.scaler``), so a live bot needs that exact scaler to
reproduce the validated behavior on fresh data.

This script rebuilds the training dataset exactly like ``report_val.py``
(projected read, 0.2 holdout, full data) and writes:

    reports/paper_strategy_<bundle>.json =
        {trial, score, parameters, features, scaler, metrics}

Usage:
    uv run python scripts/export_strategy.py \
        --strategy reports/best_strategy_flow.json \
        --bundle flow \
        --train-features cache/features.parquet \
        --out reports/paper_strategy_flow.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a plain file: ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feature_bundles import BUNDLES
from optuna_engine import SnapshotDataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="reports/best_strategy_flow.json",
                    help="validated winner report from report_val.py")
    ap.add_argument("--bundle", default="flow", choices=list(BUNDLES),
                    help="feature bundle the report was produced with")
    ap.add_argument("--train-features", default="cache/features.parquet",
                    help="training-window features file (projected read)")
    ap.add_argument("--out", default="reports/paper_strategy_flow.json",
                    help="self-contained strategy file the live bot reads")
    ap.add_argument("--validation-fraction", type=float, default=0.2,
                    help="holdout fraction used at dataset build (must match report_val)")
    args = ap.parse_args()

    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        sys.exit(f"strategy report not found: {strategy_path}")

    features = list(BUNDLES[args.bundle])
    features_path = Path(args.train_features)
    if not features_path.exists():
        sys.exit(f"train features not found: {features_path}")

    print(f"building {args.bundle} dataset from {features_path} "
          f"(projected read, {args.validation_fraction:.0%} holdout)...", flush=True)
    t0 = time.time()
    dataset = SnapshotDataset(
        features_path,
        validation_fraction=args.validation_fraction,
        sample_fraction=1.0,
        columns=features,
    )
    print(f"dataset ready in {time.time() - t0:.1f}s", flush=True)

    strategy = json.loads(strategy_path.read_text())
    scaler = {f: dataset.scaler[f] for f in features if f in dataset.scaler}
    missing = [f for f in features if f not in dataset.scaler]
    if missing:
        print(f"WARNING: scaler missing for features: {missing}", flush=True)

    out = {
        "trial": strategy.get("trial"),
        "score": strategy.get("score"),
        "metrics": strategy.get("metrics"),
        "features": features,
        "scaler": scaler,
        "parameters": strategy["parameters"],
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=4))
    print(f"saved self-contained strategy -> {out_path} "
          f"({len(features)} features, scaler baked in)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
