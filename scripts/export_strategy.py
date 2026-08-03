"""One-time bootstrap: bake the training scaler into a self-contained
strategy file for the live paper-trading bot.

``best_strategy_<bundle>.json`` stores only parameters + train/val metrics.
But the weighted eligibility score standardizes each feature with the
mean/std computed over the TRAIN rows of the training window
(``SnapshotDataset.scaler``), so a live bot needs that exact scaler to
reproduce the validated behavior on fresh data.

The scaler is computed with a lazy polars aggregation over the bundle
feature columns only (mean/std, ``ddof=0``, NaN/inf/None treated as 0), which
mirrors ``SnapshotDataset`` exactly but avoids loading the numpy feature
matrix — so this runs on a laptop, not just the 62GB HPC box.

Writes:

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


def train_scaler(
    features_path: Path,
    features: list[str],
    validation_fraction: float = 0.2,
) -> dict[str, list[float]]:
    """Per-feature (mean, std) over the train rows of the training window.

    Mirrors ``SnapshotDataset``: train rows are ``timestamp <= cutoff`` where
    cutoff is the ``1 - validation_fraction`` quantile of timestamps; NaN/None
    values count as 0.0; std uses ``ddof=0`` (population) like ``np.std``.
    Computed lazily so memory stays bounded on small machines.
    """
    import polars as pl

    needed = ["timestamp", *features]
    scan = pl.scan_parquet(features_path).select(needed)

    cutoff = scan.select(pl.col("timestamp")).collect().to_series().quantile(
        1.0 - validation_fraction, interpolation="linear"
    )
    train = scan.filter(pl.col("timestamp") <= cutoff)

    exprs = []
    for c in features:
        # Mirror SnapshotDataset exactly: cast to float64, then
        # nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0) so NaN,
        # +inf and -inf all count as 0.0 before mean/std (ddof=0).
        raw = pl.col(c).cast(pl.Float64)
        clean = (
            pl.when(raw.is_infinite()).then(0.0).otherwise(raw)
            .fill_nan(0.0).fill_null(0.0)
        )
        exprs.append(clean.mean().alias(f"{c}__mean"))
        exprs.append(clean.std(ddof=0).alias(f"{c}__std"))
    try:
        row = train.select(exprs).collect(engine="streaming").row(0)
    except TypeError:
        row = train.select(exprs).collect(streaming=True).row(0)

    return {
        c: [float(row[2 * i]), float(row[2 * i + 1])]
        for i, c in enumerate(features)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="reports/best_strategy_flow.json",
                    help="validated winner report from report_val.py")
    ap.add_argument("--bundle", default="flow", choices=list(BUNDLES),
                    help="feature bundle the report was produced with")
    ap.add_argument("--train-features", default="cache/features.parquet",
                    help="training-window features file (lazy scan)")
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

    print(f"computing train scaler from {features_path} "
          f"({len(features)} features, {args.validation_fraction:.0%} holdout)...",
          flush=True)
    t0 = time.time()
    scaler = train_scaler(features_path, features, args.validation_fraction)
    print(f"scaler ready in {time.time() - t0:.1f}s", flush=True)

    strategy = json.loads(strategy_path.read_text())
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
