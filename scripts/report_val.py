"""Replay the top-N trials from a completed Optuna study on the validation
holdout and pick the winner by validation score.

The completed 5000-trial run stores only training-side metrics on each trial
(user_attrs), because the run was interrupted before `Optimizer.run()` performed
its final validation pass. This script re-runs the top candidates through
`evaluate_params` on BOTH train and validation, writes a ranked table, and
regenerates `reports/best_strategy_<bundle>.json` from the trial with the best
`val_score`. The winner rule:

    pick best val_score; tie-break lower val_drawdown; sanity-check val_trades
    is above the proportional holdout floor (~150 * val/train).

Usage:
    uv run python scripts/report_val.py \
        [--db optuna.db] [--bundle flow] [--top N] [--out reports] \
        [--feature-cache cache/features.parquet]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import optuna

from feature_bundles import BUNDLES
from optuna_engine import (
    OptunaConfig,
    SnapshotDataset,
    evaluate_params,
    _min_trades_for,
)


def _load_study(db: Path, study_name: str) -> optuna.Study:
    storage = f"sqlite:///{db}"
    # WAL so read-only replay does not stall on a partially-written journal.
    try:
        import sqlite3

        con = sqlite3.connect(str(db), timeout=60)
        con.execute("PRAGMA journal_mode=WAL")
        con.close()
    except Exception:
        pass
    return optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=10),
    )


def build_dataset(cache: Path, features: list[str]) -> SnapshotDataset:
    ds_path = cache / "features.parquet"
    if not ds_path.exists():
        sys.exit(f"features not found: {ds_path}")
    config = OptunaConfig(
        dataset=ds_path,
        output_dir=Path("."),
        study_name="replay",
        selected_features=features,
        validation_fraction=0.2,
        sample_fraction=1.0,
    )
    return SnapshotDataset(
        config.dataset,
        validation_fraction=config.validation_fraction,
        sample_fraction=config.sample_fraction,
        columns=config.selected_features,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="optuna.db")
    ap.add_argument("--bundle", default="flow")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--save-dir", default="reports")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--no-save", action="store_true",
                    help="only print the ranked table, do not write a winner report")
    args = ap.parse_args()

    features = list(BUNDLES[args.bundle])
    study_name = f"replay_optuna_{args.bundle}"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    study = _load_study(Path(args.db), study_name)

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        print(f"no complete trials in {study_name}", flush=True)
        return 1
    ranked = sorted(complete, key=lambda t: (t.value is None, t.value or 0.0),
                    reverse=True)[: args.top]
    print(f"loaded {len(complete)} complete trials; checking top {len(ranked)}\n",
          flush=True)

    print(f"building {args.bundle} dataset from {args.cache} (full data, 0.2 holdout)...",
          flush=True)
    t0 = time.time()
    dataset = build_dataset(Path(args.cache), features)
    print(f"dataset ready in {time.time() - t0:.1f}s\n", flush=True)

    header = (f"{'trial':>6} {'train':>9} {'val_score':>9} {'val_pf':>6} "
              f"{'val_win':>7} {'val_trades':>9} {'val_dd':>8}  {'val_avg_roi':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for t in ranked:
        params = dict(t.params)
        val_score, val_metrics = evaluate_params(params, dataset, features, on_validation=True)
        train_score, train_metrics = evaluate_params(params, dataset, features, on_validation=False)
        rows.append((t, val_score, val_metrics, train_score, train_metrics))

    rows.sort(key=lambda r: (r[1], -r[2].get("drawdown", 0.0)), reverse=True)

    winner = None
    for t, val_score, val_m, train_score, train_m in rows:
        floor = _min_trades_for(dataset, on_validation=True)
        ok = "" if val_m["trades"] >= floor else f"  <-- BELOW VAL FLOOR {floor}"
        print(f"{t.number:6d} {train_score:9.3f} {val_score:9.3f} "
              f"{val_m['profit_factor']:6.2f} {val_m['win_rate']:7.1%} "
              f"{val_m['trades']:9d} {val_m['drawdown']:8.3f} "
              f"{val_m.get('avg_roi', 0.0):9.3f}{ok}", flush=True)
        if winner is None and val_m["trades"] >= floor:
            winner = (t, val_score, val_m, train_score, train_m)

    print()
    if winner is None:
        print("WARNING: no candidate cleared the validation trade floor.", flush=True)
        return 2

    t, val_score, val_metrics, train_score, train_metrics = winner
    print(f"WINNER = trial {t.number} (val_score {val_score:.4f}, "
          f"val_trades {val_metrics['trades']})", flush=True)

    if args.no_save:
        return 0

    metrics = {
        "profit_factor": train_metrics["profit_factor"],
        "win_rate": train_metrics["win_rate"],
        "drawdown": train_metrics["drawdown"],
        "trades": train_metrics["trades"],
        "val_score": val_score,
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    output = {
        "score": train_score,
        "trial": t.number,
        "parameters": dict(t.params),
        "metrics": metrics,
    }
    out_path = save_dir / f"best_strategy_{args.bundle}.json"
    out_path.write_text(json.dumps(output, indent=4))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())