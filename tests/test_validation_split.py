from pathlib import Path

import numpy as np
import polars as pl
import pytest

from optuna_engine import (
    MIN_TRAIN_TRADES,
    PF_CAP,
    ROI_CAP,
    SnapshotDataset,
    evaluate_params,
    score_simulation,
    _min_trades_for,
    _strategy_from_params,
)
from simulator import ExitReason, SimulationResult, Simulator, Trade, WeightedStrategy


def _result_with_trades(rois, profit_factor, total_return=0.0, total_pnl=0.0, max_drawdown=0.0):
    trades = [
        Trade(
            mint=f"M{i}",
            entry_time=0,
            exit_time=1,
            entry_price=1.0,
            exit_price=1.0 + roi,
            quantity=1.0,
            invested=1.0,
            received=1.0 + roi,
            pnl=roi,
            roi=roi,
            reason=ExitReason.TAKE_PROFIT,
        )
        for i, roi in enumerate(rois)
    ]
    return SimulationResult(
        final_balance=10.0 + total_pnl,
        total_return=total_return,
        trades=len(trades),
        wins=sum(1 for r in rois if r > 0),
        losses=sum(1 for r in rois if r < 0),
        win_rate=sum(1 for r in rois if r > 0) / len(rois) if rois else 0.0,
        gross_profit=sum(r for r in rois if r > 0),
        gross_loss=sum(-r for r in rois if r < 0),
        profit_factor=profit_factor,
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        equity_curve=[],
        closed_trades=trades,
    )


def _write_features(tmp_path: Path, n: int = 100) -> Path:
    path = tmp_path / "features.parquet"
    ts = np.linspace(1_000, 2_000, n)
    frame = pl.DataFrame(
        {
            "mint": ["A"] * n,
            "timestamp": ts.astype(int),
            "slot": list(range(n)),
            "price": np.linspace(1.0, 2.0, n),
            "liquidity": np.full(n, 100.0),
            "volume": np.full(n, 10.0),
        }
    )
    frame.write_parquet(path)
    return path


def test_split_partitions_rows_by_time(tmp_path):
    path = _write_features(tmp_path)
    dataset = SnapshotDataset(path, validation_fraction=0.2)

    train = list(dataset.snapshots())
    val = list(dataset.validation_snapshots())

    assert len(train) == 80
    assert len(val) == 20
    assert max(s.timestamp for s in train) < min(s.timestamp for s in val)


def test_no_split_returns_all_rows(tmp_path):
    path = _write_features(tmp_path)
    dataset = SnapshotDataset(path, validation_fraction=0.0)

    train = list(dataset.snapshots())
    val = list(dataset.validation_snapshots())

    assert len(train) == 100
    assert len(val) == 0


def test_scaler_fit_on_train_only(tmp_path):
    path = _write_features(tmp_path)
    dataset = SnapshotDataset(path, validation_fraction=0.2)

    mean, std = dataset.scaler["price"]
    all_prices = np.linspace(1.0, 2.0, 100)
    train_prices = all_prices[:80]
    assert mean == pytest.approx(train_prices.mean())
    assert std == pytest.approx(train_prices.std())


def test_validation_fraction_clamped_to_half(tmp_path):
    path = _write_features(tmp_path)
    dataset = SnapshotDataset(path, validation_fraction=0.9)
    assert dataset.validation_fraction == 0.5


def test_sample_fraction_keeps_whole_mints(tmp_path):
    n = 200
    path = tmp_path / "features.parquet"
    mints = [f"M{i % 10}" for i in range(n)]
    frame = pl.DataFrame(
        {
            "mint": mints,
            "timestamp": list(range(1_000, 1_000 + n)),
            "slot": list(range(n)),
            "price": np.linspace(1.0, 2.0, n),
            "liquidity": np.full(n, 100.0),
        }
    )
    frame.write_parquet(path)

    dataset = SnapshotDataset(path, validation_fraction=0.0, sample_fraction=0.5)
    snapshots = list(dataset.snapshots())
    kept = {s.mint for s in snapshots}
    assert len(kept) == 5  # ~50% of the 10 unique mints
    # every row belongs to a kept mint (no partial-mint slicing)
    assert all(s.mint in kept for s in snapshots)
    assert len(snapshots) == 5 * 20  # all 20 rows of each kept mint survive


def test_sample_fraction_full_is_unchanged(tmp_path):
    path = _write_features(tmp_path)
    dataset = SnapshotDataset(path, validation_fraction=0.0, sample_fraction=1.0)
    assert len(list(dataset.snapshots())) == 100


def test_evaluate_params_runs_both_splits(tmp_path):
    path = _write_features(tmp_path)
    dataset = SnapshotDataset(path, validation_fraction=0.2)

    params = {
        "w_price": 1.0,
        "minimum_score": -5.0,
        "position_size": 0.2,
        "stop_loss": 0.15,
        "take_profit": 2.0,
        "trailing_trigger": 0.3,
        "trailing_stop": 0.2,
        "ttl_seconds": 300,
        "max_positions": 3,
    }
    score_train, _ = evaluate_params(params, dataset, ["price"], on_validation=False)
    score_val, _ = evaluate_params(params, dataset, ["price"], on_validation=True)

    assert score_train == pytest.approx(score_val)  # same 20% price path per mint


def test_score_simulation_metrics_shape(tmp_path):
    path = _write_features(tmp_path, n=50)
    dataset = SnapshotDataset(path, validation_fraction=0.0)
    strategy_config, sim_config = _strategy_from_params(
        {"w_price": 1.0, "minimum_score": -5.0}, ["price"], dataset.scaler
    )
    sim = Simulator(sim_config, WeightedStrategy(strategy_config))
    result = sim.run(dataset.snapshots())
    score, metrics = score_simulation(result)
    assert {"profit_factor", "win_rate", "drawdown", "trades"} <= set(metrics)


def test_score_simulation_caps_absurd_profit_factor():
    result = _result_with_trades([0.2] * 30, profit_factor=1e12)
    score, metrics = score_simulation(result)

    assert metrics["profit_factor"] == PF_CAP
    assert score == pytest.approx(
        2.5 * PF_CAP + 1.5 * 1.0 + 2.0 * 0.2 - 3.0 * 0.0
    )
    assert abs(score) < 30  # bounded, not exploded by PF/total_pnl


def test_score_simulation_ignores_compounding_pnl():
    result = _result_with_trades(
        [0.1, 0.1, 0.1],
        profit_factor=2.0,
        total_return=1e9,
        total_pnl=1e12,
    )
    score, metrics = score_simulation(result)

    assert metrics["avg_roi"] == pytest.approx(0.1)
    assert score == pytest.approx(
        2.5 * 2.0 + 1.5 * 1.0 + 2.0 * 0.1 - 3.0 * 0.0
    )


def test_score_simulation_zero_loss_caps_profit_factor():
    result = _result_with_trades([0.3, 0.4], profit_factor=999.0)
    score, metrics = score_simulation(result)

    assert metrics["profit_factor"] == PF_CAP
    assert metrics["avg_roi"] == pytest.approx(0.35)
    assert score == pytest.approx(2.5 * PF_CAP + 1.5 * 1.0 + 2.0 * 0.35)


def test_score_simulation_caps_avg_roi():
    result = _result_with_trades([50.0, 60.0], profit_factor=5.0)
    score, metrics = score_simulation(result)

    assert metrics["avg_roi"] == ROI_CAP
    assert score == pytest.approx(2.5 * PF_CAP + 1.5 * 1.0 + 2.0 * ROI_CAP)
    assert abs(score) < 30


def test_validation_floor_is_proportional(tmp_path):
    path = _write_features(tmp_path, n=100)
    dataset = SnapshotDataset(path, validation_fraction=0.2)
    n_train = int(dataset._train_mask.sum())
    n_val = int(dataset._val_mask.sum())

    assert _min_trades_for(dataset, on_validation=False) == MIN_TRAIN_TRADES
    expected_val = max(1, round(MIN_TRAIN_TRADES * n_val / n_train))
    assert _min_trades_for(dataset, on_validation=True) == expected_val
    assert expected_val < MIN_TRAIN_TRADES  # holdout is smaller than training


def test_evaluate_params_floor_blocks_low_trade_config(tmp_path):
    path = _write_features(tmp_path)  # single mint, ~3 trades: below the floor
    dataset = SnapshotDataset(path, validation_fraction=0.0)
    params = {"w_price": 1.0, "minimum_score": -5.0}

    score, metrics = evaluate_params(params, dataset, ["price"], on_validation=False)

    assert score == -1e9
    assert metrics["trades"] < MIN_TRAIN_TRADES
