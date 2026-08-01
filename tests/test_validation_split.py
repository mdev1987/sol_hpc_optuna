from pathlib import Path

import numpy as np
import polars as pl
import pytest

from optuna_engine import SnapshotDataset, evaluate_params, score_simulation, _strategy_from_params
from simulator import Simulator, WeightedStrategy


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
