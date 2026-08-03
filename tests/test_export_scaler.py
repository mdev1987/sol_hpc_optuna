from pathlib import Path

import numpy as np
import polars as pl
import pytest

from optuna_engine import SnapshotDataset
from scripts.export_strategy import train_scaler


def _write_messy_features(tmp_path: Path, n: int = 100) -> Path:
    """Synthetic frame with NaN, +inf and -inf scattered in the features."""
    path = tmp_path / "features.parquet"
    ts = np.linspace(1_000, 2_000, n).astype(int)
    price = np.linspace(1.0, 2.0, n)
    liquidity = np.linspace(100.0, 300.0, n)
    price[10] = np.nan
    price[20] = np.inf
    price[30] = -np.inf
    liquidity[15] = np.nan
    liquidity[25] = np.inf
    frame = pl.DataFrame(
        {
            "mint": ["A"] * n,
            "timestamp": ts,
            "slot": list(range(n)),
            "price": price,
            "liquidity": liquidity,
            "volume": np.linspace(10.0, 20.0, n),
        }
    )
    frame.write_parquet(path)
    return path


@pytest.mark.parametrize("features", [["price", "liquidity"], ["volume"]])
def test_train_scaler_matches_snapshot_dataset(tmp_path, features):
    path = _write_messy_features(tmp_path)

    dataset = SnapshotDataset(path, validation_fraction=0.2)
    expected = {f: dataset.scaler[f] for f in features}
    actual = train_scaler(path, features, validation_fraction=0.2)

    assert set(actual) == set(expected)
    for f in features:
        # NaN/inf became 0.0, so std > 0 and finite in both implementations.
        assert actual[f][1] > 0.0
        assert np.isfinite(actual[f][0]) and np.isfinite(actual[f][1])
        assert actual[f][0] == pytest.approx(expected[f][0], rel=1e-9)
        assert actual[f][1] == pytest.approx(expected[f][1], rel=1e-9)


def test_train_scaler_respects_holdout_cutoff(tmp_path):
    path = _write_messy_features(tmp_path, n=100)

    dataset = SnapshotDataset(path, validation_fraction=0.2)
    expected = dataset.scaler["price"]
    actual = train_scaler(path, ["price"], validation_fraction=0.2)

    # The scaler must only depend on rows SnapshotDataset put in TRAIN, with
    # NaN/inf -> 0.0, and must match it to numerical precision.
    raw = np.linspace(1.0, 2.0, 100)
    raw[10] = np.nan
    raw[20] = np.inf
    raw[30] = -np.inf
    work = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)[dataset._train_mask]
    assert actual["price"][0] == pytest.approx(work.mean(), rel=1e-9)
    assert actual["price"][1] == pytest.approx(work.std(), rel=1e-9)
    assert expected == pytest.approx(list(actual["price"]), rel=1e-9)
