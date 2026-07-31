import polars as pl
import pytest

from feature_engine import FeatureSnapshot
from optuna_engine import SnapshotDataset
from simulator import ExitReason, Simulator, SimulatorConfig, Strategy, StrategyConfig, WeightedStrategy


class AlwaysEnter(Strategy):
    def should_enter(self, snapshot: FeatureSnapshot) -> bool:
        return True


def _snap(mint: str, ts: int, price: float) -> FeatureSnapshot:
    return FeatureSnapshot(mint=mint, timestamp=ts, slot=ts, features={"price": price})


def test_open_positions_close_at_last_market_price():
    config = SimulatorConfig(take_profit=5.0, ttl_seconds=600)
    sim = Simulator(config, AlwaysEnter())
    result = sim.run([_snap("A", 0, 1.0), _snap("A", 10, 1.5)])

    assert result.trades == 1
    trade = result.closed_trades[-1]
    assert trade.reason == ExitReason.MANUAL
    assert trade.exit_price == 1.5
    assert trade.exit_time == 10
    assert trade.exit_price != trade.entry_price


def test_score_uses_standardized_values():
    cfg = StrategyConfig(weights={"a": 1.0}, scaler={"a": (10.0, 1.0)})
    strategy = WeightedStrategy(cfg)
    snap = FeatureSnapshot(mint="A", timestamp=0, slot=0, features={"a": 5.0})

    assert strategy.score(snap) == pytest.approx(-5.0)


def test_score_zero_std_feature_contributes_zero():
    cfg = StrategyConfig(weights={"a": 1.0}, scaler={"a": (10.0, 0.0)})
    strategy = WeightedStrategy(cfg)
    snap = FeatureSnapshot(mint="A", timestamp=0, slot=0, features={"a": 5.0})

    assert strategy.score(snap) == pytest.approx(0.0)


def test_score_without_scaler_uses_raw_values():
    cfg = StrategyConfig(weights={"a": 1.0})
    strategy = WeightedStrategy(cfg)
    snap = FeatureSnapshot(mint="A", timestamp=0, slot=0, features={"a": 5.0})

    assert strategy.score(snap) == pytest.approx(5.0)


def test_snapshot_dataset_scaler(tmp_path):
    df = pl.DataFrame(
        {
            "mint": ["A", "A", "B"],
            "timestamp": [0, 10, 20],
            "slot": [0, 1, 2],
            "price": [1.0, 2.0, 3.0],
            "age_seconds": [10, 20, 30],
        }
    )
    path = tmp_path / "features.parquet"
    df.write_parquet(path)

    dataset = SnapshotDataset(path)
    assert "mint" not in dataset.scaler
    assert "price" in dataset.scaler and "age_seconds" in dataset.scaler
    assert dataset.scaler["price"] == pytest.approx((2.0, (2.0 / 3.0) ** 0.5))
    assert dataset.scaler["age_seconds"] == pytest.approx((20.0, (200.0 / 3.0) ** 0.5))
