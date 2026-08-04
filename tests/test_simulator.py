import numpy as np
import polars as pl
import pytest

from feature_engine import FeatureSnapshot
from optuna_engine import SnapshotDataset, _compute_eligible
from simulator import ExitReason, Simulator, SimulatorConfig, Strategy, StrategyConfig, WeightedStrategy

_THRESHOLD_FEATS = [
    "price_change_5",
    "price_change_20",
    "price_change_50",
    "liquidity",
    "market_cap",
    "volume",
    "buy_ratio",
    "trades",
    "unique_wallets",
    "wallet_velocity",
    "price_velocity",
    "volume_velocity",
    "liquidity_velocity",
]


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


def test_cooldown_blocks_reentry_then_allows():
    config = SimulatorConfig(cooldown_seconds=100, take_profit=0.2, ttl_seconds=10_000)
    sim = Simulator(config, AlwaysEnter())
    result = sim.run([
        _snap("A", 0, 1.0),
        _snap("A", 10, 1.3),   # +30% -> TAKE_PROFIT close, blocked until 110
        _snap("A", 20, 1.0),   # within cooldown -> rejected
        _snap("A", 120, 1.0),  # cooldown expired -> re-enters
    ])

    # The exit event itself also rejects same-tick re-entry.
    assert sim.cooldown_rejects == 2
    assert result.trades == 2
    assert result.closed_trades[0].reason == ExitReason.TAKE_PROFIT


def test_cooldown_once_never_reenters():
    config = SimulatorConfig(cooldown_seconds=-1, take_profit=0.2, ttl_seconds=10_000)
    sim = Simulator(config, AlwaysEnter())
    result = sim.run([
        _snap("A", 0, 1.0),
        _snap("A", 10, 1.3),   # TAKE_PROFIT close, blocked forever
        _snap("A", 20, 1.0),
        _snap("A", 30, 1.0),
        _snap("A", 40, 1.0),
    ])

    assert sim.cooldown_rejects == 4
    assert result.trades == 1


def test_zero_cooldown_allows_immediate_reentry():
    config = SimulatorConfig(cooldown_seconds=0, stop_loss=0.5, take_profit=0.2, ttl_seconds=10_000)
    sim = Simulator(config, AlwaysEnter())
    result = sim.run([
        _snap("A", 0, 1.0),
        _snap("A", 10, 1.3),   # TAKE_PROFIT close
        _snap("A", 20, 1.0),   # no cooldown -> re-enters, still held at finish
    ])

    assert sim.cooldown_rejects == 0
    assert result.trades == 2


def test_strategy_from_params_wires_cooldown():
    from optuna_engine import _strategy_from_params

    _, sim_config = _strategy_from_params(
        {"cooldown_seconds": 300, "position_size": 0.2}, [], {}
    )
    assert sim_config.cooldown_seconds == 300.0


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


def _synthetic_features(n_rows, n_mints, seed):
    rng = np.random.default_rng(seed)
    data = {
        "mint": [f"M{i % n_mints}" for i in range(n_rows)],
        "timestamp": np.arange(n_rows) * 7,
        "slot": np.arange(n_rows),
        "price": np.round(np.exp(rng.normal(0.0, 0.5, n_rows)), 6),
    }
    for feat in _THRESHOLD_FEATS:
        data[feat] = rng.normal(0.0, 1.0, n_rows)
    return pl.DataFrame(data)


def _trades_as_tuples(trades):
    return [
        (
            t.mint,
            t.entry_time,
            t.exit_time,
            t.entry_price,
            t.exit_price,
            t.quantity,
            t.pnl,
            t.roi,
            t.reason,
        )
        for t in trades
    ]


def _assert_identical(dataset, sim_config, strategy_config):
    eligible = _compute_eligible(dataset, strategy_config)
    indices = np.flatnonzero(dataset._train_mask)

    ref = Simulator(sim_config, WeightedStrategy(strategy_config)).run(dataset.snapshots())
    fast = Simulator(sim_config, WeightedStrategy(strategy_config)).run_indexed(
        indices,
        dataset._mints,
        dataset._timestamps,
        dataset._prices,
        eligible,
        dataset.snapshot_at,
    )

    assert _trades_as_tuples(ref.closed_trades) == _trades_as_tuples(fast.closed_trades)
    assert ref.equity_curve == pytest.approx(fast.equity_curve)
    assert ref.final_balance == pytest.approx(fast.final_balance)
    assert ref.trades == fast.trades
    assert ref.total_return == pytest.approx(fast.total_return)


def test_eligible_mask_matches_should_enter(tmp_path):
    df = _synthetic_features(3000, 8, 123)
    path = tmp_path / "features.parquet"
    df.write_parquet(path)
    dataset = SnapshotDataset(path, columns=["price_change_5", "buy_ratio"])
    rng = np.random.default_rng(7)

    for _ in range(10):
        strategy_config = StrategyConfig(
            min_price_change_5=float(rng.uniform(-3.0, 3.0)),
            min_liquidity=float(rng.uniform(0.0, 5.0)),
            min_buy_ratio=float(rng.uniform(-1.0, 0.9)),
            min_trades=int(rng.integers(0, 50)),
            minimum_score=float(rng.uniform(-2.0, 2.0)),
            weights={
                "price_change_5": float(rng.uniform(-5.0, 5.0)),
                "buy_ratio": float(rng.uniform(-5.0, 5.0)),
            },
            scaler=dataset.scaler,
        )
        strat = WeightedStrategy(strategy_config)
        mask = _compute_eligible(dataset, strategy_config)
        for i in np.flatnonzero(dataset._train_mask):
            assert bool(mask[i]) == strat.should_enter(dataset.snapshot_at(i))


def test_run_indexed_matches_run_random_params(tmp_path):
    df = _synthetic_features(4000, 12, 99)
    path = tmp_path / "features.parquet"
    df.write_parquet(path)
    dataset = SnapshotDataset(
        path, columns=["price_change_5", "price_change_20", "buy_ratio"]
    )
    rng = np.random.default_rng(99)

    for _ in range(6):
        sim_config = SimulatorConfig(
            position_size=float(rng.uniform(0.10, 0.50)),
            stop_loss=float(rng.uniform(0.08, 0.25)),
            take_profit=float(rng.uniform(0.50, 3.00)),
            trailing_trigger=float(rng.uniform(0.05, 2.00)),
            trailing_stop=float(rng.uniform(0.02, 0.80)),
            ttl_seconds=int(rng.integers(60, 600)),
            max_positions=int(rng.integers(1, 5)),
        )
        strategy_config = StrategyConfig(
            min_price_change_5=float(rng.uniform(-0.3, 0.5)),
            min_price_change_20=float(rng.uniform(-0.5, 1.0)),
            min_liquidity=float(rng.uniform(0.0, 5.0)),
            min_market_cap=float(rng.uniform(0.0, 50.0)),
            min_volume=float(rng.uniform(0.0, 2.0)),
            min_buy_ratio=float(rng.uniform(-1.0, 0.5)),
            min_trades=int(rng.integers(0, 100)),
            min_wallets=int(rng.integers(0, 50)),
            min_wallet_velocity=float(rng.uniform(-1.0, 2.0)),
            minimum_score=float(rng.uniform(-3.0, 3.0)),
            weights={
                "price_change_5": float(rng.uniform(-5.0, 5.0)),
                "price_change_20": float(rng.uniform(-5.0, 5.0)),
                "buy_ratio": float(rng.uniform(-5.0, 5.0)),
                "volume": float(rng.uniform(-5.0, 5.0)),
            },
            scaler=dataset.scaler,
        )
        _assert_identical(dataset, sim_config, strategy_config)


def test_run_indexed_matches_run_permissive(tmp_path):
    df = _synthetic_features(4000, 12, 5)
    path = tmp_path / "features.parquet"
    df.write_parquet(path)
    dataset = SnapshotDataset(path, columns=["price_change_5", "buy_ratio"])
    sim_config = SimulatorConfig(
        position_size=0.25,
        stop_loss=0.15,
        take_profit=1.0,
        trailing_trigger=0.5,
        trailing_stop=0.2,
        ttl_seconds=200,
        max_positions=2,
    )
    strategy_config = StrategyConfig(
        min_price_change_5=-100.0,
        min_price_change_20=-100.0,
        min_liquidity=-100.0,
        min_market_cap=-100.0,
        min_volume=-100.0,
        min_buy_ratio=-100.0,
        min_trades=-100,
        min_wallets=-100,
        min_wallet_velocity=-100.0,
        minimum_score=-100.0,
        weights={"price_change_5": 1.0, "buy_ratio": 0.5},
        scaler=dataset.scaler,
    )

    eligible = _compute_eligible(dataset, strategy_config)
    assert eligible.sum() > 0
    _assert_identical(dataset, sim_config, strategy_config)
