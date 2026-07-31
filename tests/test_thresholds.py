import random
from pathlib import Path
from types import SimpleNamespace

from optuna_engine import Objective, OptunaConfig


class StubTrial:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def suggest_float(self, name: str, low: float, high: float) -> float:
        return low + (high - low) * self.rng.random()

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return self.rng.randint(low, high)


def _objective():
    config = OptunaConfig(
        dataset=Path("dataset.parquet"),
        output_dir=Path("reports"),
        selected_features=[],
    )
    dataset = SimpleNamespace(scaler={})
    return Objective(config, dataset)


def test_min_liquidity_never_exceeds_max():
    rng = random.Random(0)
    objective = _objective()
    for _ in range(200):
        cfg = objective._strategy(StubTrial(rng))
        assert cfg.min_liquidity <= cfg.max_liquidity


def test_min_market_cap_never_exceeds_max():
    rng = random.Random(1)
    objective = _objective()
    for _ in range(200):
        cfg = objective._strategy(StubTrial(rng))
        assert cfg.min_market_cap <= cfg.max_market_cap
