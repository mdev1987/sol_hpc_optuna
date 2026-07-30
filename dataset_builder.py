from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from feature_engine import FeatureSnapshot
from simulator import Side, ExitReason, SimulatorConfig, Simulator, WeightedStrategy, StrategyConfig


@dataclass(slots=True)
class TrainingSample:
    features: dict[str, float]
    label: int  # 1 if profitable exit, 0 otherwise


@dataclass(slots=True)
class TrainingDataset:
    samples: list[TrainingSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, snapshot: FeatureSnapshot) -> None:
        self.samples.append(TrainingSample(features=dict(snapshot.features), label=0))

    def add_profitable(self, snapshot: FeatureSnapshot) -> None:
        self.samples.append(TrainingSample(features=dict(snapshot.features), label=1))


class DatasetBuilder:
    def __init__(self, config: SimulatorConfig, strategy: WeightedStrategy):
        self.config = config
        self.strategy = strategy

    def build(self, snapshots: Iterable[FeatureSnapshot], progress=None, task_id=None) -> TrainingDataset:
        dataset = TrainingDataset()

        for snapshot in snapshots:
            if progress is not None and task_id is not None:
                progress.advance(task_id)

            score = self.strategy.score(snapshot)
            if score < self.strategy.config.minimum_score:
                continue

            result = self._simulate_single(snapshot)
            if result > 0:
                dataset.add_profitable(snapshot)
            else:
                dataset.add(snapshot)

        return dataset

    def _simulate_single(self, snapshot: FeatureSnapshot) -> float:
        simulator = Simulator(self.config, self.strategy)
        simulator.portfolio.open_position(snapshot)
        price = snapshot.features["price"]
        timestamp = snapshot.timestamp + self.config.ttl_seconds
        simulator.portfolio.close_position(
            snapshot.mint, price, timestamp, ExitReason.TTL
        )
        return simulator.portfolio.closed[-1].pnl


class BalancedDatasetBuilder:
    def __init__(self, config: SimulatorConfig, strategy: WeightedStrategy):
        self.config = config
        self.strategy = strategy

    def build(self, snapshots: Iterable[FeatureSnapshot]) -> TrainingDataset:
        profitable: list[TrainingSample] = []
        unprofitable: list[TrainingSample] = []

        for snapshot in snapshots:
            score = self.strategy.score(snapshot)
            if score < self.strategy.config.minimum_score:
                continue

            sample = TrainingSample(features=dict(snapshot.features), label=0)
            result = self._simulate_single(snapshot)
            if result > 0:
                sample.label = 1
                profitable.append(sample)
            else:
                unprofitable.append(sample)

        count = min(len(profitable), len(unprofitable)) if profitable and unprofitable else 0
        if count == 0:
            combined = profitable + unprofitable
            return TrainingDataset(samples=combined)

        random.shuffle(profitable)
        random.shuffle(unprofitable)

        balanced = profitable[:count] + unprofitable[:count]
        random.shuffle(balanced)

        return TrainingDataset(samples=balanced)

    def _simulate_single(self, snapshot: FeatureSnapshot) -> float:
        simulator = Simulator(self.config, self.strategy)
        simulator.portfolio.open_position(snapshot)
        price = snapshot.features["price"]
        timestamp = snapshot.timestamp + self.config.ttl_seconds
        simulator.portfolio.close_position(
            snapshot.mint, price, timestamp, ExitReason.TTL
        )
        return simulator.portfolio.closed[-1].pnl
