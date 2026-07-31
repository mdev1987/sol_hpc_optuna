from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from feature_engine import FeatureSnapshot
from simulator import ExitReason, SimulatorConfig, Simulator, WeightedStrategy


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
    def __init__(self, config: SimulatorConfig, strategy: WeightedStrategy | None = None):
        self.config = config
        self.strategy = strategy

    def build(self, features_df, progress=None, task_id=None, output=None) -> int:
        """
        Build a labeled training dataset from a features DataFrame.

        Labels are forward-looking: a sample is positive (1) if its price
        reaches `take_profit` within `ttl_seconds`, computed per-mint with
        a two-pointer scan (O(n) per mint, no per-snapshot simulator).
        """
        import numpy as np
        import polars as pl

        labels = self._compute_labels(features_df)

        meta = {"mint", "timestamp", "slot"}
        feature_cols = [c for c in features_df.columns if c not in meta and c != "label"]

        writer = None
        written = 0
        buffer: list[dict] = []
        batch_rows = 100_000

        def flush() -> None:
            nonlocal writer, written
            if not buffer:
                return
            import pyarrow.parquet as pq

            frame = pl.from_dicts(buffer)
            table = frame.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="snappy")
            writer.write_table(table)
            written += len(frame)
            buffer.clear()

        try:
            for row, label in zip(features_df.iter_rows(named=True), labels, strict=False):
                if progress is not None and task_id is not None:
                    progress.advance(task_id)

                sample = {c: row[c] for c in feature_cols}
                sample["label"] = int(label)
                buffer.append(sample)
                if len(buffer) >= batch_rows:
                    flush()
            flush()
        finally:
            if writer is not None:
                writer.close()

        return written

    def _compute_labels(self, features_df) -> list[int]:
        import numpy as np

        mints = features_df["mint"].to_numpy()
        timestamps = features_df["timestamp"].to_numpy()
        prices = features_df["price"].to_numpy()

        n = len(features_df)
        labels = np.zeros(n, dtype=np.int8)
        ttl = self.config.ttl_seconds
        target = 1.0 + self.config.take_profit

        order = np.lexsort((timestamps, mints))
        sorted_mints = mints[order]
        sorted_ts = timestamps[order]
        sorted_price = prices[order]
        sorted_idx = order

        i = 0
        while i < n:
            j = i
            while j + 1 < n and sorted_mints[j + 1] == sorted_mints[i]:
                j += 1
            self._label_group(
                sorted_ts[i : j + 1],
                sorted_price[i : j + 1],
                sorted_idx[i : j + 1],
                labels,
                ttl,
                target,
            )
            i = j + 1

        return labels.tolist()

    @staticmethod
    def _label_group(ts, prices, idx, labels, ttl, target) -> None:
        m = len(ts)
        if m < 2:
            return
        j = 0
        for i in range(m):
            if j < i:
                j = i
            while j + 1 < m and ts[j + 1] - ts[i] <= ttl:
                j += 1
            if j > i:
                window = prices[i + 1 : j + 1]
                max_gain = window.max() / prices[i] - 1.0
                if max_gain >= target - 1.0:
                    labels[idx[i]] = 1


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
