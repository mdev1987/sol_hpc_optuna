from __future__ import annotations

import random
from dataclasses import dataclass, field

from simulator import SimulatorConfig


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
    def __init__(self, config: SimulatorConfig):
        self.config = config

    def build(self, features_df, progress=None, task_id=None, output=None, balanced: bool = True) -> int:
        """
        Build a labeled training dataset from a features DataFrame.

        Labels are forward-looking: a sample is positive (1) if its price
        reaches `take_profit` within `ttl_seconds`, computed per-mint with
        a two-pointer scan (O(n) per mint, no per-snapshot simulator).

        With `balanced=True`, the dataset is subsampled to equal positive
        and negative samples (shuffled) to avoid class imbalance.
        """
        import numpy as np
        import polars as pl

        labels = np.asarray(self._compute_labels(features_df), dtype=np.int8)

        if balanced and labels.size:
            pos = np.flatnonzero(labels == 1)
            neg = np.flatnonzero(labels == 0)
            if pos.size and neg.size:
                count = min(pos.size, neg.size)
                rng = random.Random(42)
                keep = np.concatenate(
                    (rng.sample(pos.tolist(), count), rng.sample(neg.tolist(), count))
                )
                keep = np.sort(keep)
                features_df = features_df[keep]
                labels = labels[keep]

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
                base = prices[i]
                if not (base > 0):
                    continue
                window = prices[i + 1 : j + 1]
                max_gain = window.max() / base - 1.0
                if max_gain >= target - 1.0:
                    labels[idx[i]] = 1
