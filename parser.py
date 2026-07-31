"""
parser.py

Streaming PumpAPI replay parser.

Pipeline
--------
.jsonl.zst
      │
      ▼
Zstandard stream
      │
      ▼
Buffered line reader
      │
      ▼
orjson
      │
      ▼
ReplayEvent
      │
      ▼
Feature Engine

Characteristics
---------------
* Streaming (constant memory)
* Zero-copy where possible
* Async friendly
* Polars compatible
* Type safe
* Production ready
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Iterable

import io

import orjson
import polars as pl
import zstandard as zstd

# ------------------------------------------------------------
# Event Model
# ------------------------------------------------------------


@dataclass(slots=True)
class ReplayEvent:
    """
    Normalized replay event.

    Unknown fields are preserved inside `raw`
    so future parser versions remain compatible.
    """

    timestamp: int

    signature: str

    slot: int

    mint: str

    trader: str

    side: str

    amount: float

    price: float

    market_cap: float

    liquidity: float

    raw: dict[str, Any]


# ------------------------------------------------------------
# Reader
# ------------------------------------------------------------


def is_valid_zstd(path: str | Path) -> bool:
    """
    True if the file is a complete, non-corrupt Zstandard stream.

    Decompresses the whole stream (discarding output) so truncation or
    corruption anywhere in the file is caught, not just at the start.
    """

    try:
        with Path(path).open("rb") as fp:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fp) as reader:
                while reader.read(4 * 1024 * 1024):
                    pass
        return True
    except zstd.ZstdError:
        return False
    except OSError:
        return False


class ReplayReader:
    """
    Streaming Zstandard reader.

    One JSON line is held in memory at a time.
    """

    def __init__(
        self,
        file: str | Path,
        buffer_size: int = 1024 * 1024,
    ):

        self.path = Path(file)

        self.buffer_size = buffer_size

    def open(self) -> BinaryIO:

        fp = self.path.open("rb")

        dctx = zstd.ZstdDecompressor()

        return io.BufferedReader(
            dctx.stream_reader(fp),
            buffer_size=self.buffer_size,
        )


# ------------------------------------------------------------
# JSON
# ------------------------------------------------------------


class JsonDecoder:
    @staticmethod
    def loads(
        line: bytes,
    ) -> dict[str, Any]:

        return orjson.loads(line)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def get_int(
    data: dict[str, Any],
    *keys: str,
    default: int = 0,
) -> int:

    for key in keys:
        if key in data and data[key] is not None:
            return int(data[key])

    return default


def get_float(
    data: dict[str, Any],
    *keys: str,
    default: float = 0.0,
) -> float:

    for key in keys:
        if key in data and data[key] is not None:
            return float(data[key])

    return default


def get_str(
    data: dict[str, Any],
    *keys: str,
    default: str = "",
) -> str:

    for key in keys:
        if key in data and data[key] is not None:
            return str(data[key])

    return default


# ------------------------------------------------------------
# Event Parser
# ------------------------------------------------------------


class EventParser:
    """
    Converts raw JSON dictionaries into ReplayEvent objects.

    Only trade events (action ``buy`` / ``sell``) are parsed; non-trade
    actions (transfer, create, createPool, claimCreatorFees, ...) yield
    ``None`` so they are excluded from the replay stream.
    """

    TRADE_ACTIONS = {"buy", "sell"}

    @staticmethod
    def parse(
        raw: dict[str, Any],
    ) -> ReplayEvent | None:

        action = get_str(raw, "action", "").lower()

        if action not in EventParser.TRADE_ACTIONS:
            return None

        return ReplayEvent(
            timestamp=get_int(raw, "timestamp", "time") // 1000,
            signature=get_str(raw, "signature"),
            slot=get_int(raw, "block", "slot"),
            mint=get_str(raw, "mint"),
            trader=get_str(raw, "txSigner", "trader", "wallet"),
            side=action,
            amount=get_float(raw, "quoteAmount", "amount"),
            price=get_float(raw, "price"),
            market_cap=get_float(
                raw,
                "marketCapQuote",
                "market_cap",
                "marketCap",
            ),
            liquidity=get_float(
                raw,
                "quoteInPool",
                "liquidity",
            ),
            raw=raw,
        )


# ------------------------------------------------------------
# Replay Iterator
# ------------------------------------------------------------


class ReplayIterator:
    """
    Streaming iterator over a replay archive.

    Example
    -------
    reader = ReplayReader("00.jsonl.zst")

    for event in ReplayIterator(reader):
        ...
    """

    def __init__(
        self,
        reader: ReplayReader,
    ):

        self.reader = reader

    def __iter__(self) -> Iterator[ReplayEvent]:

        with self.reader.open() as fp:
            for line in fp:
                line = line.strip()

                if not line:
                    continue

                try:
                    raw = JsonDecoder.loads(line)

                    event = EventParser.parse(raw)

                    if event is None:
                        #
                        # Ignore non-trade actions.
                        #
                        continue

                    yield event

                except Exception:
                    #
                    # Ignore malformed JSON lines.
                    #
                    continue


# ------------------------------------------------------------
# Replay File
# ------------------------------------------------------------


class ReplayFile:
    def __init__(
        self,
        file: str | Path,
    ):

        self.path = Path(file)

    def events(self) -> Iterator[ReplayEvent]:

        reader = ReplayReader(self.path)

        return iter(ReplayIterator(reader))

    def __iter__(self):

        return self.events()

    def __len__(self):

        count = 0

        for _ in self.events():
            count += 1

        return count


# ------------------------------------------------------------
# Replay Dataset
# ------------------------------------------------------------


class ReplayDataset:
    """
    Iterate through multiple replay files
    as a single event stream.
    """

    def __init__(
        self,
        files: list[str | Path],
    ):

        self.files = [Path(f) for f in files]

    def __iter__(self):

        for file in self.files:
            replay = ReplayFile(file)

            yield from replay


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def replay_files(
    directory: str | Path,
):

    directory = Path(directory)

    yield from sorted(directory.rglob("*.jsonl.zst"))


def load_replay(
    directory: str | Path,
) -> ReplayDataset:

    return ReplayDataset(list(replay_files(directory)))


# Example
# from parser import load_replay

# for event in load_replay("downloads"):
#     print(
#         event.timestamp,
#         event.mint,
#         event.price,
#     )


# ------------------------------------------------------------
# Filtering
# ------------------------------------------------------------

from collections.abc import Callable


EventFilter = Callable[[ReplayEvent], bool]


class FilteredReplayDataset:
    """
    Lazy filtered replay dataset.

    No additional memory is allocated.
    """

    def __init__(
        self,
        dataset: ReplayDataset,
        predicate: EventFilter,
    ):

        self.dataset = dataset
        self.predicate = predicate

    def __iter__(self):

        for event in self.dataset:
            if self.predicate(event):
                yield event


# ------------------------------------------------------------
# Event Predicates
# ------------------------------------------------------------


def by_mint(
    mint: str,
) -> EventFilter:

    return lambda event: event.mint == mint


def by_wallet(
    wallet: str,
) -> EventFilter:

    return lambda event: event.trader == wallet


def by_side(
    side: str,
) -> EventFilter:

    side = side.upper()

    return lambda event: event.side.upper() == side


def minimum_liquidity(
    liquidity: float,
) -> EventFilter:

    return lambda event: event.liquidity >= liquidity


def minimum_market_cap(
    market_cap: float,
) -> EventFilter:

    return lambda event: event.market_cap >= market_cap


def timestamp_between(
    start: int,
    end: int,
) -> EventFilter:

    return lambda event: start <= event.timestamp <= end


# ------------------------------------------------------------
# Predicate Composition
# ------------------------------------------------------------


def AND(
    *predicates: EventFilter,
) -> EventFilter:

    def predicate(event: ReplayEvent):

        for fn in predicates:
            if not fn(event):
                return False

        return True

    return predicate


def OR(
    *predicates: EventFilter,
) -> EventFilter:

    def predicate(event: ReplayEvent):

        for fn in predicates:
            if fn(event):
                return True

        return False

    return predicate


def NOT(
    predicate: EventFilter,
) -> EventFilter:

    return lambda event: not predicate(event)


# ------------------------------------------------------------
# Dataset API
# ------------------------------------------------------------


def filter_replay(
    dataset: ReplayDataset,
    predicate: EventFilter,
) -> FilteredReplayDataset:

    return FilteredReplayDataset(
        dataset,
        predicate,
    )

    # ------------------------------------------------------------


# Statistics
# ------------------------------------------------------------


from dataclasses import dataclass


@dataclass(slots=True)
class ReplayStatistics:
    events: int = 0

    buys: int = 0

    sells: int = 0

    unique_wallets: int = 0

    unique_mints: int = 0


def replay_statistics(
    dataset,
) -> ReplayStatistics:

    wallets = set()

    mints = set()

    stats = ReplayStatistics()

    for event in dataset:
        stats.events += 1

        wallets.add(event.trader)

        mints.add(event.mint)

        if event.side.upper() == "BUY":
            stats.buys += 1

        elif event.side.upper() == "SELL":
            stats.sells += 1

    stats.unique_wallets = len(wallets)

    stats.unique_mints = len(mints)

    return stats


# Example
# dataset = load_replay("downloads")

# dataset = filter_replay(
#     dataset,
#     AND(
#         minimum_liquidity(5000),
#         minimum_market_cap(25000),
#     ),
# )

# stats = replay_statistics(dataset)

# print(stats)


# ------------------------------------------------------------
# Batch Iterator
# ------------------------------------------------------------


class ReplayBatchIterator:
    """
    Iterate replay events in fixed-size batches.

    This keeps memory bounded while allowing downstream
    feature engineering to process DataFrames instead of
    one event at a time.
    """

    def __init__(
        self,
        dataset: Iterable[ReplayEvent],
        batch_size: int = 100_000,
    ):

        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):

        batch: list[ReplayEvent] = []

        for event in self.dataset:
            batch.append(event)

            if len(batch) >= self.batch_size:
                yield batch

                batch = []

        if batch:
            yield batch


# ------------------------------------------------------------
# Polars
# ------------------------------------------------------------


def events_to_dicts(
    events: list[ReplayEvent],
):

    return [asdict(event) for event in events]


def events_to_dataframe(
    events: list[ReplayEvent],
) -> pl.DataFrame:

    return pl.from_dicts(events_to_dicts(events))


# ------------------------------------------------------------
# DataFrame Stream
# ------------------------------------------------------------


def dataframe_stream(
    dataset: Iterable[ReplayEvent],
    batch_size: int = 100_000,
):

    for batch in ReplayBatchIterator(
        dataset,
        batch_size,
    ):
        yield events_to_dataframe(batch)


# ------------------------------------------------------------
# Parquet Writer
# ------------------------------------------------------------


class ReplayParquetWriter:
    def __init__(
        self,
        output,
        compression: str = "snappy",
    ):

        self.output = output
        self.compression = compression

    @staticmethod
    def _primitive_columns(frame) -> list[str]:
        import polars as pl
        columns = [c for c in frame.columns if c != "raw"]
        kept: list[str] = []
        for c in columns:
            base = frame.schema[c].base_type()
            if base in (pl.Struct, pl.Object):
                continue
            kept.append(c)
        return kept

    def write(
        self,
        dataset,
        batch_size: int = 100_000,
        progress=None,
        task_id=None,
    ):

        import pyarrow.parquet as pq

        writer = None
        try:
            for frame in dataframe_stream(dataset, batch_size):
                cols = self._primitive_columns(frame)
                if not cols:
                    continue
                frame = frame.select(cols)
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        self.output,
                        table.schema,
                        compression=self.compression,
                    )
                writer.write_table(table)
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=len(frame))
        finally:
            if writer is not None:
                writer.close()


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------


def replay_to_parquet(
    replay_directory,
    output,
    batch_size: int = 100_000,
    progress=None,
    task_id=None,
):

    dataset = load_replay(replay_directory)

    writer = ReplayParquetWriter(output)

    writer.write(
        dataset,
        batch_size=batch_size,
        progress=progress,
        task_id=task_id,
    )


# Example
# from parser import replay_to_parquet

# replay_to_parquet(
#     "downloads",
#     "week.parquet",
# )


# ------------------------------------------------------------
# Replay Index
# ------------------------------------------------------------


@dataclass(slots=True)
class ReplayMetadata:
    file: Path

    events: int

    first_timestamp: int

    last_timestamp: int


class ReplayIndex:
    """
    Lightweight replay index.

    Used to inspect replay archives without
    loading them entirely.
    """

    def __init__(
        self,
        files: list[Path],
    ):

        self.files = files

    def build(self) -> list[ReplayMetadata]:

        metadata: list[ReplayMetadata] = []

        for file in self.files:
            first = None
            last = None
            count = 0

            for event in ReplayFile(file):
                count += 1

                if first is None:
                    first = event.timestamp

                last = event.timestamp

            metadata.append(
                ReplayMetadata(
                    file=file,
                    events=count,
                    first_timestamp=first or 0,
                    last_timestamp=last or 0,
                )
            )

        return metadata


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------


class ReplayValidator:
    REQUIRED_FIELDS = (
        "timestamp",
        "mint",
        "price",
        "side",
    )

    @classmethod
    def validate(
        cls,
        event: ReplayEvent,
    ) -> bool:

        if event.timestamp <= 0:
            return False

        if not event.mint:
            return False

        if event.price <= 0:
            return False

        if not event.side:
            return False

        return True


# ------------------------------------------------------------
# Valid Dataset
# ------------------------------------------------------------


class ValidReplayDataset:
    def __init__(
        self,
        dataset,
    ):

        self.dataset = dataset

    def __iter__(self):

        for event in self.dataset:
            if ReplayValidator.validate(event):
                yield event


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def validated_replay(
    directory,
):

    return ValidReplayDataset(load_replay(directory))


def replay_index(
    directory,
):

    return ReplayIndex(list(replay_files(directory))).build()


## Example
# dataset = validated_replay("downloads")

# for event in dataset:
#     print(event)
## Build Replay Index
# for meta in replay_index("downloads"):
#     print(
#         meta.file.name,
#         meta.events,
#     )
