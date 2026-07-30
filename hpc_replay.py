from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from config import CONFIG
from constants import (
    CACHE_DIR,
    CPU_WORKERS,
    DOWNLOAD_DIR,
    DOWNLOAD_WORKERS,
    PARQUET_DIR,
    REPORT_DIR,
    OPTUNA_DB,
)
from download import ReplayDownloader
from logger import log
from paths import init_directories

app = typer.Typer(add_completion=False)
console = Console()


def _rm(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    log.info(f"Cleaned {path}")


def _parquet_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} events"),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _indeterminate_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        TimeElapsedColumn(),
    )


@app.command()
def run(
    days: int = 3,
    trials: int = 2000,
    workers: int = CPU_WORKERS,
    download_workers: int = DOWNLOAD_WORKERS,
    clean: bool = True,
    resume: bool = False,
    skip_download: bool = False,
    skip_parse: bool = False,
    skip_features: bool = False,
    skip_dataset: bool = False,
    skip_selection: bool = False,
    skip_optuna: bool = False,
):
    """Run the full HPC replay analysis pipeline."""
    init_directories()

    print("[bold cyan]Replay Optuna Pipeline[/]")
    print(f"  Days      : {days}")
    print(f"  Trials    : {trials}")
    print(f"  Workers   : {workers}")
    print(f"  Auto-clean: {clean}")
    print(f"  Storage   : sqlite:///{OPTUNA_DB}")
    print(f"  Resume    : {resume}")
    print()

    if not skip_download:
        log.info("Stage 1/6: Downloading replay...")
        asyncio.run(ReplayDownloader.run(days=days, workers=download_workers))
        log.info("Download complete.")
    else:
        log.info("Skipping download.")

    if not skip_parse:
        log.info("Stage 2/6: Parsing replay to Parquet...")
        _parse_replay()
        log.info("Parsing complete.")
        if clean:
            _rm(DOWNLOAD_DIR)
    else:
        log.info("Skipping parse.")

    if not skip_features:
        log.info("Stage 3/6: Building features...")
        _build_features()
        log.info("Features complete.")
        if clean:
            _rm(PARQUET_DIR)
    else:
        log.info("Skipping feature building.")

    if not skip_dataset:
        log.info("Stage 4/6: Building labeled dataset...")
        _build_dataset()
        log.info("Dataset complete.")
    else:
        log.info("Skipping dataset.")

    if not skip_selection:
        log.info("Stage 5/6: Selecting features...")
        _select_features()
        log.info("Selection complete.")
        if clean:
            _rm(CACHE_DIR / "training_dataset.parquet")
    else:
        log.info("Skipping feature selection.")

    if not skip_optuna:
        log.info("Stage 6/6: Running Optuna optimization...")
        _run_optuna(trials, workers, resume)
        log.info("Optuna complete.")
    else:
        log.info("Skipping Optuna.")

    log.info("Pipeline complete.")
    print("\n[bold green]✓ Pipeline finished successfully[/]")


def _parse_replay() -> None:
    import polars as pl
    from parser import replay_to_parquet

    parquet_path = PARQUET_DIR / "replay.parquet"
    if parquet_path.exists():
        log.info("Parquet cache exists, skipping.")
        return

    total_events = 0
    for f in sorted(DOWNLOAD_DIR.rglob("*.jsonl.zst")):
        total_events += 1

    with _parquet_progress() as progress:
        task = progress.add_task("Parsing", total=total_events)
        replay_to_parquet(DOWNLOAD_DIR, parquet_path, batch_size=100_000, progress=progress, task_id=task)

    log.info(f"Parquet saved: {parquet_path}")


def _build_features() -> None:
    import polars as pl
    from feature_engine import build_features_from_parquet

    parquet_path = PARQUET_DIR / "replay.parquet"
    if not parquet_path.exists():
        log.error(f"Parquet file not found: {parquet_path}")
        sys.exit(1)

    features_path = CACHE_DIR / "features.parquet"
    if features_path.exists():
        log.info("Features cache exists, skipping.")
        return

    df = pl.read_parquet(parquet_path)
    total_events = len(df)

    with _parquet_progress() as progress:
        task = progress.add_task("Features", total=total_events)
        features_df = build_features_from_parquet(df, progress=progress, task_id=task)

    features_df.write_parquet(features_path)
    log.info(f"Features saved: {features_path} ({len(features_df)} rows)")


def _build_dataset() -> None:
    import polars as pl

    from dataset_builder import DatasetBuilder
    from feature_engine import FeatureSnapshot
    from simulator import SimulatorConfig, WeightedStrategy, StrategyConfig

    features_path = CACHE_DIR / "features.parquet"
    if not features_path.exists():
        log.error(f"Features not found: {features_path}")
        sys.exit(1)

    df = pl.read_parquet(features_path)
    ignore = {"mint", "timestamp", "slot"}
    feature_cols = [c for c in df.columns if c not in ignore]

    with _parquet_progress() as progress:
        task = progress.add_task("Loading snapshots", total=len(df))
        snapshots = []
        for row in df.iter_rows(named=True):
            snapshots.append(
                FeatureSnapshot(
                    mint=row["mint"],
                    timestamp=row["timestamp"],
                    slot=row["slot"],
                    features={c: row[c] for c in feature_cols},
                )
            )
            progress.advance(task)

    strategy = WeightedStrategy(
        StrategyConfig(
            min_price_change_5=-0.30,
            min_liquidity=1_000,
            max_liquidity=5_000_000,
            min_market_cap=5_000,
            max_market_cap=10_000_000,
            min_volume=100,
            min_buy_ratio=0.10,
            min_trades=1,
            min_wallets=1,
            minimum_score=0.0,
        )
    )

    builder = DatasetBuilder(SimulatorConfig(), strategy)

    with _parquet_progress() as progress:
        task = progress.add_task("Evaluating candidates", total=len(snapshots))
        dataset = builder.build(snapshots, progress=progress, task_id=task)

    dataset_path = CACHE_DIR / "training_dataset.parquet"
    rows = [{**s.features, "label": s.label} for s in dataset.samples]
    if rows:
        df = pl.DataFrame(rows)
        df.write_parquet(dataset_path)
        log.info(f"Dataset saved: {dataset_path} ({len(df)} rows)")
    else:
        log.warning("No samples generated.")


def _select_features() -> None:
    import polars as pl

    from dataset_builder import TrainingDataset
    from feature_selection import FeatureSelector

    dataset_path = CACHE_DIR / "training_dataset.parquet"
    if not dataset_path.exists():
        log.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pl.read_parquet(dataset_path)
    dicts = df.to_dicts()

    with _parquet_progress() as progress:
        task = progress.add_task("Loading dataset", total=len(dicts))
        dataset = TrainingDataset()
        for d in dicts:
            features = {k: v for k, v in d.items() if k != "label"}
            dataset.samples.append(
                type("Sample", (), {"features": features, "label": d["label"]})()
            )
            progress.advance(task)

    selector = FeatureSelector()

    with _indeterminate_progress() as progress:
        task = progress.add_task("Training Random Forest...", total=None)
        result = selector.select(dataset)
        progress.update(task, description=f"Selected {len(result.features)} features")

    selection_path = CACHE_DIR / "selected_features.json"
    selection_path.write_text(json.dumps(result.features, indent=2))

    importance_path = REPORT_DIR / "feature_importance.json"
    importance_path.write_text(
        json.dumps(
            [
                {"feature": f.name, "importance": round(f.importance, 6)}
                for f in result.importance
            ],
            indent=2,
        )
    )

    log.info(f"Selected {len(result.features)} features: {result.features}")


def _run_optuna(trials: int, workers: int, resume: bool) -> None:
    from optuna_engine import Optimizer, OptunaConfig

    features_path = CACHE_DIR / "selected_features.json"
    selected_features: list[str] | None = None
    if features_path.exists():
        selected_features = json.loads(features_path.read_text())

    dataset_path = CACHE_DIR / "features.parquet"
    if not dataset_path.exists():
        log.error(f"Features dataset not found: {dataset_path}")
        sys.exit(1)

    config = OptunaConfig(
        dataset=dataset_path,
        output_dir=REPORT_DIR,
        study_name="replay_optuna",
        storage=f"sqlite:///{OPTUNA_DB}",
        trials=trials,
        jobs=workers,
        seed=CONFIG.random_seed,
        selected_features=selected_features,
    )

    optimizer = Optimizer(config)
    result = optimizer.run()

    log.info(f"Best score: {result.score:.4f} (trial {result.trial})")
    log.info(f"Profit factor: {result.metrics['profit_factor']:.2f}")
    log.info(f"Win rate: {result.metrics['win_rate']:.2%}")
    log.info(f"Max drawdown: {result.metrics['drawdown']:.2%}")
    log.info(f"Trades: {result.metrics['trades']}")


@app.command()
def download(
    days: int = CONFIG.days,
    workers: int = DOWNLOAD_WORKERS,
):
    """Download replay archives only."""
    init_directories()
    log.info(f"Downloading {days} days of replay data...")
    asyncio.run(ReplayDownloader.run(days=days, workers=workers))
    log.info("Done.")


@app.command()
def parse():
    """Parse downloaded replay to Parquet only."""
    init_directories()
    _parse_replay()


@app.command()
def features():
    """Build features only."""
    init_directories()
    _build_features()


@app.command()
def dataset():
    """Build labeled training dataset."""
    init_directories()
    _build_dataset()


@app.command()
def select():
    """Run feature selection."""
    init_directories()
    _select_features()


@app.command()
def optimize(
    trials: int = CONFIG.trials,
    workers: int = CPU_WORKERS,
    resume: bool = False,
):
    """Run Optuna optimization only."""
    init_directories()
    _run_optuna(trials, workers, resume)


if __name__ == "__main__":
    app()
