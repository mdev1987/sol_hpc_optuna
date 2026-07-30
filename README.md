# Replay Optuna

HPC PumpAPI replay analysis pipeline: download, parse, feature engineer, simulate, and optimize trading strategies with Optuna.

## Pipeline

```
download.py  →  parser.py  →  feature_engine.py  →  simulator.py  →  optuna_engine.py
                                                    ↓
                                              dataset_builder.py
                                                    ↓
                                              feature_selection.py
                                                    ↓
                                              optuna_engine.py  →  best_strategy.json
```

## Quick Start

```bash
uv sync
uv run replay-optuna run --days 3 --trials 5000
```

## Commands

| Command | Description |
|---------|-------------|
| `run` | Full pipeline (download → parse → features → dataset → selection → optuna) |
| `download` | Download replay archives only |
| `parse` | Parse downloaded replay to Parquet |
| `features` | Build features from Parquet |
| `dataset` | Build labeled training dataset |
| `select` | Run feature selection (Random Forest + correlation filter) |
| `optimize` | Run Optuna optimization |

## Architecture

| Module | Responsibility |
|--------|---------------|
| `download.py` | Async download of `.jsonl.zst` archives from PumpAPI replay |
| `parser.py` | Streaming decompress → parse → Parquet with batch compaction |
| `feature_engine.py` | ~35 features per token: price, liquidity, volume, wallet, momentum |
| `simulator.py` | Portfolio with stop-loss, trailing stop, take-profit, TTL |
| `dataset_builder.py` | Labeled dataset generation for feature selection |
| `feature_selection.py` | RF importance + correlation filter + low-variance filter |
| `optuna_engine.py` | Distributed Optuna with TPESampler, MedianPruner, ~20 search dimensions |
| `config.py` | Pydantic config |
| `paths.py` | Directory initialization |
| `constants.py` | All path and numeric constants |
| `logger.py` | Rich console logging |
| `utils.py` | Date utilities |

## Outputs

```
reports/
├── best_strategy.json    # Best parameters + metrics
├── feature_importance.json
cache/
├── features.parquet
├── training_dataset.parquet
├── selected_features.json
optuna.db                 # SQLite study (resume compatible)
```

## Hardware

Optimized for 32-core / 64 GB RAM servers. Uses `n_jobs=-1` for Optuna parallelism.
