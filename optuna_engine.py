from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna

from feature_engine import FeatureSnapshot
from simulator import Simulator, SimulatorConfig, StrategyConfig, WeightedStrategy

# Module-level handle to the shared feature matrix. Set in the parent
# process before forking; children inherit it via copy-on-write, so the
# ~8GB dataset is shared (not duplicated once per worker).
_DATASET: SnapshotDataset | None = None


def _finite(value: float, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    try:
        if not math.isfinite(value):
            return fallback
    except TypeError:
        return fallback
    return value


@dataclass(slots=True)
class OptunaConfig:
    dataset: Path
    output_dir: Path
    study_name: str = "replay"
    storage: str = "sqlite:///study.db"
    trials: int = 5000
    timeout: int | None = None
    direction: str = "maximize"
    jobs: int = -1
    seed: int = 42
    selected_features: list[str] | None = None
    validation_fraction: float = 0.0
    sample_fraction: float = 1.0


@dataclass(slots=True)
class OptimizationResult:
    score: float
    trial: int
    parameters: dict
    metrics: dict


class SnapshotDataset:
    def __init__(
        self,
        path: Path,
        validation_fraction: float = 0.0,
        sample_fraction: float = 1.0,
    ):
        import polars as pl

        self.frame = pl.read_parquet(path)
        self.ignore = {"mint", "timestamp", "slot"}
        self.feature_columns = [c for c in self.frame.columns if c not in self.ignore]

        # Sub-sample whole mints (never individual events) so per-mint price
        # paths stay intact for the simulator's exit logic. This shrinks both
        # per-trial runtime and the in-memory matrix, cutting VPS cost.
        self.sample_fraction = min(max(sample_fraction, 0.0), 1.0)
        if self.sample_fraction < 1.0:
            import random

            mints = self.frame["mint"].unique().to_list()
            keep_n = max(1, round(len(mints) * self.sample_fraction))
            kept = set(random.Random(42).sample(mints, keep_n))
            self.frame = self.frame.filter(pl.col("mint").is_in(kept))
        self._mints = self.frame["mint"].to_list()
        self._timestamps = self.frame["timestamp"].to_numpy()
        self._slots = self.frame["slot"].to_numpy()
        features_frame = self.frame.select(self.feature_columns)
        self._features = np.nan_to_num(features_frame.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

        n = len(self._timestamps)
        self.validation_fraction = min(max(validation_fraction, 0.0), 0.5)
        self._train_mask = np.ones(n, dtype=bool)
        self._val_mask = np.zeros(n, dtype=bool)
        if self.validation_fraction > 0.0:
            cutoff = np.quantile(np.sort(self._timestamps), 1.0 - self.validation_fraction)
            self._val_mask = self._timestamps > cutoff
            self._train_mask = ~self._val_mask

        train_features = self._features[self._train_mask]
        means = train_features.mean(axis=0)
        stds = train_features.std(axis=0)
        self.scaler = {
            col: (float(means[j]), float(stds[j]))
            for j, col in enumerate(self.feature_columns)
        }

    def _iter(self, mask: np.ndarray):
        indexes = np.flatnonzero(mask)
        for i in indexes:
            yield FeatureSnapshot(
                mint=self._mints[i],
                timestamp=int(self._timestamps[i]),
                slot=int(self._slots[i]),
                features={
                    feature: float(self._features[i, j])
                    for j, feature in enumerate(self.feature_columns)
                },
            )

    def snapshots(self):
        yield from self._iter(self._train_mask)

    def validation_snapshots(self):
        yield from self._iter(self._val_mask)


def _compute_score(sim_config: "SimulatorConfig", strategy_config: "StrategyConfig") -> tuple[float, dict]:
    """Worker-side entry point for the compute pool: run the simulator.

    Uses the module-level ``_DATASET`` (inherited via fork copy-on-write),
    so the ~8GB feature matrix stays shared instead of duplicated per worker.
    """
    global _DATASET
    assert _DATASET is not None, "worker called before parent set _DATASET"

    simulator = Simulator(sim_config, WeightedStrategy(strategy_config))
    result = simulator.run(_DATASET.snapshots())

    if result.trades < 20:
        return -1e9, {
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "drawdown": 0.0,
            "trades": int(result.trades),
        }
    return score_simulation(result)


class Objective:
    def __init__(self, config: OptunaConfig, dataset: SnapshotDataset, pool=None):
        self.config = config
        self.dataset = dataset
        self.pool = pool

    def _strategy(self, trial: optuna.Trial) -> StrategyConfig:
        weights: dict[str, float] = {}
        features = self.config.selected_features or []

        for feature in features:
            weights[feature] = trial.suggest_float(f"w_{feature}", -5.0, 5.0)

        min_liquidity = trial.suggest_float("min_liquidity", 1.0, 2_000)
        min_market_cap = trial.suggest_float("min_market_cap", 10, 10_000)

        return StrategyConfig(
            min_price_change_5=trial.suggest_float("min_price_change_5", -0.30, 0.50),
            min_price_change_20=trial.suggest_float("min_price_change_20", -0.50, 1.00),
            min_liquidity=min_liquidity,
            max_liquidity=trial.suggest_float("max_liquidity", min_liquidity, 50_000),
            min_market_cap=min_market_cap,
            max_market_cap=trial.suggest_float("max_market_cap", min_market_cap, 200_000),
            min_volume=trial.suggest_float("min_volume", 0.5, 200),
            min_buy_ratio=trial.suggest_float("min_buy_ratio", 0.10, 0.95),
            min_trades=trial.suggest_int("min_trades", 1, 500),
            min_wallets=trial.suggest_int("min_wallets", 1, 300),
            min_wallet_velocity=trial.suggest_float("min_wallet_velocity", 0.0, 5.0),
            minimum_score=trial.suggest_float("minimum_score", -2.0, 2.0),
            weights=weights,
            scaler=self.dataset.scaler,
        )

    def _simulator_config(self, trial: optuna.Trial) -> SimulatorConfig:
        return SimulatorConfig(
            position_size=trial.suggest_float("position_size", 0.05, 1.0),
            stop_loss=trial.suggest_float("stop_loss", 0.03, 0.60),
            take_profit=trial.suggest_float("take_profit", 0.10, 5.00),
            trailing_trigger=trial.suggest_float("trailing_trigger", 0.05, 2.00),
            trailing_stop=trial.suggest_float("trailing_stop", 0.02, 0.80),
            ttl_seconds=trial.suggest_int("ttl_seconds", 5, 600),
            max_positions=trial.suggest_int("max_positions", 1, 10),
        )

    def _simulator(self, trial: optuna.Trial) -> Simulator:
        return Simulator(
            self._simulator_config(trial), WeightedStrategy(self._strategy(trial))
        )

    def _evaluate(self, trial, sim_config, strategy_config) -> float:
        """Run one parameter set, returning the objective score.

        With a compute pool, the parent process (the sole SQLite writer)
        samples the trial, submits the picklable configs to the pool, and
        only collects the result — worker processes never touch storage.
        """
        if self.pool is not None:
            score, metrics = self.pool.apply(
                _compute_score, (sim_config, strategy_config)
            )
        else:
            simulator = Simulator(sim_config, WeightedStrategy(strategy_config))
            result = simulator.run(self.dataset.snapshots())
            if result.trades < 20:
                return -1e9
            score, metrics = score_simulation(result)

        trial.set_user_attr("profit_factor", metrics["profit_factor"])
        trial.set_user_attr("win_rate", metrics["win_rate"])
        trial.set_user_attr("drawdown", metrics["drawdown"])
        trial.set_user_attr("trades", int(metrics["trades"]))
        return score

    def __call__(self, trial: optuna.Trial) -> float:
        sim_config = self._simulator_config(trial)
        strategy_config = self._strategy(trial)
        return self._evaluate(trial, sim_config, strategy_config)


def score_simulation(result) -> tuple[float, dict]:
    """Compute the objective score and core metrics from a simulation result."""
    profit_factor = _finite(result.profit_factor)
    win_rate = _finite(result.win_rate)
    total_return = _finite(result.total_return)
    total_pnl = _finite(result.total_pnl)
    max_drawdown = _finite(result.max_drawdown)

    score = (
        2.5 * profit_factor
        + 1.5 * win_rate
        + 1.0 * total_return
        + 0.5 * total_pnl
        - 3.0 * max_drawdown
    )
    metrics = {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "drawdown": max_drawdown,
        "trades": int(result.trades),
    }
    return score, metrics


def _strategy_from_params(
    params: dict,
    selected_features: list[str],
    scaler: dict,
    validation_fraction: float = 0.0,
) -> tuple[StrategyConfig, SimulatorConfig]:
    weights: dict[str, float] = {}
    for feature in selected_features or []:
        weight = params.get(f"w_{feature}")
        if weight is not None:
            weights[feature] = float(weight)

    min_liquidity = float(params.get("min_liquidity", 1.0))
    min_market_cap = float(params.get("min_market_cap", 10.0))
    strategy_config = StrategyConfig(
        min_price_change_5=params.get("min_price_change_5"),
        min_price_change_20=params.get("min_price_change_20"),
        min_liquidity=min_liquidity,
        max_liquidity=params.get("max_liquidity", 50_000.0),
        min_market_cap=min_market_cap,
        max_market_cap=params.get("max_market_cap", 200_000.0),
        min_volume=params.get("min_volume"),
        min_buy_ratio=params.get("min_buy_ratio"),
        min_trades=params.get("min_trades"),
        min_wallets=params.get("min_wallets"),
        min_wallet_velocity=params.get("min_wallet_velocity"),
        minimum_score=float(params.get("minimum_score", 0.0)),
        weights=weights,
        scaler=scaler,
    )
    simulator_config = SimulatorConfig(
        position_size=float(params.get("position_size", 0.20)),
        stop_loss=float(params.get("stop_loss", 0.15)),
        take_profit=float(params.get("take_profit", 1.0)),
        trailing_trigger=float(params.get("trailing_trigger", 0.30)),
        trailing_stop=float(params.get("trailing_stop", 0.20)),
        ttl_seconds=int(params.get("ttl_seconds", 300)),
        max_positions=int(params.get("max_positions", 3)),
    )
    return strategy_config, simulator_config


def evaluate_params(
    params: dict,
    dataset: SnapshotDataset,
    selected_features: list[str],
    on_validation: bool = False,
) -> tuple[float, dict]:
    """Run a parameter set through the simulator, scoring train or held-out data."""
    strategy_config, simulator_config = _strategy_from_params(
        params, selected_features, dataset.scaler
    )
    simulator = Simulator(simulator_config, WeightedStrategy(strategy_config))
    snapshots = dataset.validation_snapshots() if on_validation else dataset.snapshots()
    result = simulator.run(snapshots)
    if result.trades < 20:
        return -1e9, {
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "drawdown": 0.0,
            "trades": int(result.trades),
        }
    return score_simulation(result)

def _is_transient_storage_error(exc: Exception) -> bool:
    """Storage failures worth retrying: SQLite lock/commit errors and the
    known optuna `_optimize.py` UnboundLocalError triggered by them."""
    import optuna.exceptions
    import sqlalchemy.exc

    if isinstance(exc, (optuna.exceptions.StorageInternalError, sqlalchemy.exc.OperationalError)):
        return True
    if isinstance(exc, UnboundLocalError):
        message = str(exc)
        return any(name in message for name in ("updated_state", "updated_sate", "frozen_trial"))
    return False


class Optimizer:
    def __init__(self, config: OptunaConfig):
        self.config = config
        self.dataset = SnapshotDataset(
            config.dataset,
            validation_fraction=config.validation_fraction,
            sample_fraction=config.sample_fraction,
        )

    def _storage(self) -> optuna.storages.RDBStorage:
        from optuna.storages import RDBStorage

        engine_kwargs = None
        if self.config.storage.startswith("sqlite:///"):
            db_path = self.config.storage.replace("sqlite:///", "", 1)
            try:
                import sqlite3

                con = sqlite3.connect(db_path, timeout=60)
                con.execute("PRAGMA journal_mode=WAL")
                con.close()
            except Exception:
                pass
            engine_kwargs = {"connect_args": {"timeout": 60}}
        return RDBStorage(url=self.config.storage, engine_kwargs=engine_kwargs)

    def study(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.config.seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=10)

        return optuna.create_study(
            study_name=self.config.study_name,
            storage=self._storage(),
            load_if_exists=True,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
        )

    def _run_sequential(self, objective: Objective) -> None:
        """Run all trials in this process, single-core."""
        for attempt in range(1, 6):
            try:
                study = self.study()
                study.optimize(
                    objective,
                    n_trials=self.config.trials,
                    timeout=self.config.timeout,
                    n_jobs=1,
                    show_progress_bar=True,
                )
                return
            except Exception as e:
                if not _is_transient_storage_error(e):
                    raise
                if attempt == 5:
                    raise
                print(f"\n[bold yellow]Transient storage error ({e}); retrying in 10s "
                      f"(attempt {attempt + 1}/5)...[/]")
                time.sleep(10)

    def _run_parallel(self, objective: Objective) -> None:
        """Optimize with a single writer process and a compute-only worker pool.

        Prior approach forked N processes, each running `study.optimize`
        against one SQLite study. RDBStorage + SQLite serializes writers,
        and N competing writers degenerated into a lock/retry loop: trials
        registered as RUNNING but never reached COMPLETE.

        Instead the parent process owns the study and is the *only* storage
        writer: `study.optimize(n_jobs=processes)` samples trials in-process
        (TPE is thread-safe) and each objective call submits picklable
        configs to a fork-based `Pool`. Worker processes run only the
        simulator against the COW-shared feature matrix and return the score;
        they never touch SQLite, so there is zero writer contention.
        """
        import multiprocessing as mp

        global _DATASET
        _DATASET = self.dataset

        processes = min(int(self.config.jobs) if self.config.jobs and self.config.jobs > 0 else 1,
                        os.cpu_count() or 1)

        # Ensure the study/database exists before starting the pool.
        self.study()

        ctx = mp.get_context("fork")
        pool = ctx.Pool(processes=processes)
        objective.pool = pool

        try:
            study = self.study()
            # One sampler drives all trials (single-writer TPE), so a single
            # seed suffices and proposals are consistent across the run.
            study.optimize(
                objective,
                n_trials=self.config.trials,
                timeout=self.config.timeout,
                n_jobs=processes,
                show_progress_bar=False,
            )
        except (BrokenPipeError, EOFError, OSError) as e:
            raise RuntimeError(
                f"Compute pool failed ({e}). Trials completed: {self._num_complete()}. "
                f"Completed trials persist in the study; re-run to resume."
            ) from e
        finally:
            pool.close()
            pool.join()

        done = self._num_complete()
        if done == 0:
            raise RuntimeError("Parallel optimization produced no complete trials.")
        print(f"  [bold cyan]{done}/{self.config.trials}[/] trials complete", flush=True)

    def _num_complete(self) -> int:
        study = self.study()
        return len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def run(self) -> OptimizationResult:
        objective = Objective(self.config, self.dataset)

        if self.config.jobs and self.config.jobs > 1:
            self._run_parallel(objective)
        else:
            self._run_sequential(objective)

        best = self.study().best_trial
        metrics = {
            "profit_factor": best.user_attrs.get("profit_factor"),
            "win_rate": best.user_attrs.get("win_rate"),
            "drawdown": best.user_attrs.get("drawdown"),
            "trades": best.user_attrs.get("trades"),
        }

        if self.dataset.validation_fraction > 0.0:
            val_score, val_metrics = evaluate_params(
                dict(best.params),
                self.dataset,
                self.config.selected_features or [],
                on_validation=True,
            )
            metrics["val_score"] = val_score
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        result = OptimizationResult(
            score=best.value,
            trial=best.number,
            parameters=dict(best.params),
            metrics=metrics,
        )
        self.save(result)
        return result

    def save(self, result: OptimizationResult) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output = {
            "score": result.score,
            "trial": result.trial,
            "parameters": result.parameters,
            "metrics": result.metrics,
        }
        name = self.config.study_name
        suffix = ""
        if name and name.startswith("replay_optuna_") and name != "replay_optuna":
            bundle = name.removeprefix("replay_optuna_")
            suffix = f"_{bundle}"
        path = self.config.output_dir / f"best_strategy{suffix}.json"
        path.write_text(json.dumps(output, indent=4))


def optimize(
    dataset: Path,
    output: Path,
    features: list[str],
    trials: int = 5000,
    jobs: int = -1,
    storage: str = "sqlite:///study.db",
) -> OptimizationResult:
    config = OptunaConfig(
        dataset=dataset,
        output_dir=output,
        trials=trials,
        jobs=jobs,
        storage=storage,
        selected_features=features,
    )
    return Optimizer(config).run()
