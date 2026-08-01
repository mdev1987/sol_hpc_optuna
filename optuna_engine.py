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


@dataclass(slots=True)
class OptimizationResult:
    score: float
    trial: int
    parameters: dict
    metrics: dict


class SnapshotDataset:
    def __init__(self, path: Path):
        import polars as pl

        self.frame = pl.read_parquet(path)
        self.ignore = {"mint", "timestamp", "slot"}
        self.feature_columns = [c for c in self.frame.columns if c not in self.ignore]
        self._mints = self.frame["mint"].to_list()
        self._timestamps = self.frame["timestamp"].to_numpy()
        self._slots = self.frame["slot"].to_numpy()
        features_frame = self.frame.select(self.feature_columns)
        self._features = np.nan_to_num(features_frame.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)

        means = self._features.mean(axis=0)
        stds = self._features.std(axis=0)
        self.scaler = {
            col: (float(means[j]), float(stds[j]))
            for j, col in enumerate(self.feature_columns)
        }

    def snapshots(self):
        for i in range(len(self._features)):
            yield FeatureSnapshot(
                mint=self._mints[i],
                timestamp=int(self._timestamps[i]),
                slot=int(self._slots[i]),
                features={
                    feature: float(self._features[i, j])
                    for j, feature in enumerate(self.feature_columns)
                },
            )


class Objective:
    def __init__(self, config: OptunaConfig, dataset: SnapshotDataset):
        self.config = config
        self.dataset = dataset

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

    def _simulator(self, trial: optuna.Trial) -> Simulator:
        simulator_config = SimulatorConfig(
            position_size=trial.suggest_float("position_size", 0.05, 1.0),
            stop_loss=trial.suggest_float("stop_loss", 0.03, 0.60),
            take_profit=trial.suggest_float("take_profit", 0.10, 5.00),
            trailing_trigger=trial.suggest_float("trailing_trigger", 0.05, 2.00),
            trailing_stop=trial.suggest_float("trailing_stop", 0.02, 0.80),
            ttl_seconds=trial.suggest_int("ttl_seconds", 5, 600),
            max_positions=trial.suggest_int("max_positions", 1, 10),
        )
        strategy = WeightedStrategy(self._strategy(trial))
        return Simulator(simulator_config, strategy)

    def __call__(self, trial: optuna.Trial) -> float:
        simulator = self._simulator(trial)
        result = simulator.run(self.dataset.snapshots())

        if result.trades < 20:
            return -1e9

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

        trial.set_user_attr("profit_factor", profit_factor)
        trial.set_user_attr("win_rate", win_rate)
        trial.set_user_attr("drawdown", max_drawdown)
        trial.set_user_attr("trades", int(result.trades))

        return score

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


def _worker_main(config: OptunaConfig, trials: int, seed: int) -> None:
    """Entry point for each forked Optuna worker process.

    Runs ``study.optimize(n_jobs=1)`` against the shared SQLite study.
    The feature matrix is read from the module-level ``_DATASET``, which
    the parent populated before forking (shared via copy-on-write).
    """
    global _DATASET

    assert _DATASET is not None, "worker called before parent set _DATASET"

    objective = Objective(config, _DATASET)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=10)

    for attempt in range(1, 6):
        try:
            study = optuna.load_study(
                study_name=config.study_name,
                storage=config.storage,
                pruner=pruner,
            )
            # `load_if_exists` restores the stored sampler, ignoring the one
            # passed to create_study. Override per-worker so TPE RNGs diverge;
            # otherwise every worker proposes identical trials.
            study.sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study.optimize(
                objective,
                n_trials=trials,
                timeout=config.timeout,
                n_jobs=1,
                show_progress_bar=False,
            )
            return
        except Exception as e:
            if not _is_transient_storage_error(e):
                raise
            if attempt == 5:
                raise
            print(f"[yellow]Worker transient storage error ({e}); retrying "
                  f"(attempt {attempt + 1}/5)...[/]")
            time.sleep(5 + attempt)


class Optimizer:
    def __init__(self, config: OptunaConfig):
        self.config = config
        self.dataset = SnapshotDataset(config.dataset)

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
        """Fan out trials across N worker processes sharing one study.

        ``n_jobs`` in optuna 4.x uses a ThreadPoolExecutor, which the GIL
        serializes for this pure-Python simulator. Instead we fork N worker
        processes; each runs `study.optimize(n_trials=..., n_jobs=1)` against
        the same SQLite study (distributed optuna). Fork + copy-on-write
        shares the ~8GB feature matrix, so RAM stays ~8GB regardless of N.
        """
        import multiprocessing as mp

        global _DATASET
        _DATASET = self.dataset

        processes = min(int(self.config.jobs) if self.config.jobs and self.config.jobs > 0 else 1,
                        os.cpu_count() or 1)
        trials_per_worker = max(1, self.config.trials // processes)

        # Ensure the study/database exists before forking, so workers only
        # need to load_if_exists (no race creating the schema).
        self.study()

        ctx = mp.get_context("fork")
        workers = [
            ctx.Process(
                target=_worker_main,
                args=(self.config, trials_per_worker, self.config.seed + i),
                name=f"optuna-{i}",
            )
            for i in range(processes)
        ]

        for w in workers:
            w.start()

        started = time.time()
        try:
            while True:
                alive = [w for w in workers if w.is_alive()]
                if not alive:
                    break
                time.sleep(10)
                done = self._count_complete()
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"  [bold cyan]{done}/{self.config.trials}[/] trials "
                    f"[green]({rate:.1f}/s, ~{(self.config.trials - done) / max(rate, 1e-9):.0f}s left)[/]",
                    flush=True,
                )
        finally:
            for w in workers:
                w.join()

        # Surface any workers that died before finishing (e.g. OOM), so a
        # silently-emptied pool is not mistaken for completion.
        dead = [w for w in workers if w.exitcode and w.exitcode != 0]
        if dead:
            raise RuntimeError(
                f"{len(dead)}/{len(workers)} optuna workers exited with "
                f"code(s): {[w.exitcode for w in dead]}. Trials completed: "
                f"{self._num_complete()}."
            )

        if self._num_complete() == 0:
            raise RuntimeError("Parallel optimization produced no complete trials.")

    def _count_complete(self) -> int:
        """Count COMPLETE trials via a direct SQL query (lightweight)."""
        if not self.config.storage.startswith("sqlite:///"):
            return self._num_complete()
        import sqlite3

        db_path = self.config.storage.replace("sqlite:///", "", 1)
        try:
            with sqlite3.connect(db_path, timeout=60) as con:
                row = con.execute(
                    "SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE'"
                ).fetchone()
            return int(row[0]) if row else 0
        except sqlite3.Error:
            return self._num_complete()

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
        result = OptimizationResult(
            score=best.value,
            trial=best.number,
            parameters=dict(best.params),
            metrics={
                "profit_factor": best.user_attrs.get("profit_factor"),
                "win_rate": best.user_attrs.get("win_rate"),
                "drawdown": best.user_attrs.get("drawdown"),
                "trades": best.user_attrs.get("trades"),
            },
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
        path = self.config.output_dir / "best_strategy.json"
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
