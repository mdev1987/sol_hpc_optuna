from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import optuna

from feature_engine import FeatureSnapshot
from simulator import Simulator, SimulatorConfig, StrategyConfig, WeightedStrategy


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

    def snapshots(self):
        ignore = {"mint", "timestamp", "slot"}
        feature_columns = [c for c in self.frame.columns if c not in ignore]

        for row in self.frame.iter_rows(named=True):
            yield FeatureSnapshot(
                mint=row["mint"],
                timestamp=row["timestamp"],
                slot=row["slot"],
                features={feature: row[feature] for feature in feature_columns},
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

        return StrategyConfig(
            min_price_change_5=trial.suggest_float("min_price_change_5", -0.30, 0.50),
            min_price_change_20=trial.suggest_float("min_price_change_20", -0.50, 1.00),
            min_liquidity=trial.suggest_float("min_liquidity", 1_000, 200_000),
            max_liquidity=trial.suggest_float("max_liquidity", 5_000, 5_000_000),
            min_market_cap=trial.suggest_float("min_market_cap", 5_000, 500_000),
            max_market_cap=trial.suggest_float("max_market_cap", 20_000, 10_000_000),
            min_volume=trial.suggest_float("min_volume", 100, 500_000),
            min_buy_ratio=trial.suggest_float("min_buy_ratio", 0.10, 0.95),
            min_trades=trial.suggest_int("min_trades", 1, 500),
            min_wallets=trial.suggest_int("min_wallets", 1, 300),
            min_wallet_velocity=trial.suggest_float("min_wallet_velocity", 0.0, 5.0),
            minimum_score=trial.suggest_float("minimum_score", -2.0, 2.0),
            weights=weights,
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

        score = (
            2.5 * result.profit_factor
            + 1.5 * result.win_rate
            + 1.0 * result.total_return
            + 0.5 * result.total_pnl
            - 3.0 * result.max_drawdown
        )

        trial.set_user_attr("profit_factor", result.profit_factor)
        trial.set_user_attr("win_rate", result.win_rate)
        trial.set_user_attr("drawdown", result.max_drawdown)
        trial.set_user_attr("trades", result.trades)

        return score


class Optimizer:
    def __init__(self, config: OptunaConfig):
        self.config = config
        self.dataset = SnapshotDataset(config.dataset)

    def study(self) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.config.seed, multivariate=True)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=100, n_warmup_steps=10)

        return optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=True,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
        )

    def run(self) -> OptimizationResult:
        study = self.study()
        objective = Objective(self.config, self.dataset)

        study.optimize(
            objective,
            n_trials=self.config.trials,
            timeout=self.config.timeout,
            n_jobs=self.config.jobs,
            show_progress_bar=True,
        )

        best = study.best_trial
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
