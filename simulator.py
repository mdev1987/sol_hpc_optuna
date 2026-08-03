from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from feature_engine import FeatureSnapshot


class Side(Enum):
    LONG = "LONG"


class ExitReason(Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    TTL = "TTL"
    MANUAL = "MANUAL"


@dataclass(slots=True)
class SimulatorConfig:
    initial_balance: float = 10.0
    max_positions: int = 3
    position_size: float = 0.20
    fee_bps: float = 30.0
    slippage_bps: float = 20.0
    stop_loss: float = 0.15
    take_profit: float = 1.00
    trailing_trigger: float = 0.30
    trailing_stop: float = 0.20
    ttl_seconds: int = 300
    max_drawdown: float = 0.30


@dataclass(slots=True)
class Position:
    mint: str
    entry_time: int
    entry_price: float
    quantity: float
    invested: float
    highest_price: float
    stop_price: float
    trailing_active: bool = False


@dataclass(slots=True)
class Trade:
    mint: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    invested: float
    received: float
    pnl: float
    roi: float
    reason: ExitReason


@dataclass(slots=True)
class Statistics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_pnl: float = 0.0
    max_balance: float = 0.0
    max_drawdown: float = 0.0
    equity_curve: list[float] = field(default_factory=list)


class Portfolio:
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.balance = config.initial_balance
        self.positions: dict[str, Position] = {}
        self.closed: list[Trade] = []
        self.stats = Statistics()
        self.stats.max_balance = self.balance

    @property
    def equity(self) -> float:
        return self.balance

    def has_position(self, mint: str) -> bool:
        return mint in self.positions

    def position_count(self) -> int:
        return len(self.positions)

    def can_open(self) -> bool:
        if self.position_count() >= self.config.max_positions:
            return False
        if self.balance < self.config.position_size:
            return False
        return True

    def open_position(self, snapshot: FeatureSnapshot) -> bool:
        if self.has_position(snapshot.mint):
            return False
        if not self.can_open():
            return False

        price = snapshot.features["price"]
        fee = self.config.position_size * self.config.fee_bps / 10_000
        invested = self.config.position_size - fee
        quantity = invested / price

        position = Position(
            mint=snapshot.mint,
            entry_time=snapshot.timestamp,
            entry_price=price,
            quantity=quantity,
            invested=invested,
            highest_price=price,
            stop_price=price * (1.0 - self.config.stop_loss),
        )

        self.balance -= self.config.position_size
        self.positions[snapshot.mint] = position
        return True

    def close_position(self, mint: str, price: float, timestamp: int, reason: ExitReason) -> None:
        position = self.positions.pop(mint)
        gross = position.quantity * price
        fee = gross * self.config.fee_bps / 10_000
        received = gross - fee
        pnl = received - position.invested
        roi = pnl / position.invested

        trade = Trade(
            mint=position.mint,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            quantity=position.quantity,
            invested=position.invested,
            received=received,
            pnl=pnl,
            roi=roi,
            reason=reason,
        )

        self.closed.append(trade)
        self.balance += received
        self.stats.trades += 1
        self.stats.total_pnl += pnl

        if pnl >= 0:
            self.stats.wins += 1
            self.stats.gross_profit += pnl
        else:
            self.stats.losses += 1
            self.stats.gross_loss += abs(pnl)

        if self.balance > self.stats.max_balance:
            self.stats.max_balance = self.balance

        drawdown = (self.stats.max_balance - self.balance) / self.stats.max_balance
        if drawdown > self.stats.max_drawdown:
            self.stats.max_drawdown = drawdown

        self.stats.equity_curve.append(self.balance)


@dataclass(slots=True)
class SimulationResult:
    final_balance: float
    total_return: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    equity_curve: list[float]
    closed_trades: list[Trade]


class Strategy:
    def should_enter(self, snapshot: FeatureSnapshot) -> bool:
        raise NotImplementedError


@dataclass(slots=True)
class StrategyConfig:
    min_price_change_5: float | None = None
    min_price_change_20: float | None = None
    min_price_change_50: float | None = None
    min_liquidity: float | None = None
    max_liquidity: float | None = None
    min_market_cap: float | None = None
    max_market_cap: float | None = None
    min_volume: float | None = None
    min_buy_ratio: float | None = None
    min_trades: int | None = None
    min_wallets: int | None = None
    min_wallet_velocity: float | None = None
    min_price_velocity: float | None = None
    min_volume_velocity: float | None = None
    min_liquidity_velocity: float | None = None
    weights: dict[str, float] = field(default_factory=dict)
    minimum_score: float = 0.50
    scaler: dict[str, tuple[float, float]] | None = None


class RiskManager:
    def __init__(self, config: SimulatorConfig):
        self.config = config

    def should_exit(
        self, position: Position, price: float, timestamp: int
    ) -> ExitReason | None:

        if price > position.highest_price:
            position.highest_price = price

        gain = (price - position.entry_price) / position.entry_price

        if gain >= self.config.take_profit:
            return ExitReason.TAKE_PROFIT

        if not position.trailing_active and gain >= self.config.trailing_trigger:
            position.trailing_active = True

        if position.trailing_active:
            trailing_stop = position.highest_price * (1.0 - self.config.trailing_stop)
            if price <= trailing_stop:
                return ExitReason.TRAILING_STOP

        if price <= position.stop_price:
            return ExitReason.STOP_LOSS

        age = timestamp - position.entry_time
        if age >= self.config.ttl_seconds:
            return ExitReason.TTL

        return None


class WeightedStrategy(Strategy):
    def __init__(self, config: StrategyConfig):
        self.config = config

    @staticmethod
    def _threshold(value: float, minimum: float | None, maximum: float | None = None) -> bool:
        if minimum is not None and value < minimum:
            return False
        if maximum is not None and value > maximum:
            return False
        return True

    def score(self, snapshot: FeatureSnapshot) -> float:
        features = snapshot.features
        scaler = self.config.scaler or {}
        score = 0.0
        total_weight = 0.0
        for feature, weight in self.config.weights.items():
            value = features.get(feature, 0.0)
            if feature in scaler:
                mean, std = scaler[feature]
                value = (value - mean) / std if std > 0 else 0.0
            score += value * weight
            total_weight += abs(weight)
        if total_weight == 0:
            return 0.0
        return score / total_weight

    def should_enter(self, snapshot: FeatureSnapshot) -> bool:
        f = snapshot.features
        cfg = self.config

        checks = [
            self._threshold(f.get("price_change_5", 0), cfg.min_price_change_5),
            self._threshold(f.get("price_change_20", 0), cfg.min_price_change_20),
            self._threshold(f.get("price_change_50", 0), cfg.min_price_change_50),
            self._threshold(f.get("liquidity", 0), cfg.min_liquidity, cfg.max_liquidity),
            self._threshold(f.get("market_cap", 0), cfg.min_market_cap, cfg.max_market_cap),
            self._threshold(f.get("volume", 0), cfg.min_volume),
            self._threshold(f.get("buy_ratio", 0), cfg.min_buy_ratio),
            self._threshold(f.get("trades", 0), cfg.min_trades),
            self._threshold(f.get("unique_wallets", 0), cfg.min_wallets),
            self._threshold(f.get("wallet_velocity", 0), cfg.min_wallet_velocity),
            self._threshold(f.get("price_velocity", 0), cfg.min_price_velocity),
            self._threshold(f.get("volume_velocity", 0), cfg.min_volume_velocity),
            self._threshold(f.get("liquidity_velocity", 0), cfg.min_liquidity_velocity),
        ]

        if not all(checks):
            return False

        return self.score(snapshot) >= cfg.minimum_score


class Simulator:
    def __init__(self, config: SimulatorConfig, strategy: Strategy):
        self.config = config
        self.strategy = strategy
        self.portfolio = Portfolio(config)
        self.risk = RiskManager(config)
        # Entries the strategy wanted but the portfolio rejected (capacity or
        # balance). Exposed for live paper-trading telemetry.
        self.missed_entries = 0

    def _update_positions(self, snapshot: FeatureSnapshot) -> None:
        if not self.portfolio.positions:
            return
        for mint in list(self.portfolio.positions.keys()):
            if mint != snapshot.mint:
                continue
            position = self.portfolio.positions[mint]
            reason = self.risk.should_exit(position, snapshot.features["price"], snapshot.timestamp)
            if reason is None:
                continue
            self.portfolio.close_position(mint, snapshot.features["price"], snapshot.timestamp, reason)

    def _update_entries(self, snapshot: FeatureSnapshot) -> None:
        if self.portfolio.has_position(snapshot.mint):
            return
        if not self.strategy.should_enter(snapshot):
            return
        if not self.portfolio.open_position(snapshot):
            self.missed_entries += 1

    def _update_row(self, snapshot: FeatureSnapshot) -> None:
        self._update_positions(snapshot)
        self._update_entries(snapshot)

    @staticmethod
    def _safe(value: float, fallback: float = 0.0) -> float:
        try:
            if not math.isfinite(value):
                return fallback
        except (TypeError, ValueError):
            return fallback
        return value

    def _result(self) -> SimulationResult:
        stats = self.portfolio.stats
        trades = stats.trades
        win_rate = stats.wins / trades if trades else 0.0
        profit_factor = stats.gross_profit / stats.gross_loss if stats.gross_loss > 0 else 999.0
        balance = self._safe(self.portfolio.balance)
        total_return = (balance - self.config.initial_balance) / self.config.initial_balance

        return SimulationResult(
            final_balance=balance,
            total_return=self._safe(total_return),
            trades=trades,
            wins=stats.wins,
            losses=stats.losses,
            win_rate=self._safe(win_rate),
            gross_profit=self._safe(stats.gross_profit),
            gross_loss=self._safe(stats.gross_loss),
            profit_factor=self._safe(profit_factor, 999.0),
            total_pnl=self._safe(stats.total_pnl),
            max_drawdown=self._safe(stats.max_drawdown),
            equity_curve=list(stats.equity_curve),
            closed_trades=list(self.portfolio.closed),
        )

    def run(self, snapshots: Iterable[FeatureSnapshot]) -> SimulationResult:
        """Sequential simulation over ``FeatureSnapshot`` rows (reference path)."""
        last_price: dict[str, float] = {}
        last_time: dict[str, int] = {}
        for snapshot in snapshots:
            last_price[snapshot.mint] = snapshot.features["price"]
            last_time[snapshot.mint] = snapshot.timestamp
            self._update_row(snapshot)

        return self._close_all(last_price, last_time)

    def run_indexed(
        self,
        indices,
        mints: list[str],
        timestamps,
        prices,
        eligible,
        make_snapshot,
    ) -> SimulationResult:
        """Array-backed simulation with a precomputed entry-eligibility mask.

        The exit logic is identical to ``run`` (sequential, per open position).
        The only difference: ``should_enter`` is evaluated only on rows where
        ``eligible`` is True, so the expensive per-snapshot feature score is
        skipped for the ~99.9% of rows that are rejected anyway.

        ``indices`` are the row positions to replay (e.g. the train rows).
        ``make_snapshot(i)`` builds a ``FeatureSnapshot`` for the (few) eligible
        rows; exits read price/timestamp directly from the arrays.
        """
        last_price: dict[str, float] = {}
        last_time: dict[str, int] = {}
        for i in indices:
            mint = mints[i]
            price = prices[i]
            timestamp = int(timestamps[i])
            last_price[mint] = price
            last_time[mint] = timestamp
            if self.portfolio.positions:
                for m in list(self.portfolio.positions.keys()):
                    if m != mint:
                        continue
                    position = self.portfolio.positions[m]
                    reason = self.risk.should_exit(position, price, timestamp)
                    if reason is None:
                        continue
                    self.portfolio.close_position(mint, price, timestamp, reason)
            if eligible[i]:
                self._update_entries(make_snapshot(i))

        return self._close_all(last_price, last_time)

    def step(self, snapshot: FeatureSnapshot) -> None:
        """Advance the simulation by one live/replay snapshot.

        Exits are evaluated before entries, matching ``run`` exactly, so the
        live paper-trading bot replicates the backtest row walk one event at a
        time instead of buffering the whole stream.
        """
        self._update_row(snapshot)

    def finish(self, last_price: dict, last_time: dict) -> SimulationResult:
        """Close any still-open positions at the last seen price and return
        the final result. Used on shutdown by live paper-trading."""
        return self._close_all(last_price, last_time)

    def _close_all(self, last_price: dict, last_time: dict) -> SimulationResult:
        for mint in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[mint]
            price = last_price.get(mint, position.entry_price)
            timestamp = last_time.get(mint, position.entry_time)
            self.portfolio.close_position(mint, price, timestamp, ExitReason.MANUAL)

        return self._result()


class PerformanceMetrics:
    @staticmethod
    def average_win(trades: list[Trade]) -> float:
        wins = [t.pnl for t in trades if t.pnl > 0]
        if not wins:
            return 0.0
        return sum(wins) / len(wins)

    @staticmethod
    def average_loss(trades: list[Trade]) -> float:
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        if not losses:
            return 0.0
        return sum(losses) / len(losses)

    @staticmethod
    def expectancy(trades: list[Trade]) -> float:
        if not trades:
            return 0.0
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        win_rate = len(wins) / len(trades)
        loss_rate = len(losses) / len(trades)
        avg_win = PerformanceMetrics.average_win(trades)
        avg_loss = PerformanceMetrics.average_loss(trades)
        return win_rate * avg_win - loss_rate * avg_loss

    @staticmethod
    def average_hold_time(trades: list[Trade]) -> float:
        if not trades:
            return 0.0
        return sum(t.exit_time - t.entry_time for t in trades) / len(trades)
