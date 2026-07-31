from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from typing import Deque

from parser import ReplayEvent


@dataclass(slots=True)
class PriceState:
    current: float = 0.0
    first: float = 0.0
    high: float = 0.0
    low: float = 0.0
    previous: float = 0.0
    vwap_numerator: float = 0.0
    vwap_denominator: float = 0.0
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=256))

    @property
    def vwap(self) -> float:
        if self.vwap_denominator == 0:
            return 0.0
        return self.vwap_numerator / self.vwap_denominator

    def ema(self, period: int) -> float:
        if len(self.history) < 2:
            return 0.0
        alpha = 2.0 / (period + 1)
        ema_val = self.history[0]
        for p in islice(self.history, 1, None):
            ema_val = alpha * p + (1 - alpha) * ema_val
        return ema_val

    def _prices(self) -> list[float]:
        return list(self.history)

    def rsi(self, period: int = 14) -> float:
        prices = self._prices()
        if len(prices) < period + 1:
            return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(-period, 0):
            change = prices[i] - prices[i - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))

    def macd(self) -> tuple[float, float, float]:
        macd_line = self.ema(12) - self.ema(26)
        signal = self.ema(9)
        return macd_line, signal, macd_line - signal

    def roc(self, period: int = 20) -> float:
        if len(self.history) <= period:
            return 0.0
        prev = self.history[-period - 1]
        if prev == 0:
            return 0.0
        return (self.history[-1] - prev) / prev


@dataclass(slots=True)
class VolumeState:
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    largest_buy: float = 0.0
    largest_sell: float = 0.0
    trades: int = 0
    buys: int = 0
    sells: int = 0
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=256))


@dataclass(slots=True)
class WalletState:
    unique: set[str] = field(default_factory=set)
    history: Deque[str] = field(default_factory=lambda: deque(maxlen=256))
    buy_count: dict[str, int] = field(default_factory=dict)
    sell_count: dict[str, int] = field(default_factory=dict)
    first_seen: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class LiquidityState:
    current: float = 0.0
    first: float = 0.0
    highest: float = 0.0
    lowest: float = 0.0
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=256))


@dataclass(slots=True)
class MarketState:
    market_cap: float = 0.0
    age_seconds: int = 0
    first_timestamp: int = 0
    last_timestamp: int = 0


@dataclass(slots=True)
class TimeState:
    history: Deque[int] = field(default_factory=lambda: deque(maxlen=256))


@dataclass(slots=True)
class TokenState:
    mint: str
    price: PriceState = field(default_factory=PriceState)
    volume: VolumeState = field(default_factory=VolumeState)
    wallets: WalletState = field(default_factory=WalletState)
    liquidity: LiquidityState = field(default_factory=LiquidityState)
    market: MarketState = field(default_factory=MarketState)
    time: TimeState = field(default_factory=TimeState)


@dataclass(slots=True)
class FeatureSnapshot:
    mint: str
    timestamp: int
    slot: int
    features: dict[str, float]


class FeatureEngine:
    def __init__(self):
        self.tokens: dict[str, TokenState] = {}

    def token(self, mint: str) -> TokenState:
        state = self.tokens.get(mint)
        if state is None:
            state = TokenState(mint=mint)
            self.tokens[mint] = state
        return state

    @staticmethod
    def pct_change(current: float, previous: float) -> float:
        if previous <= 0:
            return 0.0
        return (current - previous) / previous

    @staticmethod
    def window_change(history: Deque[float], lookback: int) -> float:
        if len(history) <= lookback:
            return 0.0
        old = history[-lookback - 1]
        if old <= 0:
            return 0.0
        return (history[-1] - old) / old

    @staticmethod
    def velocity(values: Deque[float], timestamps: Deque[int]) -> float:
        if len(values) < 2:
            return 0.0
        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (values[-1] - values[0]) / elapsed

    def _update_price(self, state: TokenState, event: ReplayEvent) -> None:
        p = state.price
        if p.first == 0.0:
            p.first = event.price
        p.previous = p.current
        p.current = event.price
        if event.price > p.high:
            p.high = event.price
        if p.low == 0.0 or event.price < p.low:
            p.low = event.price
        p.vwap_numerator += event.price * event.amount
        p.vwap_denominator += event.amount
        p.history.append(event.price)

    def _update_volume(self, state: TokenState, event: ReplayEvent) -> None:
        v = state.volume
        v.trades += 1
        v.total_volume += event.amount
        v.history.append(event.amount)
        side = event.side.upper()
        if side == "BUY":
            v.buys += 1
            v.buy_volume += event.amount
            if event.amount > v.largest_buy:
                v.largest_buy = event.amount
        elif side == "SELL":
            v.sells += 1
            v.sell_volume += event.amount
            if event.amount > v.largest_sell:
                v.largest_sell = event.amount

    def _update_wallet(self, state: TokenState, event: ReplayEvent) -> None:
        w = state.wallets
        wallet = event.trader
        w.unique.add(wallet)
        w.history.append(wallet)
        if wallet not in w.first_seen:
            w.first_seen[wallet] = event.timestamp
        side = event.side.upper()
        if side == "BUY":
            w.buy_count[wallet] = w.buy_count.get(wallet, 0) + 1
        elif side == "SELL":
            w.sell_count[wallet] = w.sell_count.get(wallet, 0) + 1

    def _update_liquidity(self, state: TokenState, event: ReplayEvent) -> None:
        liq = state.liquidity
        if liq.first == 0.0:
            liq.first = event.liquidity
        liq.current = event.liquidity
        if event.liquidity > liq.highest:
            liq.highest = event.liquidity
        if liq.lowest == 0.0 or event.liquidity < liq.lowest:
            liq.lowest = event.liquidity
        liq.history.append(event.liquidity)

    def _update_market(self, state: TokenState, event: ReplayEvent) -> None:
        m = state.market
        m.market_cap = event.market_cap
        if m.first_timestamp == 0:
            m.first_timestamp = event.timestamp
        m.last_timestamp = event.timestamp
        m.age_seconds = m.last_timestamp - m.first_timestamp

    def _update_time(self, state: TokenState, event: ReplayEvent) -> None:
        state.time.history.append(event.timestamp)

    def _price_features(self, state: TokenState) -> dict[str, float]:
        p = state.price
        return {
            "price": p.current,
            "price_first": p.first,
            "price_high": p.high,
            "price_low": p.low,
            "price_change": self.pct_change(p.current, p.previous),
            "price_change_5": self.window_change(p.history, 5),
            "price_change_20": self.window_change(p.history, 20),
            "price_change_50": self.window_change(p.history, 50),
            "price_change_100": self.window_change(p.history, 100),
            "vwap": p.vwap,
            "ema5": p.ema(5),
            "ema20": p.ema(20),
            "ema50": p.ema(50),
            "ema100": p.ema(100),
            "rsi": p.rsi(),
            "macd_line": p.macd()[0],
            "macd_signal": p.macd()[1],
            "macd_hist": p.macd()[2],
            "roc_20": p.roc(20),
        }

    def _volume_features(self, state: TokenState) -> dict[str, float]:
        v = state.volume
        buy_ratio = 0.0
        if v.total_volume > 0:
            buy_ratio = v.buy_volume / v.total_volume
        return {
            "trades": float(v.trades),
            "buys": float(v.buys),
            "sells": float(v.sells),
            "volume": v.total_volume,
            "buy_volume": v.buy_volume,
            "sell_volume": v.sell_volume,
            "buy_ratio": buy_ratio,
            "sell_ratio": 1.0 - buy_ratio,
            "avg_trade": v.total_volume / max(v.trades, 1),
            "largest_buy": v.largest_buy,
            "largest_sell": v.largest_sell,
        }

    def _wallet_features(self, state: TokenState) -> dict[str, float]:
        w = state.wallets
        return {
            "unique_wallets": float(len(w.unique)),
            "wallet_events": float(len(w.history)),
        }

    def _liquidity_features(self, state: TokenState) -> dict[str, float]:
        liq = state.liquidity
        return {
            "liquidity": liq.current,
            "liquidity_first": liq.first,
            "liquidity_high": liq.highest,
            "liquidity_low": liq.lowest,
            "liquidity_change": self.pct_change(liq.current, liq.first),
            "liquidity_change_5": self.window_change(liq.history, 5),
            "liquidity_change_20": self.window_change(liq.history, 20),
            "liquidity_change_50": self.window_change(liq.history, 50),
        }

    def _market_features(self, state: TokenState) -> dict[str, float]:
        m = state.market
        return {
            "market_cap": m.market_cap,
            "age_seconds": float(m.age_seconds),
        }

    def _velocity_features(self, state: TokenState) -> dict[str, float]:
        v = state.volume
        return {
            "price_velocity": self.velocity(state.price.history, state.time.history),
            "volume_velocity": self.velocity(v.history, state.time.history),
            "liquidity_velocity": self.velocity(
                state.liquidity.history, state.time.history
            ),
            "trade_velocity": v.trades / max(state.market.age_seconds, 1),
            "wallet_velocity": len(state.wallets.unique)
            / max(state.market.age_seconds, 1),
        }

    def _merge_features(self, *groups: dict[str, float]) -> dict[str, float]:
        features: dict[str, float] = {}
        for group in groups:
            features.update(group)
        return features

    def update(self, event: ReplayEvent) -> FeatureSnapshot:
        state = self.token(event.mint)

        self._update_price(state, event)
        self._update_volume(state, event)
        self._update_wallet(state, event)
        self._update_liquidity(state, event)
        self._update_market(state, event)
        self._update_time(state, event)

        features = self._merge_features(
            self._price_features(state),
            self._volume_features(state),
            self._wallet_features(state),
            self._liquidity_features(state),
            self._market_features(state),
            self._velocity_features(state),
        )

        return FeatureSnapshot(
            mint=event.mint,
            timestamp=event.timestamp,
            slot=event.slot,
            features=features,
        )

    def reset_token(self, mint: str) -> None:
        self.tokens.pop(mint, None)

    def reset(self) -> None:
        self.tokens.clear()

    def token_count(self) -> int:
        return len(self.tokens)

    def tracked_tokens(self) -> list[str]:
        return sorted(self.tokens.keys())

    def has_token(self, mint: str) -> bool:
        return mint in self.tokens


def feature_names() -> list[str]:
    return [
        "price",
        "price_first",
        "price_high",
        "price_low",
        "price_change",
        "price_change_5",
        "price_change_20",
        "price_change_50",
        "price_change_100",
        "vwap",
        "ema5",
        "ema20",
        "ema50",
        "ema100",
        "rsi",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "roc_20",
        "liquidity",
        "liquidity_first",
        "liquidity_high",
        "liquidity_low",
        "liquidity_change",
        "liquidity_change_5",
        "liquidity_change_20",
        "liquidity_change_50",
        "market_cap",
        "age_seconds",
        "trades",
        "buys",
        "sells",
        "volume",
        "buy_volume",
        "sell_volume",
        "buy_ratio",
        "sell_ratio",
        "avg_trade",
        "largest_buy",
        "largest_sell",
        "unique_wallets",
        "wallet_events",
        "price_velocity",
        "volume_velocity",
        "liquidity_velocity",
        "trade_velocity",
        "wallet_velocity",
    ]


def build_features(events):
    engine = FeatureEngine()
    for event in events:
        yield engine.update(event)


def build_features_from_parquet(df, batch_size: int = 100_000, progress=None, task_id=None, output=None) -> int:
    import dataclasses
    import polars as pl

    engine = FeatureEngine()
    total_rows = len(df)
    writer = None
    written = 0

    try:
        for start in range(0, total_rows, batch_size):
            batch = df[start : start + batch_size]
            snapshots: list[FeatureSnapshot] = []
            for row in batch.iter_rows(named=True):
                event = ReplayEvent(
                    timestamp=row.get("timestamp", 0),
                    signature=row.get("signature", ""),
                    slot=row.get("slot", 0),
                    mint=row.get("mint", ""),
                    trader=row.get("trader", ""),
                    side=row.get("side", ""),
                    amount=float(row.get("amount", 0)),
                    price=float(row.get("price", 0)),
                    market_cap=float(row.get("market_cap", 0)),
                    liquidity=float(row.get("liquidity", 0)),
                    raw=row,
                )
                snapshots.append(engine.update(event))

            if output is not None and snapshots:
                import pyarrow.parquet as pq

                dicts = [dataclasses.asdict(s) for s in snapshots]
                frame = pl.from_dicts(dicts)
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        output,
                        table.schema,
                        compression="snappy",
                    )
                writer.write_table(table)
                written += len(frame)

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(batch))
    finally:
        if writer is not None:
            writer.close()

    return written
