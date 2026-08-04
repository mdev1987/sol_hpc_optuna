"""Live paper-trading bot on the PumpAPI WebSocket data stream.

Streams live buy/sell events from ``wss://stream.pumpapi.io``, feeds them into
the same ``FeatureEngine`` used by the replay pipeline, and runs the validated
strategy through the reference simulator one event at a time — replicating the
backtest row-walk exactly (exits before entries, per-mint state). Nothing is
sent to the network: positions are simulated and trades are logged to a JSONL
ledger.

Strategy input must be a self-contained file produced by
``scripts/export_strategy.py`` (parameters + bundle features + training
scaler), e.g. ``reports/paper_strategy_flow.json``.

Usage:
    uv run python scripts/paper_trade_live.py \
        [--strategy reports/paper_strategy_flow.json] \
        [--ledger logs/paper_ledger_flow.jsonl] \
        [--report reports/paper_trade_flow.json] \
        [--initial-balance 10.0] \
        [--summary-every 600] \
        [--prune-after 7200] \
        [--cooldown 0] \
        [--no-rug-check]

Run under tmux/systemd so a dropped SSH session does not kill it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running as a plain file: ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orjson
import websockets

from feature_engine import FeatureEngine, FeatureSnapshot
from optuna_engine import _strategy_from_params
from parser import EventParser, ReplayEvent
from reporter import TelegramNotifier
from simulator import ExitReason, Simulator, WeightedStrategy

try:
    from rugchecks.pumpcoins import RugInfo, check as pumpcoins_check
except ImportError:
    RugInfo = None
    pumpcoins_check = None

DEFAULT_URI = "wss://stream.pumpapi.io/"

# An event price outside a mint's last valid price by more than this factor is
# treated as a feed glitch (observed: `2.22e-21` vs a `~2e-9` entry price),
# not a real move. Real pump/rug moves are < 1e4x per event.
GLITCH_FACTOR = 1e6

# Trade ExitReason -> reporter.py alert keys.
_REASON_MAP = {
    "STOP_LOSS": "sl",
    "TAKE_PROFIT": "tp",
    "TRAILING_STOP": "trailing",
    "TTL": "ttl",
    "MANUAL": "dead",
    "PRICE_UNAVAILABLE": "stale",
}


class PaperTrader:
    def __init__(
        self,
        strategy: dict,
        strategy_path: Path,
        ledger_path: Path,
        report_path: Path,
        initial_balance: float,
        summary_every: float,
        prune_after: float,
        cooldown_seconds: float,
        rug_enabled: bool,
        telegram_enabled: bool,
    ):
        features = strategy["features"]
        scaler = strategy["scaler"]
        strategy_config, sim_config = _strategy_from_params(
            strategy["parameters"], features, scaler
        )
        sim_config.initial_balance = initial_balance
        if cooldown_seconds != 0.0:
            sim_config.cooldown_seconds = cooldown_seconds
        self.strategy = strategy
        self.strategy_meta = {
            "trial": strategy.get("trial"),
            "score": strategy.get("score"),
            "file": str(strategy_path),
        }
        self.sim = Simulator(sim_config, WeightedStrategy(strategy_config))
        self.engine = FeatureEngine()

        self.ledger_path = ledger_path
        self.report_path = report_path
        self.summary_every = summary_every
        self.prune_after = prune_after

        self.last_price: dict[str, float] = {}
        self.last_time: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        # Last price/event-timestamp a mint reached without being a glitch;
        # used to freeze price on feed glitches and to force-close positions
        # whose price feed has gone stale.
        self.valid_price: dict[str, float] = {}
        self.last_valid_ts: dict[str, int] = {}
        self.price_glitches = 0
        # Rug hard-flag filter: mints whose pumpcoins check shows a dangerous
        # on-chain state (mint/freeze authority live, or unlocked LP on a live
        # pool) are blocked from entry. Pending/error checks fail open.
        self.rug_enabled = rug_enabled and pumpcoins_check is not None
        self.rug_checked: set[str] = set()
        self.rug_blocked: dict[str, RugInfo] = {}
        self.rug_checked_blocked: set[str] = set()
        self.rug_blocked_mints = 0
        self.rug_verdicts: dict[str, RugInfo] = {}
        self.stale_close_seconds = max(
            300, 5 * int(getattr(sim_config, "ttl_seconds", 60))
        )
        self.trade_events = 0
        self.events_seen = 0
        self.bytes_seen = 0
        self._logged = 0
        self.start_ts = time.time()
        self.connected_at: int | None = None
        self.last_summary_ts = time.time()
        self._stop = False
        self._finalized = False

        self.notifier = TelegramNotifier() if telegram_enabled else None
        self.entry_scores: dict[str, float] = {}

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Telegram notifications
    # ---------------------------------------------------------------

    def _spawn(self, coro) -> None:
        if self.notifier is None:
            return
        try:
            asyncio.create_task(coro)
        except RuntimeError:
            pass

    def _on_entry(self, snapshot: FeatureSnapshot) -> None:
        if self.notifier is None:
            return
        f = snapshot.features
        score = float(self.sim.strategy.score(snapshot))
        self.entry_scores[snapshot.mint] = score
        self._spawn(
            self.notifier.send_buy(
                mint=snapshot.mint,
                price=f.get("price", 0.0),
                score=score,
                wallets=f.get("unique_wallets", 0),
                volume=f.get("volume", 0.0),
                buy_ratio=f.get("buy_ratio", 0.0),
                age_ms=int(f.get("age_seconds", 0)) * 1000,
                activity=f.get("trade_velocity", 0.0),
                balance=self.sim.portfolio.balance,
                rug=self.rug_verdicts.get(snapshot.mint),
            )
        )

    def _on_trade_closed(self, trade) -> None:
        if self.notifier is None:
            return
        score = self.entry_scores.get(trade.mint, 0.0)
        self._spawn(
            self.notifier.send_sell(
                mint=trade.mint,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                pnl_pct=trade.roi,
                hold_sec=trade.exit_time - trade.entry_time,
                exit_reason=_REASON_MAP.get(trade.reason.value, "dead"),
                score=score,
                balance=self.sim.portfolio.balance,
                rug=self.rug_verdicts.get(trade.mint),
            )
        )

    def _send_telegram_summary(self) -> None:
        if self.notifier is None:
            return
        s = self._summary(final=False)
        m = s["metrics"]
        self._spawn(
            self.notifier.send_summary(
                runtime_s=s["elapsed_seconds"],
                trades=m["trades"],
                win_rate=m["win_rate"] * 100,
                pnl=m["total_pnl"],
                pf=m["profit_factor"],
                balance=self.sim.portfolio.balance,
                exit_counts=m["exit_reasons"],
                avg_win=m["avg_win"],
                avg_loss=m["avg_loss"],
                reward_risk=m["reward_risk"],
                expectancy=m["expectancy"],
                price_glitches=s["stream"]["price_glitches"],
                cooldown_rejects=self.sim.cooldown_rejects,
                rug_blocked=self.rug_blocked_mints,
            )
        )

    def _send_telegram_stopped(self) -> None:
        if self.notifier is None:
            return
        s = self._summary(final=True)
        m = s["metrics"]
        self._spawn(
            self.notifier.send_stopped(
                runtime_s=s["elapsed_seconds"],
                trades=m["trades"],
                win_rate=m["win_rate"] * 100,
                pnl=m["total_pnl"],
                msgs=self.notifier._sent_count,
            )
        )

    # ---------------------------------------------------------------
    # Signal handling
    # ---------------------------------------------------------------

    def request_stop(self) -> None:
        print("\n[paper] shutdown requested; finalizing...", flush=True)
        self._stop = True

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                pass

    # ---------------------------------------------------------------
    # Event handling
    # ---------------------------------------------------------------

    def _sanitize(self, event: ReplayEvent) -> bool:
        """Return False to drop the event; otherwise guarantee ``event.price``
        holds the mint's last plausible price.

        A price glitch is any event whose price is non-positive or jumps by
        more than ``GLITCH_FACTOR`` from the mint's last valid price. On a
        glitch the last valid price is frozen into the event *before* the
        ``FeatureEngine`` sees it, so the mint's price history, high/low and
        derived features are never polluted by the bad tick.
        """
        mint = event.mint
        price = event.price
        valid = self.valid_price.get(mint)
        glitch = (
            price <= 0
            or (valid is not None and price < valid / GLITCH_FACTOR)
            or (valid is not None and price > valid * GLITCH_FACTOR)
        )
        if not glitch:
            self.valid_price[mint] = price
            return True
        self.price_glitches += 1
        if valid is None:
            return False
        event.price = valid
        return True

    def _close_stale(self) -> None:
        """Force-close any position whose price feed went quiet past the
        staleness window, at the last valid price (``PRICE_UNAVAILABLE``)."""
        now_ts = int(time.time())
        for mint in list(self.sim.portfolio.positions):
            last = self.last_valid_ts.get(mint)
            if last is not None and now_ts - last < self.stale_close_seconds:
                continue
            position = self.sim.portfolio.positions[mint]
            price = self.valid_price.get(mint, position.entry_price)
            closed_before = len(self.sim.portfolio.closed)
            self.sim.portfolio.close_position(
                mint, price, now_ts, ExitReason.PRICE_UNAVAILABLE
            )
            for trade in self.sim.portfolio.closed[closed_before:]:
                self._on_trade_closed(trade)

    # ---------------------------------------------------------------
    # Rug hard-flag filter
    # ---------------------------------------------------------------

    @staticmethod
    def _rug_hard_blocked(info: RugInfo) -> bool:
        """Only genuinely dangerous on-chain states block an entry.

        Heuristic verdicts/scores/top-10 concentration are ignored: fresh
        pump.fun tokens always launch concentrated with little or no pool.
        """
        if info.error:
            return False
        if not info.mint_revoked:
            return True
        if not info.freeze_revoked:
            return True
        if info.has_pool and not info.lp_locked:
            return True
        return False

    def _spawn_rug_check(self, mint: str) -> None:
        if mint in self.rug_checked:
            return
        self.rug_checked.add(mint)
        try:
            asyncio.create_task(self._check_rug(mint))
        except RuntimeError:
            pass

    async def _check_rug(self, mint: str) -> None:
        info = await pumpcoins_check(mint)
        if not info.error:
            self.rug_verdicts[mint] = info
        if self._rug_hard_blocked(info):
            self.rug_blocked[mint] = info

    def _on_raw(self, raw: dict) -> None:
        self.events_seen += 1
        event = EventParser.parse(raw)
        if event is None:
            return
        self.trade_events += 1
        if not self._sanitize(event):
            return
        if self.rug_enabled:
            self._spawn_rug_check(event.mint)
            if (
                event.mint in self.rug_blocked
                and not self.sim.portfolio.has_position(event.mint)
            ):
                if event.mint not in self.rug_checked_blocked:
                    self.rug_checked_blocked.add(event.mint)
                    self.rug_blocked_mints += 1
                return
        snapshot = self.engine.update(event)
        price = snapshot.features["price"]
        self.last_price[snapshot.mint] = price
        self.last_time[snapshot.mint] = snapshot.timestamp
        self.last_seen[snapshot.mint] = snapshot.timestamp
        self.last_valid_ts[snapshot.mint] = snapshot.timestamp
        was_open = self.sim.portfolio.has_position(snapshot.mint)
        closed_before = len(self.sim.portfolio.closed)
        self.sim.step(snapshot)
        if len(self.sim.portfolio.closed) > closed_before:
            for trade in self.sim.portfolio.closed[closed_before:]:
                self._on_trade_closed(trade)
        if not was_open and self.sim.portfolio.has_position(snapshot.mint):
            self._on_entry(snapshot)
        self._close_stale()

    def _log_closed(self) -> None:
        closed = self.sim.portfolio.closed
        new_trades = closed[self._logged:]
        if not new_trades:
            return
        lines = []
        for trade in new_trades:
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mint": trade.mint,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "invested": trade.invested,
                "received": trade.received,
                "pnl": trade.pnl,
                "roi": trade.roi,
                "reason": trade.reason.value,
            }
            lines.append(orjson.dumps(record).decode())
        self.ledger_path.open("a").write("\n".join(lines) + "\n")
        self._logged = len(closed)

    def _prune_stale(self) -> None:
        if self.prune_after <= 0:
            return
        cutoff = int(time.time())
        held = set(self.sim.portfolio.positions)
        stale = [
            m for m, t in self.last_seen.items()
            if m not in held and (cutoff - t) > self.prune_after
        ]
        for m in stale:
            self.engine.reset_token(m)
            self.last_seen.pop(m, None)
            self.last_price.pop(m, None)
            self.last_time.pop(m, None)
            self.valid_price.pop(m, None)
            self.last_valid_ts.pop(m, None)

    # ---------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------

    def _open_equity(self) -> float:
        fee = self.sim.config.fee_bps / 10_000
        value = 0.0
        for mint, position in self.sim.portfolio.positions.items():
            price = self.last_price.get(mint, position.entry_price)
            value += position.quantity * price * (1.0 - fee)
        return value

    def _summary(self, final: bool) -> dict:
        p = self.sim.portfolio
        trades = p.closed
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = sum(abs(t.pnl) for t in losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            999.0 if gross_profit > 0 else 0.0
        )
        avg_roi = (sum(t.roi for t in trades) / len(trades)) if trades else 0.0
        avg_hold = (
            sum(t.exit_time - t.entry_time for t in trades) / len(trades)
            if trades else 0
        )
        win_rate = (len(wins) / len(trades)) if trades else 0.0
        avg_win = (sum(t.pnl for t in wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(abs(t.pnl) for t in losses) / len(losses)) if losses else 0.0
        reward_risk = avg_win / avg_loss if avg_loss > 0 else (
            999.0 if avg_win > 0 else 0.0
        )
        expectancy = (win_rate * avg_win) - ((1.0 - win_rate) * avg_loss)
        reasons: dict[str, int] = {}
        for t in trades:
            reasons[t.reason.value] = reasons.get(t.reason.value, 0) + 1

        balance = p.balance
        equity = balance + self._open_equity()
        return {
            "strategy": self.strategy_meta,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - self.start_ts, 1),
            "final": final,
            "stream": {
                "events_seen": self.events_seen,
                "trade_events": self.trade_events,
                "bytes_seen": self.bytes_seen,
                "price_glitches": self.price_glitches,
                "connected": self.connected_at is not None,
            },
            "portfolio": {
                "initial_balance": self.sim.config.initial_balance,
                "balance": round(balance, 6),
                "equity": round(equity, 6),
                "open_positions": p.position_count(),
                "entries": len(p.closed) + p.position_count(),
                "missed_entries": self.sim.missed_entries,
                "cooldown_rejects": self.sim.cooldown_rejects,
                "rug_blocked_mints": self.rug_blocked_mints,
            },
            "metrics": {
                "trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(win_rate, 4),
                "gross_profit": round(gross_profit, 6),
                "gross_loss": round(gross_loss, 6),
                "profit_factor": round(profit_factor, 4),
                "total_pnl": round(sum(t.pnl for t in trades), 6),
                "avg_roi": round(avg_roi, 6),
                "avg_hold_seconds": avg_hold,
                "avg_win": round(avg_win, 6),
                "avg_loss": round(avg_loss, 6),
                "reward_risk": round(reward_risk, 4),
                "expectancy": round(expectancy, 6),
                "max_drawdown": round(p.stats.max_drawdown, 6),
                "exit_reasons": reasons,
            },
        }

    def write_report(self, final: bool = False) -> None:
        self.report_path.write_text(
            json.dumps(self._summary(final), indent=4)
        )

    def print_summary(self, final: bool = False) -> None:
        s = self._summary(final)
        m = s["metrics"]
        print(
            f"[paper] {'FINAL ' if final else ''}summary "
            f"elapsed={s['elapsed_seconds']:.0f}s "
            f"events={s['stream']['trade_events']} "
            f"trades={m['trades']} wins={m['wins']} losses={m['losses']} "
            f"pf={m['profit_factor']} win_rate={m['win_rate']:.2%} "
            f"dd={m['max_drawdown']:.2%} "
            f"equity={s['portfolio']['equity']:.4f} "
            f"open={s['portfolio']['open_positions']} "
            f"missed={s['portfolio']['missed_entries']} "
            f"cooldown_rejects={s['portfolio']['cooldown_rejects']} "
            f"rug_blocked={s['portfolio']['rug_blocked_mints']}",
            flush=True,
        )

    # ---------------------------------------------------------------
    # Streaming
    # ---------------------------------------------------------------

    async def _consume(self, ws) -> None:
        self.connected_at = int(time.time())
        print("[paper] connected to stream", flush=True)
        backoff = 1.0
        while not self._stop:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                self._close_stale()
                continue
            except websockets.ConnectionClosed:
                print("[paper] stream closed", flush=True)
                return
            if message is None:
                continue
            self.bytes_seen += len(message)
            try:
                raw = orjson.loads(message)
            except Exception:
                continue
            if not isinstance(raw, dict):
                continue
            self._on_raw(raw)
            backoff = 1.0
            now = time.time()
            if now - self.last_summary_ts >= self.summary_every:
                self._log_closed()
                self.write_report(final=False)
                self.print_summary(final=False)
                self._send_telegram_summary()
                self._prune_stale()
                self.last_summary_ts = now

    async def run(self) -> int:
        if self.notifier is not None:
            print("[paper] telegram notifications enabled", flush=True)
            self._spawn(
                self.notifier.send_startup(
                    f"Trial `{self.strategy_meta['trial']}` | "
                    f"{len(self.strategy['features'])} features | "
                    f"balance `{self.sim.config.initial_balance:.2f} SOL` | "
                    f"summary every `{self.summary_every:.0f}s`"
                )
            )
        uri = DEFAULT_URI
        backoff = 1.0
        while not self._stop:
            try:
                async with websockets.connect(
                    uri, ping_interval=20, ping_timeout=20, max_size=2**24
                ) as ws:
                    await self._consume(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any error
                print(f"[paper] connection error: {exc}", flush=True)
            if self._stop:
                break
            print(f"[paper] reconnecting in {backoff:.0f}s...", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

        self._log_closed()
        result = self.sim.finish(self.last_price, self.last_time)
        self._finalized = True
        self.write_report(final=True)
        self.print_summary(final=True)
        self._send_telegram_stopped()
        print(
            f"[paper] final balance {result.final_balance:.6f} "
            f"(return {result.total_return:.2%}) over {result.trades} trades",
            flush=True,
        )
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="reports/paper_strategy_flow.json",
                    help="self-contained strategy file (from export_strategy.py)")
    ap.add_argument("--ledger", default="logs/paper_ledger_flow.jsonl",
                    help="JSONL ledger of closed trades")
    ap.add_argument("--report", default="reports/paper_trade_flow.json",
                    help="summary report output")
    ap.add_argument("--initial-balance", type=float, default=10.0)
    ap.add_argument("--summary-every", type=float, default=600.0,
                    help="write a summary report every N seconds")
    ap.add_argument("--prune-after", type=float, default=7200.0,
                    help="drop per-token state after N seconds of inactivity "
                         "(0 disables)")
    ap.add_argument("--cooldown", default="0",
                    help="seconds a mint waits after a close before re-entry; "
                         "'once' = one trade per mint (0 disables)")
    ap.add_argument("--no-rug-check", action="store_true",
                    help="disable the pumpcoins hard-flag rug filter")
    ap.add_argument("--no-telegram", action="store_true",
                    help="disable Telegram alerts (uses BOT_TOKEN/CHAT_ID from "
                         ".env when not set)")
    args = ap.parse_args()

    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        sys.exit(
            f"strategy file not found: {strategy_path}\n"
            "Run scripts/export_strategy.py first to bake the training scaler "
            "into a self-contained strategy file."
        )
    strategy = json.loads(strategy_path.read_text())
    for key in ("parameters", "features", "scaler"):
        if key not in strategy:
            sys.exit(f"strategy file {strategy_path} is missing '{key}'")

    cooldown = args.cooldown.strip().lower()
    if cooldown == "once":
        cooldown_seconds = -1.0
    else:
        try:
            cooldown_seconds = float(cooldown)
        except ValueError:
            sys.exit(f"invalid --cooldown value: {args.cooldown} "
                     "(seconds or 'once')")
        if cooldown_seconds < 0:
            sys.exit("--cooldown must be >= 0 seconds or 'once'")

    trader = PaperTrader(
        strategy=strategy,
        strategy_path=strategy_path,
        ledger_path=Path(args.ledger),
        report_path=Path(args.report),
        initial_balance=args.initial_balance,
        summary_every=args.summary_every,
        prune_after=args.prune_after,
        cooldown_seconds=cooldown_seconds,
        rug_enabled=not args.no_rug_check,
        telegram_enabled=not args.no_telegram,
    )
    print(
        f"[paper] trial {strategy.get('trial')} | "
        f"{len(strategy['features'])} features | "
        f"initial_balance {args.initial_balance} | "
        f"summary_every {args.summary_every:.0f}s | "
        f"cooldown {args.cooldown} | "
        f"rug_check {'on' if trader.rug_enabled else 'off'}",
        flush=True,
    )

    async def _main() -> int:
        loop = asyncio.get_running_loop()
        trader.install_signal_handlers(loop)
        return await trader.run()

    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
