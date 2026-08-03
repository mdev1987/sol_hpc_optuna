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
        [--prune-after 7200]

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

from feature_engine import FeatureEngine
from optuna_engine import _strategy_from_params
from parser import EventParser
from simulator import Simulator, WeightedStrategy

DEFAULT_URI = "wss://stream.pumpapi.io/"


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
    ):
        features = strategy["features"]
        scaler = strategy["scaler"]
        sim_config, strategy_config = _strategy_from_params(
            strategy["parameters"], features, scaler
        )
        sim_config.initial_balance = initial_balance
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
        self.trade_events = 0
        self.events_seen = 0
        self.bytes_seen = 0
        self._logged = 0
        self.start_ts = time.time()
        self.connected_at: int | None = None
        self.last_summary_ts = time.time()
        self._stop = False
        self._finalized = False

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

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

    def _on_raw(self, raw: dict) -> None:
        self.events_seen += 1
        event = EventParser.parse(raw)
        if event is None:
            return
        self.trade_events += 1
        snapshot = self.engine.update(event)
        self.last_price[snapshot.mint] = snapshot.features["price"]
        self.last_time[snapshot.mint] = snapshot.timestamp
        self.last_seen[snapshot.mint] = snapshot.timestamp
        self.sim.step(snapshot)

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
                "connected": self.connected_at is not None,
            },
            "portfolio": {
                "initial_balance": self.sim.config.initial_balance,
                "balance": round(balance, 6),
                "equity": round(equity, 6),
                "open_positions": p.position_count(),
                "entries": len(p.closed) + p.position_count(),
                "missed_entries": self.sim.missed_entries,
            },
            "metrics": {
                "trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
                "gross_profit": round(gross_profit, 6),
                "gross_loss": round(gross_loss, 6),
                "profit_factor": round(profit_factor, 4),
                "total_pnl": round(sum(t.pnl for t in trades), 6),
                "avg_roi": round(avg_roi, 6),
                "avg_hold_seconds": avg_hold,
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
            f"missed={s['portfolio']['missed_entries']}",
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
                self._prune_stale()
                self.last_summary_ts = now

    async def run(self) -> int:
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

    trader = PaperTrader(
        strategy=strategy,
        strategy_path=strategy_path,
        ledger_path=Path(args.ledger),
        report_path=Path(args.report),
        initial_balance=args.initial_balance,
        summary_every=args.summary_every,
        prune_after=args.prune_after,
    )
    print(
        f"[paper] trial {strategy.get('trial')} | "
        f"{len(strategy['features'])} features | "
        f"initial_balance {args.initial_balance} | "
        f"summary_every {args.summary_every:.0f}s",
        flush=True,
    )

    async def _main() -> int:
        loop = asyncio.get_running_loop()
        trader.install_signal_handlers(loop)
        return await trader.run()

    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
