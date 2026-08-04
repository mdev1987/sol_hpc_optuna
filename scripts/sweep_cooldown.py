"""Offline sweep over replay data to choose a re-entry cooldown.

Replays a parquet of raw trade events through the exact live pipeline
(price-glitch sanitize -> ``FeatureEngine.update`` -> ``Simulator.step``)
for several ``cooldown_seconds`` configs, then reports trades / PnL / PF /
win-rate / max-drawdown plus per-mint re-entry stats so a live cooldown value
can be picked without touching the strategy parameters.

The replay file must be sorted by ``timestamp`` (ascending). The default
``parquet/replay.parquet`` is downloaded in roughly chronological order with
local disorder of a few seconds; pass a globally-sorted copy for a faithful
replay (e.g. produced once via a streaming ``sort``).

Usage:
    uv run python scripts/sweep_cooldown.py \
        --parquet parquet/replay.parquet \
        --strategy reports/paper_strategy_flow.json \
        --cooldowns 0,180,300,600,once \
        --sample-fraction 1.0 \
        --prune-after 7200

Note: the replay is processed one Python event at a time (mirroring the live
bot), so a full-day file is ~85 minutes per cooldown config. Use
``--sample-fraction`` or run overnight / on the HPC box for multi-config runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running as a plain file: ensure the repo root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from feature_engine import FeatureEngine
from optuna_engine import _strategy_from_params
from parser import EventParser
from simulator import ExitReason, Simulator, WeightedStrategy

GLITCH_FACTOR = 1e6

COOLDOWN_LABELS = {0.0: "0", -1.0: "once"}


def _label(cooldown: float) -> str:
    return COOLDOWN_LABELS.get(cooldown, str(int(cooldown)))


def _rows(
    parquet_path: Path,
    batch_size: int,
    allowed_mints: set[str] | None,
):
    """Yield raw event dicts in timestamp order with bounded memory."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[
        "timestamp", "mint", "trader", "side", "amount", "price",
        "market_cap", "liquidity",
    ]):
        df = pl.from_arrow(batch)
        rows = df.iter_rows(named=True)
        for r in rows:
            if allowed_mints is not None and r["mint"] not in allowed_mints:
                continue
            yield {
                "action": r["side"],
                # parquet stores epoch seconds; EventParser divides by 1000.
                "timestamp": int(r["timestamp"]) * 1000,
                "mint": r["mint"],
                "txSigner": r["trader"],
                "amount": r["amount"],
                "price": r["price"],
                "market_cap": r["market_cap"],
                "liquidity": r["liquidity"],
            }


class _ReplayMirror:
    """Faithful offline copy of the live bot's per-event handling."""

    def __init__(self, sim_config, strategy_config, stale_close_seconds: int, prune_after: float):
        self.sim = Simulator(sim_config, WeightedStrategy(strategy_config))
        self.engine = FeatureEngine()
        self.stale_close_seconds = stale_close_seconds
        self.prune_after = prune_after
        self.valid_price: dict[str, float] = {}
        self.last_valid_ts: dict[str, int] = {}
        self.last_seen: dict[str, int] = {}
        self.last_price: dict[str, float] = {}
        self.last_time: dict[str, int] = {}
        self.price_glitches = 0

    def close_stale(self, now_ts: int) -> None:
        for mint in list(self.sim.portfolio.positions):
            last = self.last_valid_ts.get(mint)
            if last is not None and now_ts - last < self.stale_close_seconds:
                continue
            position = self.sim.portfolio.positions[mint]
            price = self.valid_price.get(mint, position.entry_price)
            self.sim.portfolio.close_position(
                mint, price, now_ts, ExitReason.PRICE_UNAVAILABLE
            )

    def sanitize(self, event) -> bool:
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

    def on_event(self, raw: dict) -> None:
        event = EventParser.parse(raw)
        if event is None:
            return
        if not self.sanitize(event):
            return
        snapshot = self.engine.update(event)
        self.last_price[event.mint] = snapshot.features["price"]
        self.last_time[event.mint] = snapshot.timestamp
        self.last_seen[event.mint] = snapshot.timestamp
        self.last_valid_ts[event.mint] = snapshot.timestamp
        self.sim.step(snapshot)

    def prune(self, now_ts: int) -> None:
        if self.prune_after <= 0:
            return
        held = set(self.sim.portfolio.positions)
        stale = [
            m for m, t in self.last_seen.items()
            if m not in held and (now_ts - t) > self.prune_after
        ]
        for m in stale:
            self.engine.reset_token(m)
            for store in (self.last_seen, self.last_price, self.last_time,
                          self.valid_price, self.last_valid_ts):
                store.pop(m, None)


def _entry_stats(closed, open_positions) -> dict:
    counts = Counter()
    for t in closed:
        counts[t.mint] += 1
    for mint in open_positions:
        counts[mint] += 1
    max_reentry = max(counts.values()) if counts else 0
    reentry_hist = Counter(counts.values())
    return {
        "unique_mints": len(counts),
        "max_reentries": max_reentry,
        "reentry_hist": dict(sorted(reentry_hist.items())),
    }


def run_cooldown(
    sim_config,
    strategy_config,
    rows,
    batch_size: int,
    prune_after: float,
) -> dict:
    stale_close_seconds = max(300, 5 * int(sim_config.ttl_seconds))
    mirror = _ReplayMirror(sim_config, strategy_config, stale_close_seconds, prune_after)

    start = time.time()
    n = 0
    now_ts = 0
    for raw in rows:
        mirror.on_event(raw)
        n += 1
        now_ts = int(raw["timestamp"]) // 1000
        if n % batch_size == 0:
            mirror.close_stale(now_ts)
            mirror.prune(now_ts)
    mirror.close_stale(now_ts)

    result = mirror.sim.finish(mirror.last_price, mirror.last_time)
    stats = result.closed_trades
    wins = [t for t in stats if t.pnl > 0]
    losses = [t for t in stats if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = sum(abs(t.pnl) for t in losses)
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "seconds": round(time.time() - start, 1),
        "events": n,
        "price_glitches": mirror.price_glitches,
        "trades": len(stats),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(stats), 4) if stats else 0.0,
        "total_pnl": round(sum(t.pnl for t in stats), 6),
        "profit_factor": round(pf, 4),
        "max_drawdown": round(result.max_drawdown, 4),
        "final_balance": round(result.final_balance, 6),
        "cooldown_rejects": mirror.sim.cooldown_rejects,
        "entry_stats": _entry_stats(stats, mirror.sim.portfolio.positions),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="parquet/replay.parquet",
                    help="raw replay parquet (sorted by timestamp)")
    ap.add_argument("--strategy", default="reports/paper_strategy_flow.json")
    ap.add_argument("--cooldowns", default="0,180,300,600,once",
                    help="comma-separated cooldown configs: seconds or 'once'")
    ap.add_argument("--initial-balance", type=float, default=10.0)
    ap.add_argument("--sample-fraction", type=float, default=1.0,
                    help="fraction of mints to keep (0-1, seeded)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prune-after", type=float, default=7200.0)
    ap.add_argument("--batch-size", type=int, default=2_000_000)
    args = ap.parse_args()

    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        sys.exit(f"strategy file not found: {strategy_path}")
    strategy = json.loads(strategy_path.read_text())
    for key in ("parameters", "features", "scaler"):
        if key not in strategy:
            sys.exit(f"strategy file {strategy_path} is missing '{key}'")

    strategy_config, base_sim = _strategy_from_params(
        strategy["parameters"], strategy["features"], strategy["scaler"]
    )
    base_sim.initial_balance = args.initial_balance
    allowed_mints = None
    if args.sample_fraction < 1.0:
        mints = (
            pl.scan_parquet(args.parquet)
            .select(pl.col("mint").unique())
            .collect(engine="streaming")["mint"].to_list()
        )
        import random

        rng = random.Random(args.seed)
        rng.shuffle(mints)
        allowed_mints = set(mints[: max(1, int(len(mints) * args.sample_fraction))])
        print(f"[sweep] sampling {len(allowed_mints)}/{len(mints)} mints "
              f"(seed {args.seed})", flush=True)

    cooldowns = []
    for c in args.cooldowns.split(","):
        c = c.strip().lower()
        cooldowns.append(-1.0 if c == "once" else float(c))

    print(f"[sweep] cooldown sweep over {args.parquet} "
          f"(trial {strategy.get('trial')})", flush=True)
    results = []
    for cooldown in cooldowns:
        sim_config = SimulatorConfigProxy(base_sim, cooldown)
        rows = _rows(Path(args.parquet), args.batch_size, allowed_mints)
        print(f"[sweep] cooldown={_label(cooldown)} ...", flush=True)
        res = run_cooldown(sim_config, strategy_config, rows, args.batch_size, args.prune_after)
        res["cooldown"] = _label(cooldown)
        results.append(res)
        print(f"[sweep]   done in {res['seconds']}s "
              f"trades={res['trades']} pnl={res['total_pnl']} "
              f"pf={res['profit_factor']}", flush=True)

    print("\n=== cooldown sweep results ===")
    hdr = (f"{'cooldown':<10}{'trades':>8}{'wins':>6}{'losses':>7}"
           f"{'win%':>8}{'pnl':>12}{'PF':>8}{'dd':>8}{'mints':>7}"
           f"{'maxRe':>7}{'cdRej':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        es = r["entry_stats"]
        print(
            f"{r['cooldown']:<10}{r['trades']:>8}{r['wins']:>6}{r['losses']:>7}"
            f"{r['win_rate']*100:>8.1f}{r['total_pnl']:>12.4f}{r['profit_factor']:>8.2f}"
            f"{r['max_drawdown']:>8.2%}{es['unique_mints']:>7}{es['max_reentries']:>7}"
            f"{r['cooldown_rejects']:>8}"
        )
    for r in results:
        print(f"\ncooldown={r['cooldown']} re-entry histogram: "
              f"{r['entry_stats']['reentry_hist']}")
    return 0


class SimulatorConfigProxy:
    """Forward baseline simulator settings, overriding only the cooldown."""

    def __init__(self, base, cooldown: float):
        self._base = base
        self.cooldown_seconds = cooldown

    def __getattr__(self, name):
        return getattr(self._base, name)


if __name__ == "__main__":
    sys.exit(main())
