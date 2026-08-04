"""Self-contained rug-check client for pumpcoins.net.

Fetches ``GET https://pumpcoins.net/api/check?mint={mint}`` and normalizes the
response into a :class:`RugInfo`. Every failure path returns an ``ERROR``
``RugInfo`` instead of raising, so callers can fail open. Results are cached
for ``CACHE_TTL`` seconds (errors are never cached).

The vendor's ``verdict`` is a heuristic over top-10 concentration and
liquidity and is unreliable for fresh pump.fun tokens; prefer the on-chain
hard flags (``mint_revoked`` / ``freeze_revoked`` / ``lp_locked``) when
blocking entries.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx

CACHE_TTL = 300
MAX_CACHE_SIZE = 5000
TIMEOUT_SECONDS = 10.0
BASE_URL = "https://pumpcoins.net/api/check"

RUG_EMOJI = {"PASS": "\u2705", "WARN": "\u26a0\ufe0f", "FAIL": "\u274c"}

_cache: OrderedDict[str, tuple[RugInfo, float]] = OrderedDict()


@dataclass(slots=True)
class RugInfo:
    source: str = "pumpcoins"
    verdict: str = ""
    score: float = 0.0
    rugged: bool = False
    mint_revoked: bool = False
    freeze_revoked: bool = False
    lp_locked: bool = False
    has_pool: bool = False
    top10_ok: bool = True
    top10_pct: float = 0.0
    holders: int = 0
    risk_factors: list[str] = field(default_factory=list)
    error: str = ""


def _invalid_mint(mint: str) -> RugInfo | None:
    if not mint or not isinstance(mint, str) or len(mint) != 44:
        return RugInfo(error="invalid mint address")
    return None


async def check(mint: str, client: httpx.AsyncClient | None = None) -> RugInfo:
    invalid = _invalid_mint(mint)
    if invalid is not None:
        return invalid

    cached = _cache.get(mint)
    if cached is not None and time.time() - cached[1] < CACHE_TTL:
        return cached[0]

    info = await _fetch(mint, client)
    if not info.error:
        _cache[mint] = (info, time.time())
        _cache.move_to_end(mint)
        if len(_cache) > MAX_CACHE_SIZE:
            _cache.popitem(last=False)
    return info


async def _fetch(mint: str, client: httpx.AsyncClient | None) -> RugInfo:
    headers = {
        "Accept": "application/json",
        "Referer": f"https://pumpcoins.net/rug-check?mint={mint}",
    }
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as own:
                resp = await own.get(f"{BASE_URL}?mint={mint}", headers=headers)
        else:
            resp = await client.get(f"{BASE_URL}?mint={mint}", headers=headers)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - fail open on any transport/parse error
        return RugInfo(error=f"{type(exc).__name__}: {exc}")

    checks = data.get("checks") or {}
    verdict = (data.get("verdict") or "UNKNOWN").upper()
    score = float(checks.get("rugScore", 0) or 0)
    top10_pct = float(checks.get("top10Pct", 0) or 0)
    flags = data.get("flags") or checks.get("flags") or []

    return RugInfo(
        verdict=verdict,
        score=score,
        mint_revoked=bool(checks.get("mintRevoked", False)),
        freeze_revoked=bool(checks.get("freezeRevoked", False)),
        lp_locked=bool(checks.get("lpLocked", False)),
        has_pool=bool(checks.get("hasPool", False)),
        top10_ok=bool(checks.get("top10Ok", True)),
        top10_pct=top10_pct,
        holders=int(checks.get("holderCount", 0) or 0),
        risk_factors=list(flags),
    )
