"""Self-contained rug-check client for api.rugcheck.xyz.

Fetches ``GET https://api.rugcheck.xyz/v1/tokens/{mint}/report`` and
normalizes the response into a :class:`RugInfo`. Every failure path returns an
``ERROR`` ``RugInfo`` instead of raising, so callers can fail open. Results
are cached for ``CACHE_TTL`` seconds (errors are never cached).

The vendor's numeric ``score`` is a heuristic that also flags legitimate
concentrated tokens (e.g. large memecoins); prefer the hard flags
(``rugged`` / ``mint_revoked`` / ``freeze_revoked``) when blocking entries.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx

CACHE_TTL = 300
MAX_CACHE_SIZE = 5000
TIMEOUT_SECONDS = 15.0
BASE_URL = "https://api.rugcheck.xyz/v1/tokens"

# rugcheck.xyz score bands: >= 15000 danger, >= 5000 elevated risk.
DANGER_SCORE = 15000
ELEVATED_SCORE = 5000
TOP10_SAFE_PCT = 50.0

RUG_EMOJI = {"PASS": "\u2705", "WARN": "\u26a0\ufe0f", "FAIL": "\u274c"}

_cache: OrderedDict[str, tuple[RugInfo, float]] = OrderedDict()


@dataclass(slots=True)
class RugInfo:
    source: str = "rugcheck_xyz"
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


def _verdict_from_score(score: float) -> str:
    if score >= DANGER_SCORE:
        return "FAIL"
    if score >= ELEVATED_SCORE:
        return "WARN"
    return "PASS"


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
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as own:
                resp = await own.get(f"{BASE_URL}/{mint}/report")
        else:
            resp = await client.get(f"{BASE_URL}/{mint}/report")
        if resp.status_code == 404:
            return RugInfo(verdict="UNKNOWN", error="token not found")
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 - fail open on any transport/parse error
        return RugInfo(error=f"{type(exc).__name__}: {exc}")

    score = float(data.get("score") or 0)
    rugged = bool(data.get("rugged"))
    mint_revoked = data.get("mintAuthority") is None
    freeze_revoked = data.get("freezeAuthority") is None

    lockers = data.get("lockers") or []
    markets = data.get("markets") or []
    has_pool = len(markets) > 0
    lp_locked = len(lockers) > 0

    holders = data.get("topHolders") or []
    top10_pct = sum(
        float(h.get("pct") or 0) for h in holders[:10]
    )
    total_holders = int(data.get("totalHolders") or 0)

    risks = [
        r.get("name", str(r))
        for r in (data.get("risks") or [])
    ]

    verdict = "FAIL" if (rugged or not mint_revoked or not freeze_revoked) else _verdict_from_score(score)

    return RugInfo(
        verdict=verdict,
        score=score,
        rugged=rugged,
        mint_revoked=mint_revoked,
        freeze_revoked=freeze_revoked,
        lp_locked=lp_locked,
        has_pool=has_pool,
        top10_ok=top10_pct < TOP10_SAFE_PCT,
        top10_pct=round(top10_pct, 4),
        holders=total_holders,
        risk_factors=risks,
    )
