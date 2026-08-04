"""Self-contained rug-check client for trade.clearank.com.

Posts ``POST https://trade.clearank.com/api/rugcheck`` with
``{"address": mint, "chain": "solana"}`` and normalizes the aggregated
GoPlus + on-chain facts into a :class:`RugInfo`. Every failure path returns
an ``ERROR`` ``RugInfo`` instead of raising, so callers can fail open.
Results are cached for ``CACHE_TTL`` seconds (errors are never cached).

This source reports exact on-chain facts (authorities renounced, token-2022
extensions, holder count) rather than a heuristic verdict, so it is the most
trustworthy for hard-risk determination. LP lock state is NOT asserted by
GoPlus: ``lp_locked`` is left ``True`` with an ``lp_lock_unverified`` risk
factor so consumers never block on it from this source.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx

CACHE_TTL = 300
MAX_CACHE_SIZE = 5000
TIMEOUT_SECONDS = 20.0
BASE_URL = "https://trade.clearank.com/api/rugcheck"
TOP10_SAFE_PCT = 50.0

RUG_EMOJI = {"PASS": "\u2705", "WARN": "\u26a0\ufe0f", "FAIL": "\u274c"}

_cache: OrderedDict[str, tuple[RugInfo, float]] = OrderedDict()


@dataclass(slots=True)
class RugInfo:
    source: str = "clearank"
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
    payload = {"address": mint, "chain": "solana"}
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as own:
                resp = await own.post(BASE_URL, json=payload)
        else:
            resp = await client.post(BASE_URL, json=payload)
        resp.raise_for_status()
        body = resp.json()
        data = body.get("data") or {}
    except Exception as exc:  # noqa: BLE001 - fail open on any transport/parse error
        return RugInfo(error=f"{type(exc).__name__}: {exc}")

    goplus = data.get("goplus") or {}
    onchain = data.get("solana_onchain") or {}
    dex = data.get("dex") or {}

    mint_revoked = bool(goplus.get("_sol_mint_renounced", onchain.get("is_mint_renounced", False)))
    freeze_revoked = bool(goplus.get("_sol_freeze_renounced", onchain.get("is_freeze_renounced", False)))

    has_transfer_hook = bool(goplus.get("_sol_has_transfer_hook", onchain.get("has_transfer_hook", False)))
    has_permanent_delegate = bool(goplus.get("_sol_has_permanent_delegate", onchain.get("has_permanent_delegate", False)))
    has_default_frozen = bool(goplus.get("_sol_has_default_frozen", onchain.get("has_default_frozen", False)))
    has_transfer_fee = bool(goplus.get("_sol_has_transfer_fee", onchain.get("has_transfer_fee", False)))
    transfer_fee_bps = int(goplus.get("_sol_transfer_fee_bps") or onchain.get("transfer_fee_bps") or 0)
    is_token_2022 = bool(goplus.get("_sol_is_token_2022", onchain.get("is_token_2022", False)))

    risk_factors: list[str] = []
    if not mint_revoked:
        risk_factors.append("mint_authority_active")
    if not freeze_revoked:
        risk_factors.append("freeze_authority_active")
    if has_transfer_hook:
        risk_factors.append("transfer_hook")
    if has_permanent_delegate:
        risk_factors.append("permanent_delegate")
    if has_default_frozen:
        risk_factors.append("default_frozen")
    if has_transfer_fee and transfer_fee_bps > 0:
        risk_factors.append(f"transfer_fee_{transfer_fee_bps}bps")
    if is_token_2022:
        risk_factors.append("token_2022")

    holders = goplus.get("holders") or []
    top10_pct = sum(float(h.get("percent") or 0) for h in holders[:10]) * 100.0
    holder_count = int(goplus.get("holder_count") or 0)

    liquidity_usd = float(dex.get("liquidityUsd") or 0)
    has_pool = int(dex.get("pairCount") or 0) > 0 or liquidity_usd > 0
    if has_pool:
        risk_factors.append("lp_lock_unverified")

    verdict = (
        "FAIL"
        if (
            not mint_revoked
            or not freeze_revoked
            or has_transfer_hook
            or has_permanent_delegate
            or has_default_frozen
            or (has_transfer_fee and transfer_fee_bps > 0)
        )
        else "PASS"
    )

    return RugInfo(
        verdict=verdict,
        mint_revoked=mint_revoked,
        freeze_revoked=freeze_revoked,
        lp_locked=True,
        has_pool=has_pool,
        top10_ok=top10_pct < TOP10_SAFE_PCT,
        top10_pct=round(top10_pct, 4),
        holders=holder_count,
        risk_factors=risk_factors,
    )
