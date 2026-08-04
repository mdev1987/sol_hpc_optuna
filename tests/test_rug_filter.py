import asyncio

import httpx
import pytest

from rugchecks.clearank import check as clearank_check
from rugchecks.pumpcoins import check as pumpcoins_check
from rugchecks.pumpcoins import RugInfo
from rugchecks.rugcheck_xyz import check as rugcheck_xyz_check


def _client(payload: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# pumpcoins.net normalization
# ---------------------------------------------------------------------------

async def _run_pumpcoins():
    return await pumpcoins_check(
        "EBgqMxyVmiqAuhh7VCbycdrf8UHB6TYAGmzuwMAtzNY9",
        _client({
            "verdict": "FAIL",
            "checks": {
                "rugScore": 54,
                "mintRevoked": True,
                "freezeRevoked": True,
                "lpLocked": True,
                "hasPool": True,
                "top10Pct": 100.0,
                "top10Ok": False,
            },
            "flags": ["Low Amount of holders"],
        }),
    )


async def _run_rugcheck_xyz():
    return await rugcheck_xyz_check(
        "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
        _client({
            "score": 17737,
            "rugged": False,
            "mintAuthority": None,
            "freezeAuthority": None,
            "topHolders": [{"pct": 75.0}, {"pct": 20.0}],
            "totalHolders": 1000,
            "lockers": [{"address": "lock1"}],
            "markets": [{"market": "m"}],
            "risks": [{"name": "High ownership"}],
        }),
    )


async def _run_clearank():
    return await clearank_check(
        "6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN",
        _client({
            "data": {
                "goplus": {
                    "_sol_mint_renounced": True,
                    "_sol_freeze_renounced": True,
                    "_sol_is_token_2022": False,
                    "_sol_has_transfer_hook": False,
                    "_sol_has_transfer_fee": False,
                    "_sol_has_permanent_delegate": False,
                    "_sol_has_default_frozen": False,
                    "holder_count": "1000",
                    "holders": [{"percent": "0.75"}, {"percent": "0.20"}],
                    "lp_holders": [],
                },
                "solana_onchain": {},
                "dex": {"pairCount": 1, "liquidityUsd": 1000.0},
            }
        }),
    )


def test_pumpcoins_parses_hard_flags():
    info = asyncio.run(_run_pumpcoins())
    assert info.error == ""
    assert info.verdict == "FAIL"
    assert info.score == 54
    assert info.mint_revoked is True
    assert info.freeze_revoked is True
    assert info.lp_locked is True
    assert info.has_pool is True
    assert info.top10_ok is False
    assert info.top10_pct == 100.0
    assert info.risk_factors == ["Low Amount of holders"]


def test_pumpcoins_dangerous_state_flag():
    payload = {
        "verdict": "PASS",
        "checks": {"mintRevoked": False, "freezeRevoked": True, "hasPool": False},
        "flags": [],
    }
    info = asyncio.run(pumpcoins_check("A" * 44, _client(payload)))
    assert info.mint_revoked is False


def test_rugcheck_xyz_parses_and_verdicts():
    info = asyncio.run(_run_rugcheck_xyz())
    assert info.error == ""
    assert info.score == 17737
    assert info.rugged is False
    assert info.mint_revoked is True
    assert info.freeze_revoked is True
    assert info.lp_locked is True
    assert info.has_pool is True
    assert info.top10_pct == 95.0
    assert info.holders == 1000
    assert info.verdict == "FAIL"
    assert "High ownership" in info.risk_factors


def test_rugcheck_xyz_active_authority_fails():
    payload = {"score": 0, "mintAuthority": "addr", "freezeAuthority": None}
    info = asyncio.run(rugcheck_xyz_check("B" * 44, _client(payload)))
    assert info.mint_revoked is False
    assert info.verdict == "FAIL"


def test_rugcheck_xyz_rugged_flag_fails():
    payload = {"score": 0, "rugged": True, "mintAuthority": None, "freezeAuthority": None}
    info = asyncio.run(rugcheck_xyz_check("C" * 44, _client(payload)))
    assert info.rugged is True
    assert info.verdict == "FAIL"


def test_rugcheck_xyz_404_fails_open():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    info = asyncio.run(rugcheck_xyz_check("D" * 44, client))
    assert info.error == "token not found"
    assert info.verdict == "UNKNOWN"


def test_clearank_parses_facts():
    info = asyncio.run(_run_clearank())
    assert info.error == ""
    assert info.verdict == "PASS"
    assert info.mint_revoked is True
    assert info.freeze_revoked is True
    assert info.has_pool is True
    assert info.lp_locked is True
    assert info.top10_pct == 95.0
    assert info.holders == 1000
    assert "lp_lock_unverified" in info.risk_factors


def test_clearank_dangerous_extension_fails():
    payload = {
        "data": {
            "goplus": {
                "_sol_mint_renounced": True,
                "_sol_freeze_renounced": True,
                "_sol_has_transfer_hook": True,
                "_sol_has_transfer_fee": False,
                "_sol_has_permanent_delegate": False,
                "_sol_has_default_frozen": False,
                "holder_count": "0",
                "holders": [],
            },
            "solana_onchain": {},
            "dex": {},
        }
    }
    info = asyncio.run(clearank_check("E" * 44, _client(payload)))
    assert info.verdict == "FAIL"
    assert "transfer_hook" in info.risk_factors


def test_clearank_active_mint_authority_fails():
    payload = {
        "data": {
            "goplus": {
                "_sol_mint_renounced": False,
                "_sol_freeze_renounced": True,
                "holder_count": "0",
                "holders": [],
            },
            "solana_onchain": {},
            "dex": {},
        }
    }
    info = asyncio.run(clearank_check("F" * 44, _client(payload)))
    assert info.verdict == "FAIL"
    assert "mint_authority_active" in info.risk_factors


def test_invalid_mint_fails_open_without_network():
    info = asyncio.run(pumpcoins_check("not-a-valid-mint"))
    assert info.error == "invalid mint address"


# ---------------------------------------------------------------------------
# bot hard-flag blocking policy (less-strict: hard flags only)
# ---------------------------------------------------------------------------

def _hard_blocked(info: RugInfo) -> bool:
    from scripts.paper_trade_live import PaperTrader

    return PaperTrader._rug_hard_blocked(info)


def test_hard_blocked_mint_authority_active():
    assert _hard_blocked(RugInfo(mint_revoked=False, freeze_revoked=True))


def test_hard_blocked_freeze_authority_active():
    assert _hard_blocked(RugInfo(mint_revoked=True, freeze_revoked=False))


def test_hard_blocked_unlocked_lp_on_pool():
    assert _hard_blocked(RugInfo(mint_revoked=True, freeze_revoked=True,
                                 has_pool=True, lp_locked=False))


def test_hard_block_ignores_heuristics():
    # Concentrated fresh pump token: verdict FAIL, high top10, no pool -> allow.
    info = RugInfo(verdict="FAIL", score=58, mint_revoked=True, freeze_revoked=True,
                   has_pool=False, lp_locked=True, top10_ok=False, top10_pct=100.0)
    assert not _hard_blocked(info)


def test_hard_block_ignores_error():
    assert not _hard_blocked(RugInfo(error="boom"))
