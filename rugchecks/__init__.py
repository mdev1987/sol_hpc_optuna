"""Rug-check clients for three independent providers.

Each module is self-contained: its own HTTP client, TTL cache and
:class:`RugInfo` dataclass. All expose an identical async entry point
``await check(mint) -> RugInfo`` so they are drop-in interchangeable.

- :mod:`rugchecks.pumpcoins` — pumpcoins.net (free, hard flags)
- :mod:`rugchecks.rugcheck_xyz` — api.rugcheck.xyz (score + rugged flag)
- :mod:`rugchecks.clearank` — trade.clearank.com (GoPlus on-chain facts)
"""

from rugchecks.pumpcoins import RUG_EMOJI, RugInfo, check as pumpcoins_check
from rugchecks.rugcheck_xyz import check as rugcheck_xyz_check
from rugchecks.clearank import check as clearank_check

__all__ = [
    "RUG_EMOJI",
    "RugInfo",
    "pumpcoins_check",
    "rugcheck_xyz_check",
    "clearank_check",
]
