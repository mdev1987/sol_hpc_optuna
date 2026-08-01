from __future__ import annotations


BUNDLES: dict[str, list[str]] = {
    "structure": [
        "liquidity",
        "liquidity_change",
        "market_cap",
        "avg_trade",
        "largest_buy",
        "largest_sell",
        "age_seconds",
    ],
    "flow": [
        "liquidity",
        "liquidity_change",
        "market_cap",
        "avg_trade",
        "largest_buy",
        "largest_sell",
        "age_seconds",
        "trade_velocity",
        "wallet_velocity",
        "buys",
        "sells",
        "volume",
        "unique_wallets",
        "sell_ratio",
    ],
    "early_momentum": [
        "liquidity",
        "liquidity_change",
        "market_cap",
        "avg_trade",
        "largest_buy",
        "largest_sell",
        "age_seconds",
        "trade_velocity",
        "wallet_velocity",
        "buys",
        "sells",
        "volume",
        "unique_wallets",
        "sell_ratio",
        "vwap",
        "price_high",
        "price_first",
        "price_change_50",
    ],
}

# `reduced_full` resolves at runtime to the current selected_features.json.
ALLOWED_BUNDLES = ("structure", "flow", "early_momentum", "reduced_full")
