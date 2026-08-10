"""
fee_model.py — Coinbase Derivatives futures fee calculator
All fees in decimal (not percent) unless noted.

NFA fee: $0.30 round-trip per contract (fixed, all volume tiers)
Taker/maker fees: percentage of notional, tiered by 30-day volume
"""

NFA_RT_DOLLARS = 0.30  # per contract, round-trip

# Taker per side, indexed by tier (1 = lowest volume, 10 = highest)
TAKER_PER_SIDE = {
    1:  0.00100,
    2:  0.00090,
    3:  0.00080,
    4:  0.00070,
    5:  0.00060,
    6:  0.00050,
    7:  0.00040,
    8:  0.00030,
    9:  0.00020,
    10: 0.00010,
}

# Maker is always 0.5bps better than taker at each tier
MAKER_PER_SIDE = {t: v - 0.00005 for t, v in TAKER_PER_SIDE.items()}

# ── AVAX WARNING: NFA-dominated fee — monitor price ───────────────────────────
# NFA_pct = $0.30 / (10 * price). As AVAX price falls, NFA% rises fast.
# At ~$25 : effective RT T1 ≈ 0.32% (acceptable)
# At ~$15 : effective RT T1 ≈ 0.50% — reassess inclusion at this level
# At ~$10 : effective RT T1 ≈ 0.60% — strong case for exclusion
# Current price ~$9.68 (2026-03-25) — already in warning zone (~0.51% RT T1)
# ─────────────────────────────────────────────────────────────────────────────

# Coinbase Derivatives contract sizes (underlying units per contract)
CONTRACT_SIZE = {
    'BTC':  0.01,
    'ETH':  0.1,
    'SOL':  5,
    'XRP':  500,
    'DOGE': 5000,
    'ADA':  1000,
    'AVAX': 10,
    'LINK': 50,
    'LTC':  5,
    'BCH':  1,
    'XLM':  5000,
    'SUI':  500,
    'DOT':  100,
    'HBAR': 5000,
}

# 30-day volume thresholds for each tier (USD) — for reference only
TIER_THRESHOLDS_USD = {
    1:  0,
    2:  1_000,
    3:  10_000,
    4:  50_000,
    5:  500_000,
    6:  1_000_000,
    7:  15_000_000,
    8:  50_000_000,
    9:  100_000_000,
    10: 250_000_000,
}


def contract_notional(asset: str, price: float) -> float:
    """USD notional value of one contract at given price."""
    return CONTRACT_SIZE[asset] * price


def nfa_pct(asset: str, price: float) -> float:
    """
    NFA fee as a fraction of notional.
    = $0.30 / (contract_size * price)
    This shifts as price moves — must be computed at each bar.
    """
    notional = contract_notional(asset, price)
    if notional <= 0:
        raise ValueError(f"Zero or negative notional for {asset} at price {price}")
    return NFA_RT_DOLLARS / notional


def total_rt_fee(asset: str, price: float, tier: int = 1,
                 maker: bool = False) -> float:
    """
    Total round-trip fee as a fraction of notional.
    = (taker_or_maker_per_side * 2) + nfa_pct(asset, price)

    Args:
        asset:  Asset symbol (must be in CONTRACT_SIZE)
        price:  Current bar close price in USD
        tier:   Fee tier 1–10 (1 = lowest volume / highest fees)
        maker:  If True, use maker rate instead of taker rate
    Returns:
        Decimal fraction (e.g., 0.00242 = 0.242%)
    """
    per_side = MAKER_PER_SIDE[tier] if maker else TAKER_PER_SIDE[tier]
    return (per_side * 2) + nfa_pct(asset, price)


def min_account_for_asset(asset: str, price: float,
                           d: float = 0.10,
                           leverage_long: float = 4.3,
                           leverage_short: float = 3.5,
                           side: str = 'long') -> float:
    """
    Minimum account equity to trade at least 1 contract.

    Args:
        asset:           Asset symbol
        price:           Current price in USD
        d:               Fraction of equity allocated to this position (default 10%)
        leverage_long:   Max leverage available for longs
        leverage_short:  Max leverage available for shorts
        side:            'long' or 'short'
    Returns:
        Minimum USD equity needed
    """
    notional = contract_notional(asset, price)
    lev = leverage_long if side == 'long' else leverage_short
    margin = notional / lev
    return margin / d


def n_contracts(asset: str, price: float, account_equity: float,
                d: float = 0.10,
                leverage_long: float = 4.3,
                leverage_short: float = 3.5,
                side: str = 'long') -> int:
    """
    Number of contracts tradeable with current equity allocation.

    Args:
        asset:           Asset symbol
        price:           Current price in USD
        account_equity:  Total portfolio equity in USD
        d:               Fraction of equity allocated to this position (default 10%)
        leverage_long:   Max leverage for longs
        leverage_short:  Max leverage for shorts
        side:            'long' or 'short'
    Returns:
        Integer number of contracts (floored, minimum 0)
    """
    notional = contract_notional(asset, price)
    lev = leverage_long if side == 'long' else leverage_short
    margin_per_contract = notional / lev
    available_margin = account_equity * d
    return int(available_margin / margin_per_contract)


# ── Sanity check table ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pandas as pd

    # Reference prices (approximate at time of writing — for display only)
    REF_PRICES = {
        'BTC':  85000,
        'ETH':  2000,
        'SOL':  130,
        'XRP':  0.55,
        'DOGE': 0.17,
        'ADA':  0.70,
        'AVAX': 25,
        'LINK': 13,
        'LTC':  90,
        'BCH':  350,
        'XLM':  0.28,
        'SUI':  3.0,
        'DOT':  6.5,
        'HBAR': 0.22,
    }

    rows = []
    for asset, px in REF_PRICES.items():
        fee_t1 = total_rt_fee(asset, px, tier=1)
        fee_t4 = total_rt_fee(asset, px, tier=4)
        nfa    = nfa_pct(asset, px)
        notl   = contract_notional(asset, px)
        min_eq = min_account_for_asset(asset, px)
        rows.append({
            'Asset': asset,
            'Price': f"${px:,.2f}",
            'Notional/contract': f"${notl:,.2f}",
            'NFA%': f"{nfa*100:.4f}%",
            'RT fee T1': f"{fee_t1*100:.4f}%",
            'RT fee T4': f"{fee_t4*100:.4f}%",
            'Min equity (10%)': f"${min_eq:,.0f}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print("NFA_RT_DOLLARS =", NFA_RT_DOLLARS)
    print("Taker T1 (per side) =", TAKER_PER_SIDE[1])
    print("Formula: RT fee = (taker_per_side * 2) + NFA_RT / contract_notional")
