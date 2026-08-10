"""
Cross-Sectional Momentum Backtest — v2
Universe:  BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK, LTC, BCH, XLM, SUI, DOT, HBAR
Signal:    24h return rank → long top 2, vol-scaled weights (25% each, 50% max)
Rebal:     Every 7 days (168 bars)
Regime:    BTC 168h MA filter with ±0.5% hysteresis
Costs:     Dynamic per-asset fee: (taker_per_side * 2) + NFA_pct(asset, price)
           NFA_pct = $0.30 / (contract_size * price)
           Computed at each rebalance bar — NFA% shifts as prices move
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import fee_model as fm

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / "cappuccino" / "data"
OUT_DIR    = Path.home() / "cappuccino" / "momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS        = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA",
                 "AVAX", "LINK", "LTC", "BCH", "XLM", "SUI", "DOT", "HBAR"]
LOOKBACK_BARS = 24       # 24h momentum signal
TOP_N         = 2        # long top-2
MAX_WEIGHT    = 0.25     # 25% per asset
REBAL_BARS    = 168      # weekly (168h)
REGIME_MA     = 168      # BTC 168h (7d) MA
REGIME_BUFFER = 0.005    # ±0.5% hysteresis band
SHARPE_MIN    = 0.4
N_FOLDS       = 3
HOURS_PER_YR  = 8760

# Capital sizing assumptions (used only for feasibility check)
ACCOUNT_EQUITY = 500.0   # USD — used to estimate contract count
POSITION_ALLOC = 0.25    # fraction of equity per position (25%)
LEVERAGE_LONG  = 4.3
LEVERAGE_SHORT = 3.5


# ── Data ──────────────────────────────────────────────────────────────────────
def load_prices() -> pd.DataFrame:
    frames = {}
    for asset in ASSETS:
        path = DATA_DIR / f"{asset}_1h.parquet"
        if not path.exists():
            print(f"  [WARN] No parquet for {asset} — skipping")
            continue
        df = pd.read_parquet(path)
        s  = df.set_index("datetime")["close"]
        s.index = pd.to_datetime(s.index, utc=True)
        frames[asset] = s
    prices = pd.DataFrame(frames).sort_index().dropna()
    return prices


# ── Fee computation ───────────────────────────────────────────────────────────
def compute_bar_fee(asset: str, price: float, tier: int, maker: bool) -> float:
    """
    Dynamic RT fee for asset at this price bar.
    Falls back to taker T1 if asset not in CONTRACT_SIZE (should never happen).
    """
    if asset not in fm.CONTRACT_SIZE:
        return fm.TAKER_PER_SIDE[1] * 2  # fallback: taker T1 no NFA
    return fm.total_rt_fee(asset, price, tier=tier, maker=maker)


# ── Feasibility check ─────────────────────────────────────────────────────────
def feasibility_check(prices: pd.DataFrame, rebal_idx: int,
                      equity: float = ACCOUNT_EQUITY,
                      d: float = POSITION_ALLOC,
                      side: str = 'long',
                      verbose: bool = True) -> dict[str, bool]:
    """
    At a given rebalance bar index, check which assets are tradeable
    given current equity and capital allocation fraction d.

    Returns dict: {asset: True/False (tradeable)}
    """
    bar_prices = prices.iloc[rebal_idx]
    tradeable  = {}
    for asset in prices.columns:
        if asset not in fm.CONTRACT_SIZE:
            tradeable[asset] = False
            continue
        px      = bar_prices[asset]
        min_eq  = fm.min_account_for_asset(asset, px, d=d,
                                            leverage_long=LEVERAGE_LONG,
                                            leverage_short=LEVERAGE_SHORT,
                                            side=side)
        n_cont  = fm.n_contracts(asset, px, equity, d=d,
                                  leverage_long=LEVERAGE_LONG,
                                  leverage_short=LEVERAGE_SHORT,
                                  side=side)
        can_trade = n_cont >= 1
        tradeable[asset] = can_trade
        if verbose:
            status = f"TRADE ({n_cont} contracts)" if can_trade else "SKIP"
            print(f"  [FEASIBILITY] {asset:<5} {side:<5}  "
                  f"min_equity=${min_eq:>8,.0f}  "
                  f"current=${equity:>8,.0f}  "
                  f"{status}")
    return tradeable


# ── Core backtest ─────────────────────────────────────────────────────────────
def backtest(
    prices: pd.DataFrame,
    use_regime: bool = True,
    vol_scale: bool = True,
    fee_tier: int = 1,
    use_maker: bool = False,
    equity: float = ACCOUNT_EQUITY,
    verbose_feasibility: bool = False,
    lookback: int = LOOKBACK_BARS,
    dispersion_threshold: float = 0.0,
    corr_threshold: float = 0.0,
    use_shorts: bool = False,
    ret7d_gate: bool = False,
    vol_threshold: float = 0.0,
) -> dict:
    """
    No lookahead:
      - Signal computed at bar close t using data[t-24 : t]
      - Regime MA computed at bar t using data[t-REGIME_MA : t]
      - Weight determined at t → shifted +1 → realised on bar t+1

    Fee model:
      - Per-asset, per-bar: total_rt_fee(asset, price, tier, maker)
      - Applied proportional to weight change at each rebalance
      - Assets where n_contracts==0 are skipped (weight forced to 0)

    vol_scale: weight top-N assets by inverse 24h vol (equal-risk allocation).
    use_regime: apply BTC MA regime filter with ±REGIME_BUFFER hysteresis.
    """
    asset_cols = list(prices.columns)
    asset_rets = prices.pct_change()

    # momentum signal and ranks (lookback is configurable)
    signal = prices.pct_change(lookback)
    ranks  = signal.rank(axis=1, ascending=False, method="first")

    # Vol-scaled weights: top-N assets, weighted by inverse 24h realised vol
    vol_24h = asset_rets.rolling(24).std()
    raw_w   = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    n_assets = len(asset_cols)
    ret_7d   = prices.pct_change(168) if ret7d_gate else None

    # 7d gate stats (only meaningful at rebalance bars)
    ret7d_n_rebal   = 0
    ret7d_n_flat    = 0   # bars where 0 eligible → go flat
    ret7d_n_partial = 0   # bars where only 1 eligible → 1 position

    for i in range(len(signal)):
        # ── 7d return gate (rebalance bars only) ─────────────────────────────
        if ret7d_gate and i % REBAL_BARS == 0 and ret_7d is not None:
            ret7d_n_rebal += 1
            eligible = ret_7d.iloc[i].notna() & (ret_7d.iloc[i] > 0)
            n_eligible = int(eligible.sum())
            if n_eligible == 0:
                raw_w.iloc[i] = 0.0
                ret7d_n_flat += 1
                continue
            # Re-rank composite signal within eligible assets only
            elig_signal = signal.iloc[i][eligible]
            elig_ranks  = elig_signal.rank(ascending=False, method="first")
            n_take  = min(TOP_N, n_eligible)
            if n_eligible == 1:
                ret7d_n_partial += 1
            top_mask = pd.Series(False, index=signal.columns)
            in_top   = elig_ranks <= n_take
            top_mask[in_top.index[in_top]] = True
        else:
            top_mask = ranks.iloc[i] <= TOP_N
            n_take   = TOP_N

        if not top_mask.any():
            continue
        if vol_scale:
            inv_v = 1.0 / vol_24h.iloc[i].where(top_mask).replace(0, np.nan).dropna()
            if inv_v.empty:
                raw_w.iloc[i][top_mask] = MAX_WEIGHT
            else:
                scaled = inv_v / inv_v.sum() * (MAX_WEIGHT * n_take)
                for asset, w in scaled.items():
                    raw_w.at[raw_w.index[i], asset] = w
        else:
            raw_w.iloc[i][top_mask] = MAX_WEIGHT

        # Short bottom-N at -MAX_WEIGHT each (equal-weighted, no vol scaling)
        if use_shorts and n_assets > TOP_N * 2:
            bottom_mask = ranks.iloc[i] > (n_assets - TOP_N)
            raw_w.iloc[i][bottom_mask] = -MAX_WEIGHT

    ret7d_flat_pct    = (ret7d_n_flat    / ret7d_n_rebal * 100 if ret7d_n_rebal > 0 else 0.0)
    ret7d_partial_pct = (ret7d_n_partial / ret7d_n_rebal * 100 if ret7d_n_rebal > 0 else 0.0)

    # ── Cross-sectional dispersion filter ────────────────────────────────────
    # At each rebalance bar, if xs_std(ret_24h) < threshold → go flat.
    # Always uses 24h returns regardless of lookback (dispersion is a market
    # regime property, not a signal property).
    disp_n_rebal    = 0
    disp_n_filtered = 0
    disp_log: list[str] = []

    if dispersion_threshold > 0.0:
        ret_24h_disp = prices.pct_change(24)
        for i in range(0, len(raw_w), REBAL_BARS):
            row = ret_24h_disp.iloc[i].dropna()
            if row.empty:
                continue
            disp_n_rebal += 1
            xs_std = float(row.std())
            if xs_std < dispersion_threshold:
                raw_w.iloc[i] = 0.0
                disp_n_filtered += 1
                disp_log.append(
                    f"[DISPERSION FILTER] bar={i}  "
                    f"xs_std={xs_std:.4f} < {dispersion_threshold:.4f} — going flat"
                )

    disp_filtered_pct = (disp_n_filtered / disp_n_rebal * 100
                         if disp_n_rebal > 0 else 0.0)

    # ── Correlation regime filter ─────────────────────────────────────────────
    # At each rebalance bar, compute rolling 168h pairwise return correlation.
    # If median pairwise correlation > corr_threshold → go flat.
    # High correlation = cross-sectional ranking is noise.
    corr_n_rebal    = 0
    corr_n_filtered = 0
    corr_log: list[str] = []

    if corr_threshold > 0.0:
        rets_1h = prices.pct_change()
        for i in range(0, len(raw_w), REBAL_BARS):
            if i < 2:
                continue
            window = rets_1h.iloc[max(0, i - 168):i].dropna()
            if len(window) < 24:
                continue
            corr_matrix = window.corr()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            median_corr = float(upper.stack().median())

            # Vol gate: avg annualised 168h realised vol across assets
            if vol_threshold > 0.0:
                avg_vol = float((window.std() * np.sqrt(8760)).mean())
                vol_gate_met = avg_vol > vol_threshold
            else:
                vol_gate_met = True   # no vol gate → treat as always met

            corr_n_rebal += 1
            if median_corr > corr_threshold and vol_gate_met:
                raw_w.iloc[i] = 0.0
                corr_n_filtered += 1
                vol_str = f"  avg_vol={avg_vol:.3f} > {vol_threshold:.3f}" if vol_threshold > 0.0 else ""
                corr_log.append(
                    f"[CORR FILTER] bar={i}  "
                    f"median_corr={median_corr:.3f} > {corr_threshold:.3f}{vol_str} — going flat"
                )

    corr_filtered_pct = (corr_n_filtered / corr_n_rebal * 100
                         if corr_n_rebal > 0 else 0.0)

    # Weekly rebalancing: carry forward between rebal bars
    rebal_w = raw_w.copy()
    for i in range(1, len(rebal_w)):
        if i % REBAL_BARS != 0:
            rebal_w.iloc[i] = rebal_w.iloc[i - 1]

    # BTC regime filter with ±REGIME_BUFFER hysteresis
    btc_ma = prices["BTC"].rolling(REGIME_MA).mean()
    if use_regime:
        in_regime_state = True
        in_regime = pd.Series(True, index=prices.index)
        for i in range(len(prices)):
            ma_val = btc_ma.iloc[i]
            if pd.isna(ma_val):
                in_regime.iloc[i] = True
                continue
            btc_px = prices["BTC"].iloc[i]
            if in_regime_state:
                if btc_px < ma_val * (1 - REGIME_BUFFER):
                    in_regime_state = False
            else:
                if btc_px > ma_val * (1 + REGIME_BUFFER):
                    in_regime_state = True
            in_regime.iloc[i] = in_regime_state

        for i in range(len(rebal_w)):
            if not in_regime.iloc[i]:
                rebal_w.iloc[i] = 0.0
    else:
        in_regime = pd.Series(True, index=prices.index)

    # ── Dynamic fee computation ───────────────────────────────────────────────
    # Shift weights first, then diff — weight change lands at rebal_bar+1
    # We detect actual cost events by finding bars where ANY weight changed,
    # rather than by index modulo (which misses the +1 offset from .shift(1)).
    #
    # Two-pass approach:
    #   Pass 1: compute initial weights_shifted and weight_chg to find rebal bars
    #   Pass 2: apply feasibility filter, update rebal_w, recompute final weights
    weights_shifted_pre = rebal_w.shift(1)
    weight_chg_pre      = weights_shifted_pre.diff().abs()

    # Bars where any weight actually changed — these are our cost events
    rebal_cost_bars = list(weight_chg_pre.index[weight_chg_pre.sum(axis=1) > 0])

    skipped_assets: dict[str, list[str]] = {}  # date_str -> [skipped assets]
    avax_fees_at_rebal: list[float] = []        # effective AVAX RT fee at each rebal

    cost_vals          = pd.Series(0.0, index=prices.index)
    per_asset_cost_df  = pd.DataFrame(0.0, index=prices.index, columns=asset_cols)

    for bar_ts in rebal_cost_bars:
        i = prices.index.get_loc(bar_ts)
        bar_prices_row = prices.loc[bar_ts]
        bar_cost = 0.0

        for asset in asset_cols:
            wc = weight_chg_pre.loc[bar_ts, asset]
            if wc == 0.0:
                continue

            px = bar_prices_row[asset]
            if px <= 0 or pd.isna(px):
                continue

            # Capital feasibility: skip asset if can't trade even 1 contract
            if asset in fm.CONTRACT_SIZE:
                trade_side = ('short'
                              if weights_shifted_pre.loc[bar_ts, asset] < 0
                              else 'long')
                n_cont = fm.n_contracts(
                    asset, px, equity,
                    d=POSITION_ALLOC,
                    leverage_long=LEVERAGE_LONG,
                    leverage_short=LEVERAGE_SHORT,
                    side=trade_side
                )
                if n_cont == 0:
                    min_eq = fm.min_account_for_asset(
                        asset, px, d=POSITION_ALLOC,
                        leverage_long=LEVERAGE_LONG,
                        leverage_short=LEVERAGE_SHORT,
                        side=trade_side
                    )
                    if verbose_feasibility:
                        print(f"  [FEASIBILITY] {asset:<5} {trade_side:<5}  "
                              f"min_equity=${min_eq:>8,.0f}  "
                              f"current=${equity:>8,.0f}  SKIP")
                    bar_key = str(bar_ts.date())
                    if bar_key not in skipped_assets:
                        skipped_assets[bar_key] = []
                    skipped_assets[bar_key].append(asset)
                    # Force weight to 0 for this asset going forward in rebal_w
                    # Find the source rebal bar (one bar before the cost event)
                    src_i = i - 1
                    if src_i >= 0:
                        rebal_w.iloc[src_i][asset] = 0.0
                    continue
                elif verbose_feasibility:
                    n_cont_str = f"TRADE ({n_cont} contracts)"
                    min_eq = fm.min_account_for_asset(asset, px, d=POSITION_ALLOC)
                    print(f"  [FEASIBILITY] {asset:<5} long   "
                          f"min_equity=${min_eq:>8,.0f}  "
                          f"current=${equity:>8,.0f}  {n_cont_str}")

            fee           = compute_bar_fee(asset, px, tier=fee_tier, maker=use_maker)
            asset_cost    = wc * fee
            bar_cost     += asset_cost
            per_asset_cost_df.at[bar_ts, asset] = asset_cost

            # Track AVAX effective RT fee
            if asset == 'AVAX':
                avax_fees_at_rebal.append(fee)

        cost_vals.loc[bar_ts] = bar_cost

    cost = cost_vals

    # Recompute weights after feasibility filtering (rebal_w may have been modified)
    weights = rebal_w.shift(1)
    weight_chg = weights.diff().abs()
    gross_ret = (weights * asset_rets).sum(axis=1)
    net_ret   = gross_ret - cost

    # Trim warm-up
    start = max(lookback, REGIME_MA) + 1
    return {
        "net_ret"           : net_ret.iloc[start:],
        "gross_ret"         : gross_ret.iloc[start:],
        "weights"           : weights.iloc[start:],
        "weight_chg"        : weight_chg.iloc[start:],
        "cost"              : cost.iloc[start:],
        "per_asset_cost"    : per_asset_cost_df.iloc[start:],
        "btc_price"         : prices["BTC"].iloc[start:],
        "btc_ma"            : btc_ma.iloc[start:] if use_regime else None,
        "skipped_assets"    : skipped_assets,
        "avax_fees_at_rebal": avax_fees_at_rebal,
        "fee_tier"              : fee_tier,
        "lookback"              : lookback,
        "dispersion_filtered_pct": round(disp_filtered_pct, 1),
        "dispersion_log"        : disp_log,
        "corr_filtered_pct"     : round(corr_filtered_pct, 1),
        "corr_log"              : corr_log,
        "ret7d_flat_pct"        : round(ret7d_flat_pct, 1),
        "ret7d_partial_pct"     : round(ret7d_partial_pct, 1),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(r: pd.Series, label: str = "") -> dict:
    ann  = np.sqrt(HOURS_PER_YR)
    cum  = (1 + r).cumprod()
    yrs  = len(r) / HOURS_PER_YR
    cagr = (cum.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    sh   = r.mean() / r.std() * ann if r.std() > 0 else 0
    dd   = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    down = r[r < 0]
    so   = r.mean() / down.std() * ann if len(down) > 0 and down.std() > 0 else 0
    return dict(label=label, bars=len(r), years=round(yrs, 2),
                sharpe=round(sh, 3), sortino=round(so, 3),
                cagr_pct=round(cagr, 2), max_dd_pct=round(dd, 2),
                total_ret_pct=round((cum.iloc[-1] - 1) * 100, 2))


def extra_metrics(res: dict) -> dict:
    w   = res["weights"]
    wc  = res["weight_chg"]
    invested     = (w.sum(axis=1) > 0).mean() * 100
    # sum across assets first, then average across time, scale to daily
    turnover_day = float(wc.sum(axis=1).mean() * 24)
    fee_drag_yr  = float(res["cost"].mean() * HOURS_PER_YR * 100)

    # Avg holding period per position (hours)
    on   = (w > 0).astype(int)
    runs = []
    for col in on.columns:
        p = on[col]
        g = (p != p.shift()).cumsum()
        for _, grp in p.groupby(g):
            if grp.iloc[0] == 1:
                runs.append(len(grp))
    avg_hold_hrs = round(np.mean(runs), 1) if runs else 0

    # Regime stats
    if res["btc_ma"] is not None:
        in_regime  = (res["btc_price"] >= res["btc_ma"])
        regime_pct = in_regime.mean() * 100
    else:
        regime_pct = 100.0

    # AVAX average effective RT fee
    avax_fees = res.get("avax_fees_at_rebal", [])
    avax_avg_fee = round(np.mean(avax_fees) * 100, 4) if avax_fees else None

    return dict(pct_time_invested=round(invested, 1),
                turnover_per_day=round(turnover_day, 4),
                fee_drag_pct_yr=round(fee_drag_yr, 4),
                avg_holding_hrs=avg_hold_hrs,
                pct_time_in_regime=round(regime_pct, 1),
                avax_avg_fee_rt_pct=avax_avg_fee,
                dispersion_filtered_pct=res.get("dispersion_filtered_pct", 0.0),
                corr_filtered_pct=res.get("corr_filtered_pct", 0.0),
                ret7d_flat_pct=res.get("ret7d_flat_pct", 0.0),
                ret7d_partial_pct=res.get("ret7d_partial_pct", 0.0))


# ── Per-asset P&L decomposition ───────────────────────────────────────────────
def asset_decomposition(res: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Break down Fold/backtest P&L contribution by asset.
    Returns DataFrame sorted by net_contrib_pct (worst first).
    Columns:
        asset, net_contrib_pct, gross_contrib_pct, fee_drag_pct,
        n_holds_hrs, avg_hold_ret_pct, avg_fee_per_trade_pct
    """
    weights  = res["weights"]
    per_cost = res["per_asset_cost"]

    rets = prices.pct_change().loc[weights.index]

    rows = []
    for asset in weights.columns:
        w        = weights[asset]
        r        = rets[asset] if asset in rets.columns else pd.Series(0.0, index=weights.index)
        held     = w > 0
        n_holds  = int(held.sum())

        gross_contrib = float((w * r).sum()) * 100
        fee_drag      = float(per_cost[asset].sum()) * 100     # positive number
        net_contrib   = gross_contrib - fee_drag

        avg_hold_ret  = float(r[held].mean()) * 100 if n_holds > 0 else 0.0

        traded = per_cost[asset][per_cost[asset] > 0]
        avg_fee = float(traded.mean()) * 100 if len(traded) > 0 else 0.0

        rows.append({
            "asset"               : asset,
            "net_contrib_pct"     : round(net_contrib, 3),
            "gross_contrib_pct"   : round(gross_contrib, 3),
            "fee_drag_pct"        : round(-fee_drag, 3),      # negative = cost
            "n_holds_hrs"         : n_holds,
            "avg_hold_ret_pct"    : round(avg_hold_ret, 4),
            "avg_fee_per_trade_pct": round(avg_fee, 4),
        })

    return pd.DataFrame(rows).sort_values("net_contrib_pct").reset_index(drop=True)


# ── Walk-forward ──────────────────────────────────────────────────────────────
def walk_forward(
    prices: pd.DataFrame,
    fee_tier: int = 1,
    use_maker: bool = False,
    verbose_feasibility: bool = False,
    equity: float = ACCOUNT_EQUITY,
    lookback: int = LOOKBACK_BARS,
    decompose: bool = False,
    n_folds: int = N_FOLDS,
    dispersion_threshold: float = 0.0,
    corr_threshold: float = 0.0,
    use_shorts: bool = False,
    ret7d_gate: bool = False,
    vol_threshold: float = 0.0,
) -> tuple[list[dict], dict, list[dict]]:
    """
    Returns:
        fold_metrics:  per-fold metric dicts
        combined:      metrics over concatenated OOS returns
        fold_extras:   per-fold extra metrics (includes AVAX fee, skipped assets)
    """
    n         = len(prices)
    fold_size = n // n_folds
    fold_ms   = []
    fold_exs  = []
    oos_rets  = []

    print(f"Data:  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Bars:  {n}  |  Folds: {n_folds}  |  ~{fold_size} bars each ({fold_size/HOURS_PER_YR:.2f} yrs)")
    filter_parts = []
    if dispersion_threshold > 0:
        filter_parts.append(f"disp_thresh={dispersion_threshold:.3f}")
    if corr_threshold > 0:
        ct_str = f"corr_thresh={corr_threshold:.3f}"
        if vol_threshold > 0:
            ct_str += f"+vol_thresh={vol_threshold:.3f}"
        filter_parts.append(ct_str)
    if ret7d_gate:
        filter_parts.append("ret7d_gate=ON")
    filter_str = ("  |  " + "  ".join(filter_parts)) if filter_parts else ""
    print(f"Fee tier: T{fee_tier}  |  {'Maker' if use_maker else 'Taker'}  |  equity=${equity:,.0f}  |  lookback={lookback}h{filter_str}\n")

    for i in range(n_folds):
        fp  = prices.iloc[i * fold_size : (i + 1) * fold_size]
        res = backtest(fp, fee_tier=fee_tier, use_maker=use_maker,
                       verbose_feasibility=verbose_feasibility, equity=equity,
                       lookback=lookback, dispersion_threshold=dispersion_threshold,
                       corr_threshold=corr_threshold, use_shorts=use_shorts,
                       ret7d_gate=ret7d_gate, vol_threshold=vol_threshold)
        m   = metrics(res["net_ret"], f"Fold {i+1}")
        ex  = extra_metrics(res)
        fold_ms.append(m)
        fold_exs.append({**ex, "skipped": res["skipped_assets"],
                         "avax_fees": res["avax_fees_at_rebal"]})
        oos_rets.append(res["net_ret"])

        skipped_names = sorted(set(
            a for assets in res["skipped_assets"].values() for a in assets
        ))
        avax_fee_str = (f"{ex['avax_avg_fee_rt_pct']:.4f}%"
                        if ex['avax_avg_fee_rt_pct'] is not None else "N/A")
        skip_str = ", ".join(skipped_names) if skipped_names else "none"

        filter_tags = ""
        if dispersion_threshold > 0:
            disp_pct = res.get("dispersion_filtered_pct", 0.0)
            filter_tags += f"  | disp_filtered={disp_pct:.0f}%"
        if corr_threshold > 0:
            corr_pct = res.get("corr_filtered_pct", 0.0)
            filter_tags += f"  | corr_filtered={corr_pct:.0f}%"
        if ret7d_gate:
            filter_tags += (f"  | 7d_flat={res.get('ret7d_flat_pct',0):.0f}%"
                            f"  7d_partial={res.get('ret7d_partial_pct',0):.0f}%")
        print(f"  Fold {i+1}  {fp.index[0].date()} → {fp.index[-1].date()}"
              f"  Sharpe={m['sharpe']:+.3f}  CAGR={m['cagr_pct']:+.1f}%"
              f"  MaxDD={m['max_dd_pct']:.1f}%"
              f"  | AVAX_fee={avax_fee_str}"
              f"  | skipped=[{skip_str}]{filter_tags}")

        if decompose:
            dec = asset_decomposition(res, fp)
            print(f"\n  Per-asset decomposition — Fold {i+1}:")
            print("  " + dec.to_string(index=False).replace("\n", "\n  "))
            top_drag = dec[dec["net_contrib_pct"] < 0]
            if not top_drag.empty:
                worst = top_drag.iloc[0]
                total_net = dec["net_contrib_pct"].sum()
                pct_of_loss = worst["net_contrib_pct"] / total_net * 100 if total_net < 0 else 0
                print(f"\n  Worst actor: {worst['asset']}  net={worst['net_contrib_pct']:.3f}%"
                      f"  ({abs(pct_of_loss):.0f}% of total fold loss)")
                if abs(pct_of_loss) >= 40:
                    print(f"  *** FLAG: {worst['asset']} accounts for >{abs(pct_of_loss):.0f}% of Fold {i+1} loss ***")
            print()

    combined = pd.concat(oos_rets)
    cm       = metrics(combined, "Combined OOS")
    return fold_ms, cm, fold_exs


# ── Fee sweep ─────────────────────────────────────────────────────────────────
def backtest_fee_sweep(
    prices: pd.DataFrame,
    fee_tiers: list[int],
    use_maker: bool = False,
    equity: float = ACCOUNT_EQUITY,
    lookback: int = LOOKBACK_BARS,
    n_folds: int = N_FOLDS,
    dispersion_threshold: float = 0.0,
    corr_threshold: float = 0.0,
    use_shorts: bool = False,
    ret7d_gate: bool = False,
    vol_threshold: float = 0.0,
) -> list[dict]:
    """
    Run full walk-forward backtest for each fee tier in fee_tiers.
    Returns list of result dicts, one per tier.
    """
    results = []
    print("=" * 76)
    print("FEE TIER SWEEP")
    print("=" * 76)
    print(f"{'Tier':<6} {'30d Vol':>12} {'Taker/side':>12} {'OOS Sharpe':>12}"
          f" {'OOS CAGR':>10} {'MaxDD':>8}  Verdict")
    print("─" * 76)

    tier_vol_labels = {
        1: "<$1K", 2: "$1K-$10K", 3: "$10K-$50K", 4: "$50K-$500K",
        5: "$500K-$1M", 6: "$1M-$15M", 7: "$15M-$50M", 8: "$50M-$100M",
        9: "$100M-$250M", 10: "$250M+",
    }

    for tier in fee_tiers:
        fold_ms, combined, fold_exs = walk_forward(
            prices, fee_tier=tier, use_maker=use_maker, verbose_feasibility=False,
            equity=equity, lookback=lookback, n_folds=n_folds,
            dispersion_threshold=dispersion_threshold, corr_threshold=corr_threshold,
            use_shorts=use_shorts, ret7d_gate=ret7d_gate, vol_threshold=vol_threshold,
        )
        taker_pct = fm.TAKER_PER_SIDE[tier] * 100
        vol_lbl   = tier_vol_labels.get(tier, "?")
        verdict   = "PASS" if combined["sharpe"] >= SHARPE_MIN else "FAIL <<<STOP>>>"

        print(f"  T{tier:<4} {vol_lbl:>12} {taker_pct:>11.3f}%"
              f" {combined['sharpe']:>+12.3f}"
              f" {combined['cagr_pct']:>+9.1f}%"
              f" {combined['max_dd_pct']:>7.1f}%  {verdict}")

        # Per-fold detail
        for j, (fm_dict, ex) in enumerate(zip(fold_ms, fold_exs)):
            skipped_names = sorted(set(
                a for assets in ex["skipped"].values() for a in assets
            ))
            avax_str = (f"{ex['avax_avg_fee_rt_pct']:.4f}%"
                        if ex['avax_avg_fee_rt_pct'] is not None else "N/A")
            skip_str = ", ".join(skipped_names) if skipped_names else "none"
            print(f"    Fold {j+1}: Sharpe={fm_dict['sharpe']:+.3f}"
                  f"  CAGR={fm_dict['cagr_pct']:+.1f}%"
                  f"  AVAX_fee={avax_str}"
                  f"  skipped=[{skip_str}]")

        results.append({
            "tier": tier,
            "taker_per_side_pct": taker_pct,
            "fold_metrics": fold_ms,
            "combined_oos": combined,
            "fold_extras": fold_exs,
            "passed": combined["sharpe"] >= SHARPE_MIN,
        })
        print()

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(prices: pd.DataFrame, fee_tier: int = 1, lookback: int = LOOKBACK_BARS,
         n_folds: int = N_FOLDS, dispersion_threshold: float = 0.0,
         corr_threshold: float = 0.0, use_shorts: bool = False,
         ret7d_gate: bool = False, vol_threshold: float = 0.0):
    res_regime   = backtest(prices, use_regime=True,  fee_tier=fee_tier, lookback=lookback,
                            dispersion_threshold=dispersion_threshold,
                            corr_threshold=corr_threshold, use_shorts=use_shorts,
                            ret7d_gate=ret7d_gate, vol_threshold=vol_threshold)
    res_noregime = backtest(prices, use_regime=False, fee_tier=fee_tier, lookback=lookback,
                            dispersion_threshold=dispersion_threshold,
                            corr_threshold=corr_threshold, use_shorts=use_shorts,
                            ret7d_gate=ret7d_gate, vol_threshold=vol_threshold)

    nr   = res_regime["net_ret"]
    nr2  = res_noregime["net_ret"]
    cum  = (1 + nr).cumprod()
    cum2 = (1 + nr2).cumprod()
    dd   = ((cum - cum.cummax()) / cum.cummax()) * 100

    n = len(prices)
    fold_size = n // n_folds
    fold_spans = []
    for i in range(n_folds):
        fp  = prices.iloc[i * fold_size : (i + 1) * fold_size]
        res = backtest(fp, fee_tier=fee_tier, lookback=lookback,
                       dispersion_threshold=dispersion_threshold,
                       corr_threshold=corr_threshold, use_shorts=use_shorts,
                       ret7d_gate=ret7d_gate, vol_threshold=vol_threshold)
        idx = res["net_ret"].index
        if len(idx):
            fold_spans.append((idx[0], idx[-1]))

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    shade  = ["#e3f2fd", "#f3e5f5", "#e8f5e9"]
    for i, (s, e) in enumerate(fold_spans):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(s, e, alpha=0.2, color=shade[i])
        ax1.axvline(s, color="#bdbdbd", lw=0.5, ls="--")

    ax1.plot(cum.index,  cum.values,  color="#1565c0", lw=1.3,
             label=f"With regime filter (T{fee_tier} fees)")
    ax1.plot(cum2.index, cum2.values, color="#90a4ae", lw=0.9,
             ls="--", label="No regime filter", alpha=0.7)
    ax1.axhline(1.0, color="#9e9e9e", lw=0.5, ls=":")
    ax1.set_ylabel("Cumulative Return (x)", fontsize=10)
    ax1.set_title(
        f"Cross-Sectional Momentum v2  |  14-asset universe  |  "
        f"24h signal, top-2  |  Weekly rebal  |  BTC 168h MA regime filter  |  Fee T{fee_tier}",
        fontsize=10, pad=8)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.2)
    ax1.tick_params(labelbottom=False)

    ax2.fill_between(dd.index, dd.values, 0, color="#c62828", alpha=0.5)
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.grid(alpha=0.2)
    ax2.tick_params(labelbottom=False)

    btc = res_regime["btc_price"]
    ma  = res_regime["btc_ma"]
    ax3.plot(btc.index, btc.values, color="#f57c00", lw=0.8, label="BTC price")
    ax3.plot(ma.index,  ma.values,  color="#1a237e", lw=1.0, ls="--", label="168h MA")
    bear = btc < ma
    ax3.fill_between(btc.index, btc.values, ma.values,
                     where=bear, color="#c62828", alpha=0.15, label="Bear regime (cash)")
    ax3.set_ylabel("BTC (USD)", fontsize=10)
    ax3.set_xlabel("Date", fontsize=10)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(alpha=0.2)

    out_path = OUT_DIR / "backtest_v2_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> bool:
    parser = argparse.ArgumentParser(
        description="Cappuccino momentum backtest v2 — dynamic Coinbase Derivatives fee model"
    )
    parser.add_argument(
        "--fee-sweep", action="store_true",
        help="Run walk-forward backtest for each specified fee tier"
    )
    parser.add_argument(
        "--tiers", nargs="+", type=int, default=[1],
        metavar="TIER",
        help="Fee tier(s) to test (1=lowest vol/highest fee ... 10=highest vol/lowest fee)"
    )
    parser.add_argument(
        "--maker", action="store_true",
        help="Use maker fee rates instead of taker"
    )
    parser.add_argument(
        "--feasibility", action="store_true",
        help="Print per-asset capital feasibility check at first rebal bar"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip chart generation"
    )
    parser.add_argument(
        "--equity", type=float, default=ACCOUNT_EQUITY,
        metavar="USD",
        help=f"Account equity in USD for capital feasibility checks (default: {ACCOUNT_EQUITY})"
    )
    parser.add_argument(
        "--lookback", type=int, default=LOOKBACK_BARS,
        metavar="HOURS",
        help=f"Momentum signal lookback in hours (default: {LOOKBACK_BARS})"
    )
    parser.add_argument(
        "--decompose", action="store_true",
        help="Print per-asset P&L decomposition for each fold"
    )
    parser.add_argument(
        "--fee-tier", type=int, default=None, metavar="TIER",
        help="Shorthand for --tiers N (single tier; overrides --tiers)"
    )
    parser.add_argument(
        "--folds", type=int, default=N_FOLDS,
        metavar="N",
        help=f"Number of walk-forward folds (default: {N_FOLDS}, range: 3–7)"
    )
    parser.add_argument(
        "--dispersion-threshold", type=float, default=0.0,
        metavar="FLOAT",
        help="Go flat when xs_std(ret_24h) < threshold (0.0 = disabled)"
    )
    parser.add_argument(
        "--corr-threshold", type=float, default=0.0,
        metavar="FLOAT",
        help="Go flat when median pairwise 168h return correlation > threshold (0.0 = disabled)"
    )
    parser.add_argument(
        "--shorts", action="store_true",
        help="Short bottom-2 assets at -25%% each (symmetric long/short)"
    )
    parser.add_argument(
        "--ret7d-gate", action="store_true",
        help="Only select assets with positive 7-day return; re-rank within eligible set"
    )
    parser.add_argument(
        "--vol-threshold", type=float, default=0.0,
        metavar="FLOAT",
        help="Annualised avg realised vol gate: go flat only when corr AND vol > threshold (0=disabled)"
    )
    parser.add_argument(
        "--exclude-assets", nargs="+", default=[], metavar="ASSET",
        help="Assets to remove from the universe (e.g. --exclude-assets AVAX XLM)"
    )
    parser.add_argument(
        "--assets", nargs="+", default=None, metavar="ASSET",
        help="Explicit asset list — overrides default universe and --exclude-assets"
    )
    args = parser.parse_args()

    # --fee-tier N is sugar for --tiers N (single tier, no sweep)
    if args.fee_tier is not None:
        args.tiers = [args.fee_tier]

    print("=" * 76)
    print("CROSS-SECTIONAL MOMENTUM BACKTEST  v2")
    print("Dynamic Coinbase Derivatives fee model: taker+NFA per asset per bar")
    print("=" * 76)

    prices = load_prices()
    print(f"Loaded {len(prices)} bars x {len(prices.columns)} assets: {list(prices.columns)}")

    # Apply universe filters
    if args.assets:
        missing = [a for a in args.assets if a not in prices.columns]
        if missing:
            print(f"  [WARN] Requested assets not in data: {missing}")
        prices = prices[[a for a in args.assets if a in prices.columns]]
        print(f"Universe override (--assets): {list(prices.columns)}")
    elif args.exclude_assets:
        removed = [a for a in args.exclude_assets if a in prices.columns]
        not_found = [a for a in args.exclude_assets if a not in prices.columns]
        prices = prices[[a for a in prices.columns if a not in args.exclude_assets]]
        print(f"Excluded: {removed}  |  Active universe ({len(prices.columns)}): {list(prices.columns)}")
        if not_found:
            print(f"  [WARN] Exclusion targets not found in data: {not_found}")
    print()

    # ── Capital feasibility snapshot ─────────────────────────────────────────
    if args.feasibility:
        print("=" * 76)
        print(f"CAPITAL FEASIBILITY CHECK  (equity=${args.equity:.0f}, alloc={POSITION_ALLOC*100:.0f}%)")
        print("=" * 76)
        first_full_bar = max(LOOKBACK_BARS, REGIME_MA) + 1
        feasibility_check(prices, first_full_bar,
                          equity=args.equity,
                          d=POSITION_ALLOC,
                          verbose=True)
        print()

    # ── Fee sweep mode ────────────────────────────────────────────────────────
    if args.fee_sweep:
        sweep_results = backtest_fee_sweep(
            prices, fee_tiers=args.tiers, use_maker=args.maker,
            equity=args.equity, lookback=args.lookback,
            n_folds=args.folds, dispersion_threshold=args.dispersion_threshold,
            corr_threshold=args.corr_threshold,
            ret7d_gate=args.ret7d_gate, vol_threshold=args.vol_threshold,
        )

        # Hard stop check
        any_negative = any(
            m["sharpe"] < 0
            for r in sweep_results
            for m in r["fold_metrics"]
        )
        if any_negative:
            print("=" * 76)
            print("HARD STOP: At least one fold Sharpe < 0 after fee correction.")
            print("DO NOT proceed to engine_v2.py. Investigate before continuing.")
            print("=" * 76)
        else:
            # Check overall pass
            all_pass = all(r["passed"] for r in sweep_results)
            print("=" * 76)
            if not all_pass:
                failing = [r["tier"] for r in sweep_results if not r["passed"]]
                print(f"WARNING: Tiers {failing} have Combined OOS Sharpe < {SHARPE_MIN}.")
                print("Review fee tier assumptions before proceeding to live.")
            else:
                print(f"ALL TIERS PASS: Combined OOS Sharpe >= {SHARPE_MIN} at all tested tiers.")
            print("=" * 76)

        # Save results
        out_path = OUT_DIR / "backtest_v2_fee_sweep.json"
        with open(out_path, "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        print(f"Sweep results -> {out_path}")

        # Plot using first (most conservative) tier
        if not args.no_plot:
            plot(prices, fee_tier=args.tiers[0], lookback=args.lookback,
                 n_folds=args.folds, dispersion_threshold=args.dispersion_threshold,
                 corr_threshold=args.corr_threshold,
                 ret7d_gate=args.ret7d_gate, vol_threshold=args.vol_threshold)

        return not any_negative

    # ── Single-tier mode ──────────────────────────────────────────────────────
    tier = args.tiers[0]
    print(f"Single-tier run: T{tier}  ({'maker' if args.maker else 'taker'})\n")

    print(f"WALK-FORWARD ({args.folds} sequential folds, no random splits)")
    print("─" * 76)
    fold_ms, combined, fold_exs = walk_forward(
        prices, fee_tier=tier, use_maker=args.maker,
        verbose_feasibility=args.feasibility, equity=args.equity,
        lookback=args.lookback, decompose=args.decompose,
        n_folds=args.folds, dispersion_threshold=args.dispersion_threshold,
        corr_threshold=args.corr_threshold, use_shorts=args.shorts,
        ret7d_gate=args.ret7d_gate, vol_threshold=args.vol_threshold,
    )

    # Full-sample
    res_full = backtest(prices, fee_tier=tier, use_maker=args.maker,
                        equity=args.equity, lookback=args.lookback,
                        dispersion_threshold=args.dispersion_threshold,
                        corr_threshold=args.corr_threshold,
                        use_shorts=args.shorts,
                        ret7d_gate=args.ret7d_gate, vol_threshold=args.vol_threshold)
    full_m   = metrics(res_full["net_ret"], "Full sample")
    extra    = extra_metrics(res_full)

    print("\n" + "=" * 76)
    print("FULL-SAMPLE METRICS")
    print("=" * 76)
    for k, v in {**full_m, **extra}.items():
        if k == "label": continue
        print(f"  {k:<35} {v}")

    # No-regime comparison
    res_raw = backtest(prices, use_regime=False, fee_tier=tier,
                       use_maker=args.maker, lookback=args.lookback,
                       dispersion_threshold=args.dispersion_threshold,
                       corr_threshold=args.corr_threshold,
                       use_shorts=args.shorts,
                       ret7d_gate=args.ret7d_gate, vol_threshold=args.vol_threshold)
    raw_m   = metrics(res_raw["net_ret"], "No regime filter")
    print(f"\n  {'--- without regime filter ---'}")
    print(f"  {'sharpe (no filter)':<35} {raw_m['sharpe']}")
    print(f"  {'cagr_pct (no filter)':<35} {raw_m['cagr_pct']}")

    print("\n" + "─" * 76)
    print("WALK-FORWARD SUMMARY")
    print("─" * 76)
    for m in fold_ms:
        print(f"  {m['label']}: Sharpe={m['sharpe']:+.3f}  CAGR={m['cagr_pct']:+.1f}%"
              f"  MaxDD={m['max_dd_pct']:.1f}%  ({m['years']:.2f} yrs)")
    print(f"  Combined:  Sharpe={combined['sharpe']:+.3f}  CAGR={combined['cagr_pct']:+.1f}%"
          f"  MaxDD={combined['max_dd_pct']:.1f}%")

    print("\n" + "=" * 76)
    passed = combined["sharpe"] >= SHARPE_MIN
    if not passed:
        print(f"HARD STOP: Combined OOS Sharpe {combined['sharpe']:.3f} < {SHARPE_MIN}")
        print("DO NOT proceed to live trading.")
    else:
        print(f"PASS: Combined OOS Sharpe {combined['sharpe']:.3f} >= {SHARPE_MIN}")

    if not args.no_plot:
        plot(prices, fee_tier=tier, lookback=args.lookback,
             n_folds=args.folds, dispersion_threshold=args.dispersion_threshold,
             corr_threshold=args.corr_threshold, use_shorts=args.shorts,
             ret7d_gate=args.ret7d_gate, vol_threshold=args.vol_threshold)

    result = dict(fold_metrics=fold_ms, combined_oos=combined,
                  full_sample={**full_m, **extra},
                  no_regime=raw_m, passed=passed,
                  fee_tier=tier)
    out_path = OUT_DIR / "backtest_v2_metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Metrics -> {out_path}")
    return passed


if __name__ == "__main__":
    main()
