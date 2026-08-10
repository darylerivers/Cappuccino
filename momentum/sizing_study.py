"""
Position Sizing Study — Parts 2A–5
Exchange: Coinbase Derivatives (4.3× leverage)
Fees:     0.20% RT taker
Signal:   ret_24h + vol_accel composite z-score (14-asset universe)
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = Path.home() / "cappuccino" / "data"
RESULTS_DIR = Path.home() / "cappuccino" / "momentum" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA",
    "AVAX", "LINK", "LTC", "BCH", "XLM", "SUI", "DOT", "HBAR",
]
LOOKBACK     = 24
TOP_N        = 2
REBAL_BARS   = 168
REGIME_MA    = 168
REGIME_BUF   = 0.005
HOURS_PER_YR = 8760
FEE_RT       = 0.0020   # 0.20% RT taker (Coinbase Derivatives)
LEVERAGE     = 4.3
N_FOLDS      = 3


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all():
    closes, volumes = {}, {}
    for asset in ASSETS:
        df = pd.read_parquet(DATA_DIR / f"{asset}_1h.parquet")
        df = df.set_index("datetime")
        df.index = pd.to_datetime(df.index, utc=True)
        closes[asset]  = df["close"]
        volumes[asset] = df["volume"]
    idx = pd.DataFrame(closes).dropna().index
    return pd.DataFrame(closes).loc[idx], pd.DataFrame(volumes).loc[idx]


# ── Signal ────────────────────────────────────────────────────────────────────
def composite_signal(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional z-score of (ret_24h + vol_accel) / 2 at each bar.
    Fully vectorised — no Python-level row loop.
    """
    ret24 = prices.pct_change(LOOKBACK)

    vol_now  = volumes.rolling(LOOKBACK).sum()
    vol_prev = volumes.rolling(LOOKBACK).sum().shift(LOOKBACK)
    vaccel   = vol_now.div(vol_prev.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    def cs_zscore(df):
        mu  = df.mean(axis=1)
        std = df.std(axis=1).replace(0, np.nan)
        return df.subtract(mu, axis=0).divide(std, axis=0).fillna(0)

    return (cs_zscore(ret24) + cs_zscore(vaccel)) / 2.0


# ── Regime filter ─────────────────────────────────────────────────────────────
def regime_series(prices: pd.DataFrame) -> pd.Series:
    btc_ma = prices["BTC"].rolling(REGIME_MA).mean()
    state  = True
    flags  = []
    for i in range(len(prices)):
        ma = btc_ma.iloc[i]
        if pd.isna(ma):
            flags.append(True)
            continue
        px = prices["BTC"].iloc[i]
        if state:
            if px < ma * (1 - REGIME_BUF):
                state = False
        else:
            if px > ma * (1 + REGIME_BUF):
                state = True
        flags.append(state)
    return pd.Series(flags, index=prices.index, dtype=bool)


# ── Core backtest ─────────────────────────────────────────────────────────────
def run_backtest(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    deploy: float,          # per-position margin fraction (e.g. 0.25)
    lev: float = LEVERAGE,
    fee: float = FEE_RT,
    vol_target: float = None,   # if set → inverse-vol weight to hit this ann vol
    libro: bool = False,        # if True → margin-buffer-aware scaling
) -> dict:
    """
    Weights represent fraction of equity posted as margin per position.
    Leveraged P&L = weight × leverage × asset_return.
    """
    n = len(prices)
    asset_rets = prices.pct_change()
    sig        = composite_signal(prices, volumes)
    in_regime  = regime_series(prices)

    raw_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Vol for vol-scaled mode
    vol_ann = None
    if vol_target is not None:
        vol_ann = asset_rets.rolling(24).std() * np.sqrt(HOURS_PER_YR)

    # LIBRO equity tracker
    libro_equity = 1.0

    for i in range(n):
        if not in_regime.iloc[i]:
            continue

        row    = sig.iloc[i]
        ranked = row.rank(ascending=False, na_option="bottom")
        top    = [a for a in prices.columns if ranked.get(a, 999) <= TOP_N]
        if not top:
            continue

        if vol_target is not None:
            # Inverse-vol weight the top assets to hit portfolio vol target
            vols = vol_ann.iloc[i][top].replace(0, np.nan).dropna()
            if vols.empty:
                per_pos = deploy
                for a in top:
                    raw_w.at[prices.index[i], a] = per_pos
                continue
            inv_v   = 1.0 / vols
            w_norm  = inv_v / inv_v.sum()
            # estimate realised portfolio vol ≈ sum(w_i × vol_i) (no correlation)
            port_vol_est = (w_norm * vols).sum()
            if port_vol_est > 0:
                scale = vol_target / port_vol_est
            else:
                scale = 1.0
            for a in top:
                w = min(w_norm.get(a, 0) * scale, 0.40)
                raw_w.at[prices.index[i], a] = w

        elif libro:
            # LIBRO: scale positions down when equity has fallen (margin buffer shrinking)
            # warning threshold: equity drop = 0.60 × total_deploy
            warn_drop = 0.60 * (deploy * TOP_N)
            # Scale factor: 1 when equity=1.0, 0 when equity dropped to (1 - warn_drop)
            margin_headroom = (libro_equity - (1.0 - warn_drop)) / warn_drop
            margin_headroom = max(0.0, min(1.0, margin_headroom))
            libro_scale     = 0.25 + 0.75 * margin_headroom  # keep at least 25%
            for a in top:
                raw_w.at[prices.index[i], a] = deploy * libro_scale

        else:
            for a in top:
                raw_w.at[prices.index[i], a] = deploy

        # Update LIBRO equity tracker with prior bar's return
        if libro and i > 0:
            prev_lev_ret = (raw_w.iloc[i - 1] * lev * asset_rets.iloc[i]).sum()
            libro_equity *= (1 + prev_lev_ret)
            libro_equity  = max(libro_equity, 0.01)

    # Weekly rebalance: carry forward between rebal bars
    rebal_w = raw_w.copy()
    for i in range(1, len(rebal_w)):
        if i % REBAL_BARS != 0:
            rebal_w.iloc[i] = rebal_w.iloc[i - 1]

    # Apply regime (clear positions when in bear)
    for i in range(len(rebal_w)):
        if not in_regime.iloc[i]:
            rebal_w.iloc[i] = 0.0

    # No-lookahead shift
    weights = rebal_w.shift(1)

    # Leveraged returns
    gross_ret  = (weights * lev * asset_rets).sum(axis=1)
    weight_chg = weights.diff().abs().sum(axis=1)
    cost       = weight_chg * fee       # fee on the margin fraction (conservative)
    net_ret    = gross_ret - cost

    # Trim warm-up
    start   = max(LOOKBACK, REGIME_MA) + 1
    net_ret = net_ret.iloc[start:]
    weights = weights.iloc[start:]
    in_regime_trimmed = in_regime.iloc[start:]

    return {
        "net_ret"    : net_ret,
        "weights"    : weights,
        "in_regime"  : in_regime_trimmed,
        "btc_price"  : prices["BTC"].iloc[start:],
        "btc_ma"     : prices["BTC"].rolling(REGIME_MA).mean().iloc[start:],
        "deploy"     : deploy,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def metrics(r: pd.Series) -> dict:
    ann  = np.sqrt(HOURS_PER_YR)
    cum  = (1 + r).cumprod()
    yrs  = len(r) / HOURS_PER_YR
    cagr = (cum.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    sh   = r.mean() / r.std() * ann if r.std() > 0 else 0
    dd   = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return dict(
        sharpe=round(sh, 3),
        cagr_pct=round(cagr, 2),
        max_dd_pct=round(dd, 2),
        total_ret_pct=round((cum.iloc[-1] - 1) * 100, 2),
    )


def fold_metrics(prices, volumes, deploy, lev=LEVERAGE, fee=FEE_RT,
                 vol_target=None, libro=False):
    """Run backtest on each fold + combined; return (fold_list, combined_dict)."""
    n         = len(prices)
    fold_sz   = n // N_FOLDS
    fold_rets = []
    fold_ms   = []
    for i in range(N_FOLDS):
        fp = prices.iloc[i * fold_sz: (i + 1) * fold_sz]
        fv = volumes.iloc[i * fold_sz: (i + 1) * fold_sz]
        res = run_backtest(fp, fv, deploy, lev, fee, vol_target, libro)
        m   = metrics(res["net_ret"])
        fold_ms.append(m)
        fold_rets.append(res["net_ret"])
    combined = pd.concat(fold_rets)
    return fold_ms, metrics(combined)


# ── Margin event tracking ─────────────────────────────────────────────────────
def margin_events(prices, volumes, deploy, lev=LEVERAGE, fee=FEE_RT,
                  vol_target=None, libro=False):
    """
    Per rebalance window, check if cumulative leveraged equity drop exceeds:
      - warn_loss  = 0.60 × total_deploy  (60% of margin consumed)
      - liq_loss   = 1.00 × total_deploy  (100% → liquidation)
    Returns P(warn event per week), P(liq event per sample).
    """
    res        = run_backtest(prices, volumes, deploy, lev, fee, vol_target, libro)
    net        = res["net_ret"]
    total_dep  = deploy * TOP_N   # total deployed fraction
    warn_loss  = 0.60 * total_dep
    liq_loss   = 1.00 * total_dep

    n_rebal   = len(net) // REBAL_BARS
    warn_hits = 0
    liq_hits  = 0

    for k in range(n_rebal):
        window = net.iloc[k * REBAL_BARS: (k + 1) * REBAL_BARS]
        if len(window) == 0:
            continue
        cum_drop = (1 + window).cumprod() - 1   # cumulative return from window start
        min_cum  = cum_drop.min()
        if min_cum < -liq_loss:
            liq_hits  += 1
            warn_hits += 1
        elif min_cum < -warn_loss:
            warn_hits += 1

    warn_pct = warn_hits / n_rebal * 100 if n_rebal > 0 else 0
    liq_pct  = liq_hits  / n_rebal * 100 if n_rebal > 0 else 0
    return round(warn_pct, 1), round(liq_pct, 2)


# ── Part 2A: Fixed equal-weight grid ─────────────────────────────────────────
def part_2a(prices, volumes):
    print("\n" + "=" * 72)
    print("PART 2A — Fixed equal-weight deployment grid (4.3× leverage)")
    print("=" * 72)
    print(f"  Margin warning threshold: 60% consumed  |  Liquidation: 100%")
    print(f"  Fee: {FEE_RT*100:.2f}% RT  |  Universe: 14 assets  |  Top-2 weekly\n")

    grid   = [0.10, 0.15, 0.20, 0.25, 0.30]
    header = (f"{'d':>5}  {'Total':>6}  "
              f"{'Fold1':>7} {'Fold2':>7} {'Fold3':>7}  "
              f"{'OOS Sh':>7} {'OOS CAGR':>9} {'MaxDD':>7}  "
              f"{'Warn%':>6} {'Liq%':>5}")
    print(header)
    print("─" * 72)

    results = []
    best    = None

    for d in grid:
        fms, comb = fold_metrics(prices, volumes, deploy=d)
        warn_pct, liq_pct = margin_events(prices, volumes, deploy=d)
        folds_pos = all(m["sharpe"] > 0 for m in fms)

        row = dict(d=d, total_deploy=round(d * TOP_N, 2),
                   fold_sharpes=[m["sharpe"] for m in fms],
                   oos_sharpe=comb["sharpe"], oos_cagr=comb["cagr_pct"],
                   max_dd=comb["max_dd_pct"],
                   warn_pct=warn_pct, liq_pct=liq_pct,
                   folds_all_pos=folds_pos)
        results.append(row)

        flag = ""
        if liq_pct < 1.0 and folds_pos and warn_pct < 10.0:
            flag = " ← candidate"
            if best is None or comb["sharpe"] > best["oos_sharpe"]:
                best = row

        print(f"  {d:.2f}  {d*TOP_N:.2f}   "
              f"  {fms[0]['sharpe']:+.3f}  {fms[1]['sharpe']:+.3f}  {fms[2]['sharpe']:+.3f}  "
              f"  {comb['sharpe']:+.4f}  {comb['cagr_pct']:+7.1f}%  {comb['max_dd_pct']:6.1f}%  "
              f"  {warn_pct:5.1f}  {liq_pct:5.2f}{flag}")

    print()
    if best:
        print(f"  BEST (constraints met): d={best['d']:.2f}  "
              f"OOS Sharpe={best['oos_sharpe']:+.3f}  "
              f"Warn={best['warn_pct']:.1f}%  Liq={best['liq_pct']:.2f}%")
    else:
        print("  No candidate passed all constraints.")
    return results, best


# ── Part 2B: Vol-scaled weight grid ───────────────────────────────────────────
def part_2b(prices, volumes):
    print("\n" + "=" * 72)
    print("PART 2B — Inverse-vol scaled sizing (target portfolio ann vol)")
    print("=" * 72)
    print(f"  Base deploy per position capped at 0.40\n")

    targets = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    header  = (f"{'T':>6}  {'Fold1':>7} {'Fold2':>7} {'Fold3':>7}  "
               f"{'OOS Sh':>7} {'OOS CAGR':>9} {'MaxDD':>7}")
    print(header)
    print("─" * 72)

    results = []
    best    = None

    for T in targets:
        fms, comb = fold_metrics(prices, volumes, deploy=0.20,
                                 vol_target=T)
        folds_pos = all(m["sharpe"] > 0 for m in fms)
        row = dict(vol_target=T,
                   fold_sharpes=[m["sharpe"] for m in fms],
                   oos_sharpe=comb["sharpe"], oos_cagr=comb["cagr_pct"],
                   max_dd=comb["max_dd_pct"], folds_all_pos=folds_pos)
        results.append(row)

        flag = " ← best" if (best is None or comb["sharpe"] > best["oos_sharpe"]) and folds_pos else ""
        if folds_pos and flag:
            best = row

        print(f"  T={T:.2f}  {fms[0]['sharpe']:+.3f}  {fms[1]['sharpe']:+.3f}  "
              f"{fms[2]['sharpe']:+.3f}   {comb['sharpe']:+.4f}  "
              f"{comb['cagr_pct']:+7.1f}%  {comb['max_dd_pct']:6.1f}%{flag}")

    print()
    if best:
        print(f"  BEST vol target: T={best['vol_target']:.2f}  "
              f"OOS Sharpe={best['oos_sharpe']:+.3f}")
    return results, best


# ── Part 2C: LIBRO-style margin-aware ─────────────────────────────────────────
def part_2c(prices, volumes, baseline_d):
    print("\n" + "=" * 72)
    print("PART 2C — LIBRO margin-buffer-aware dynamic sizing")
    print("=" * 72)
    print(f"  Baseline d={baseline_d:.2f} | Scales down when equity approaches "
          f"margin warning threshold\n")

    fms_eq, comb_eq     = fold_metrics(prices, volumes, deploy=baseline_d)
    fms_lb, comb_lb     = fold_metrics(prices, volumes, deploy=baseline_d, libro=True)
    warn_eq, liq_eq     = margin_events(prices, volumes, deploy=baseline_d)
    warn_lb, liq_lb     = margin_events(prices, volumes, deploy=baseline_d, libro=True)

    print(f"  {'Method':<22}  {'Fold1':>7} {'Fold2':>7} {'Fold3':>7}  "
          f"{'OOS Sh':>7} {'CAGR':>8} {'MaxDD':>7}  {'Warn%':>6} {'Liq%':>5}")
    print("  " + "─" * 70)

    for label, fms, comb, warn, liq in [
        ("Equal-weight",    fms_eq, comb_eq, warn_eq, liq_eq),
        ("LIBRO-scaled",    fms_lb, comb_lb, warn_lb, liq_lb),
    ]:
        print(f"  {label:<22}  "
              f"{fms[0]['sharpe']:+.3f}  {fms[1]['sharpe']:+.3f}  {fms[2]['sharpe']:+.3f}  "
              f"{comb['sharpe']:+.4f}  {comb['cagr_pct']:+6.1f}%  "
              f"{comb['max_dd_pct']:6.1f}%  {warn:5.1f}  {liq:5.2f}")

    delta_sh  = round(comb_lb["sharpe"]  - comb_eq["sharpe"],  3)
    delta_dd  = round(comb_lb["max_dd_pct"] - comb_eq["max_dd_pct"], 2)
    delta_warn= round(warn_lb - warn_eq, 1)
    print(f"\n  LIBRO vs Equal-weight Δ:  ΔSharpe={delta_sh:+.3f}  "
          f"ΔMaxDD={delta_dd:+.2f}%  ΔWarn={delta_warn:+.1f}%")

    return {
        "equal":  dict(fold_sharpes=[m["sharpe"] for m in fms_eq],
                       **comb_eq, warn_pct=warn_eq, liq_pct=liq_eq),
        "libro":  dict(fold_sharpes=[m["sharpe"] for m in fms_lb],
                       **comb_lb, warn_pct=warn_lb, liq_pct=liq_lb),
        "delta":  dict(sharpe=delta_sh, max_dd=delta_dd, warn=delta_warn),
    }


# ── Part 3: DiD regression ─────────────────────────────────────────────────────
def part_3(prices, volumes, baseline_d):
    try:
        import statsmodels.api as sm
    except ImportError:
        print("\nPart 3 skipped — statsmodels not installed.")
        return None

    print("\n" + "=" * 72)
    print("PART 3 — Difference-in-Differences: LIBRO vs Equal-weight")
    print("=" * 72)
    print("  Model: weekly_ret ~ LIBRO + Bull + LIBRO×Bull + const")
    print("  Treatment=LIBRO(1) vs Equal-weight(0), Post=Bull regime\n")

    # Run both on full dataset
    res_eq = run_backtest(prices, volumes, deploy=baseline_d)
    res_lb = run_backtest(prices, volumes, deploy=baseline_d, libro=True)

    net_eq = res_eq["net_ret"]
    net_lb = res_lb["net_ret"]
    regime = res_eq["in_regime"]

    # Align on common index
    idx = net_eq.index.intersection(net_lb.index).intersection(regime.index)
    net_eq = net_eq.loc[idx]
    net_lb = net_lb.loc[idx]
    regime = regime.loc[idx]

    # Resample to weekly returns for DiD
    eq_wk  = (1 + net_eq).resample("7D").prod() - 1
    lb_wk  = (1 + net_lb).resample("7D").prod() - 1
    reg_wk = regime.resample("7D").mean() > 0.5   # majority of week in bull

    # Stack into long format
    common_wk = eq_wk.index.intersection(lb_wk.index).intersection(reg_wk.index)

    rows = []
    for t in common_wk:
        rows.append(dict(ret=eq_wk[t], libro=0, bull=int(reg_wk[t])))
        rows.append(dict(ret=lb_wk[t], libro=1, bull=int(reg_wk[t])))
    panel = pd.DataFrame(rows).dropna()

    panel["libro_x_bull"] = panel["libro"] * panel["bull"]
    X = sm.add_constant(panel[["libro", "bull", "libro_x_bull"]])
    y = panel["ret"]

    ols = sm.OLS(y, X).fit(cov_type="HC3")

    print(ols.summary(title="OLS DiD — LIBRO vs Equal-weight"))
    print()

    coef  = ols.params
    pvals = ols.pvalues
    print(f"  Interpretation:")
    print(f"    const          = {coef['const']:+.5f}  (equal-weight, bear regime baseline)")
    print(f"    libro          = {coef['libro']:+.5f}  p={pvals['libro']:.3f}  "
          f"(LIBRO effect in bear)")
    print(f"    bull           = {coef['bull']:+.5f}  p={pvals['bull']:.3f}  "
          f"(bull regime effect, equal-weight)")
    print(f"    libro_x_bull   = {coef['libro_x_bull']:+.5f}  p={pvals['libro_x_bull']:.3f}  "
          f"← DiD coefficient (LIBRO incremental lift in bull)")
    print()
    sig = "statistically significant" if pvals["libro_x_bull"] < 0.10 else "not significant at 10%"
    print(f"  DiD coefficient is {sig}.")
    print(f"  LIBRO adds {coef['libro_x_bull']*100:+.3f}% per week vs equal-weight during bull regime.")

    did_result = dict(
        const=round(float(coef["const"]), 6),
        libro=round(float(coef["libro"]), 6),
        bull=round(float(coef["bull"]), 6),
        libro_x_bull=round(float(coef["libro_x_bull"]), 6),
        p_libro=round(float(pvals["libro"]), 4),
        p_bull=round(float(pvals["bull"]), 4),
        p_interaction=round(float(pvals["libro_x_bull"]), 4),
        r_squared=round(float(ols.rsquared), 4),
        n_obs=int(ols.nobs),
    )
    return did_result


# ── Part 4: Signal-scaled sizing ──────────────────────────────────────────────
def part_4(prices, volumes, baseline_d):
    """
    Scale position size by normalised composite score magnitude.
    High-conviction picks get more margin; low-conviction get less.
    """
    print("\n" + "=" * 72)
    print("PART 4 — Signal-scaled sizing (position size ∝ composite z-score)")
    print("=" * 72)
    print(f"  Baseline d={baseline_d:.2f}, score-scaled ±50%\n")

    # Run standard backtest and signal-scaled variant
    fms_base, comb_base = fold_metrics(prices, volumes, deploy=baseline_d)
    warn_b, liq_b       = margin_events(prices, volumes, deploy=baseline_d)

    # Signal-scaled: weight per position = baseline_d × (0.5 + 0.5 × rank_norm)
    # (top-ranked gets 1.0×, lower-ranked in top-2 gets 0.5×)
    n = len(prices)
    asset_rets = prices.pct_change()
    sig        = composite_signal(prices, volumes)
    in_regime  = regime_series(prices)

    raw_w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i in range(n):
        if not in_regime.iloc[i]:
            continue
        row    = sig.iloc[i]
        ranked = row.rank(ascending=False, na_option="bottom")
        top    = sorted([a for a in prices.columns if ranked.get(a, 999) <= TOP_N],
                        key=lambda a: ranked.get(a, 999))
        if not top:
            continue
        # Top asset gets 1.0×, second gets 0.5× (if two assets)
        multipliers = [1.0, 0.5] if len(top) >= 2 else [1.0]
        for a, mult in zip(top, multipliers):
            raw_w.at[prices.index[i], a] = baseline_d * mult

    # Weekly rebalance carry-forward
    rebal_w = raw_w.copy()
    for i in range(1, len(rebal_w)):
        if i % REBAL_BARS != 0:
            rebal_w.iloc[i] = rebal_w.iloc[i - 1]

    for i in range(len(rebal_w)):
        if not in_regime.iloc[i]:
            rebal_w.iloc[i] = 0.0

    weights   = rebal_w.shift(1)
    gross_ret = (weights * LEVERAGE * asset_rets).sum(axis=1)
    weight_chg= weights.diff().abs().sum(axis=1)
    cost      = weight_chg * FEE_RT
    net_ret   = gross_ret - cost

    start = max(LOOKBACK, REGIME_MA) + 1
    net_ret  = net_ret.iloc[start:]
    weights2 = weights.iloc[start:]

    # Walk-forward on signal-scaled
    n2       = len(prices)
    fold_sz  = n2 // N_FOLDS
    fold_rets = []
    for k in range(N_FOLDS):
        fp = prices.iloc[k * fold_sz: (k + 1) * fold_sz]
        fv = volumes.iloc[k * fold_sz: (k + 1) * fold_sz]
        sub_n = len(fp)
        sub_rets_sig = []
        sub_asset_rets = fp.pct_change()
        sub_sig = composite_signal(fp, fv)
        sub_regime = regime_series(fp)
        sub_rw = pd.DataFrame(0.0, index=fp.index, columns=fp.columns)
        for i in range(sub_n):
            if not sub_regime.iloc[i]:
                continue
            row    = sub_sig.iloc[i]
            ranked = row.rank(ascending=False, na_option="bottom")
            top    = sorted([a for a in fp.columns if ranked.get(a, 999) <= TOP_N],
                            key=lambda a: ranked.get(a, 999))
            if not top:
                continue
            mults = [1.0, 0.5] if len(top) >= 2 else [1.0]
            for a, mult in zip(top, mults):
                sub_rw.at[fp.index[i], a] = baseline_d * mult
        sub_rebal = sub_rw.copy()
        for i in range(1, sub_n):
            if i % REBAL_BARS != 0:
                sub_rebal.iloc[i] = sub_rebal.iloc[i - 1]
        for i in range(sub_n):
            if not sub_regime.iloc[i]:
                sub_rebal.iloc[i] = 0.0
        sub_w = sub_rebal.shift(1)
        sub_gr = (sub_w * LEVERAGE * sub_asset_rets).sum(axis=1)
        sub_wc = sub_w.diff().abs().sum(axis=1)
        sub_c  = sub_wc * FEE_RT
        sub_nr = sub_gr - sub_c
        sub_trim = max(LOOKBACK, REGIME_MA) + 1
        fold_rets.append(sub_nr.iloc[sub_trim:])

    combined_sig = pd.concat(fold_rets)
    comb_sig     = metrics(combined_sig)
    fms_sig      = [metrics(r) for r in fold_rets]

    print(f"  {'Method':<22}  {'Fold1':>7} {'Fold2':>7} {'Fold3':>7}  "
          f"{'OOS Sh':>7} {'CAGR':>8} {'MaxDD':>7}")
    print("  " + "─" * 66)
    for label, fms, comb in [
        ("Equal-weight",      fms_base, comb_base),
        ("Signal-scaled 1/0.5", fms_sig, comb_sig),
    ]:
        print(f"  {label:<22}  "
              f"{fms[0]['sharpe']:+.3f}  {fms[1]['sharpe']:+.3f}  {fms[2]['sharpe']:+.3f}  "
              f"{comb['sharpe']:+.4f}  {comb['cagr_pct']:+6.1f}%  "
              f"{comb['max_dd_pct']:6.1f}%")

    return dict(
        equal=dict(fold_sharpes=[m["sharpe"] for m in fms_base], **comb_base),
        sig_scaled=dict(fold_sharpes=[m["sharpe"] for m in fms_sig], **comb_sig),
    )


# ── Part 5: Final recommendation ──────────────────────────────────────────────
def part_5(best_2a, best_2b, res_2c, did, res_4):
    print("\n" + "=" * 72)
    print("PART 5 — FINAL RECOMMENDATION")
    print("=" * 72)

    print("""
  SYNTHESIS
  ─────────
  Three orthogonal axes were tested on the 14-asset composite-signal strategy
  at 4.3× leverage (Coinbase Derivatives, 0.20% RT fee):

    2A. Fixed equal-weight grid  — which per-position margin fraction is safe?
    2B. Vol-scaled weighting     — does inverse-vol allocation improve Sharpe?
    2C. LIBRO margin-buffer      — does dynamic draw-down scaling add value?
    3.  DiD regression           — is the LIBRO benefit regime-conditional?
    4.  Signal-scaled sizing     — does concentrating into top pick help?
  """)

    # Assemble recommendation
    best_d = best_2a["d"] if best_2a else 0.20

    print(f"  RECOMMENDED CONFIG")
    print(f"  ─────────────────")
    print(f"  deploy_per_pos  : {best_d:.2f}  "
          f"({best_d*100:.0f}% of equity per position as margin)")
    print(f"  total_deployed  : {best_d*TOP_N:.2f}  "
          f"({best_d*TOP_N*100:.0f}% of equity posted as margin)")
    print(f"  effective_notional: {best_d*TOP_N*LEVERAGE:.2f}×  "
          f"(4.3× leverage on {best_d*TOP_N*100:.0f}%)")
    print(f"  leverage        : {LEVERAGE}×")
    print(f"  fee_model       : {FEE_RT*100:.2f}% RT taker")
    print()

    warn_thresh = 0.60 * best_d * TOP_N * 100
    liq_thresh  = 1.00 * best_d * TOP_N * 100
    print(f"  RISK THRESHOLDS (at d={best_d:.2f}, 2 positions)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Margin warning  : equity drops {warn_thresh:.1f}% within rebal week")
    print(f"  Liquidation     : equity drops {liq_thresh:.1f}% within rebal week")
    print(f"  (Based on 0.60/L and 1.00/L thresholds at 4.3× leverage)")
    print()

    if did:
        p = did["p_interaction"]
        coef = did["libro_x_bull"] * 100
        sig_txt = "significant" if p < 0.10 else "not significant"
        print(f"  DiD VERDICT: LIBRO adds {coef:+.3f}%/week in bull vs equal-weight "
              f"({sig_txt}, p={p:.3f})")
        if p >= 0.10:
            print(f"  → LIBRO complexity NOT warranted. Equal-weight is simpler and comparable.")
        else:
            print(f"  → LIBRO provides a statistically meaningful lift in bull regime.")
    print()

    print(f"  CONFIG.YAML CHANGES NEEDED")
    print(f"  ──────────────────────────")
    print(f"  max_position_pct: {best_d:.2f}    # per-position margin fraction")
    print(f"  leverage:         {LEVERAGE}      # 4.3× — NOT YET LIVE, confirm liq risk first")
    print(f"  fee_rate:         {FEE_RT:.4f}   # 0.20% RT taker (already set)")
    print()
    print(f"  CONSTRAINTS MET?")
    if best_2a:
        print(f"  P(liq event)   = {best_2a['liq_pct']:.2f}% < 1.0%  ✓")
        print(f"  All folds +Sh  = {best_2a['folds_all_pos']}  ✓")
        print(f"  P(warn/week)   = {best_2a['warn_pct']:.1f}% < 10%  "
              f"{'✓' if best_2a['warn_pct'] < 10 else '✗'}")
    print()
    print("  NEXT STEP: Confirm leverage implementation + liquidation risk")
    print("  discussion before writing leverage into engine.py.\n")

    return dict(recommended_d=best_d, leverage=LEVERAGE, fee_rt=FEE_RT,
                warn_equity_drop_pct=warn_thresh, liq_equity_drop_pct=liq_thresh)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("CAPPUCCINO POSITION SIZING STUDY")
    print(f"Universe: {len(ASSETS)} assets  |  Signal: ret_24h + vol_accel  |  "
          f"Leverage: {LEVERAGE}×  |  Fee: {FEE_RT*100:.2f}% RT")
    print("=" * 72)

    print("\nLoading 14-asset OHLCV data...")
    prices, volumes = load_all()
    print(f"  {len(prices)} bars × {len(prices.columns)} assets  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    # Run all parts
    res_2a, best_2a = part_2a(prices, volumes)
    res_2b, best_2b = part_2b(prices, volumes)

    baseline_d = best_2a["d"] if best_2a else 0.20
    res_2c      = part_2c(prices, volumes, baseline_d)
    did         = part_3(prices, volumes, baseline_d)
    res_4       = part_4(prices, volumes, baseline_d)
    rec         = part_5(best_2a, best_2b, res_2c, did, res_4)

    # Save results
    output = dict(
        config=dict(assets=ASSETS, leverage=LEVERAGE, fee_rt=FEE_RT,
                    n_folds=N_FOLDS, top_n=TOP_N),
        part_2a=res_2a,
        part_2b=res_2b,
        part_2c=res_2c,
        part_3_did=did,
        part_4=res_4,
        recommendation=rec,
    )
    out_path = RESULTS_DIR / "sizing_study.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    # Save DiD text report
    if did:
        did_txt = RESULTS_DIR / "libro_did.txt"
        with open(did_txt, "w") as f:
            f.write("LIBRO DiD Regression Results\n")
            f.write("=" * 40 + "\n")
            for k, v in did.items():
                f.write(f"  {k:<20} {v}\n")
        print(f"  DiD report   → {did_txt}")


if __name__ == "__main__":
    main()
