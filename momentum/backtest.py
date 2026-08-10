"""
Cross-Sectional Momentum Backtest — v3
Universe:  14 assets (matches live engine config.yaml)
Signal:    Equal-weight z-score composite of ret_24h + vol_accel
             ret_24h  : 24h price return (cross-sectional momentum)
             vol_accel: current 24h rolling volume / prior 24h rolling volume
           Top-2 by composite score, equal weight (25% each, 50% max)
Rebal:     Every 168 bars (weekly)
Regime:    BTC 168h MA with ±0.5% hysteresis (matches engine exactly)
Costs:     0.20% round-trip taker (Coinbase Derivatives baseline)
Parity:    Signal logic mirrors engine.py::compute_target_weights() exactly.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path.home() / "cappuccino" / "data"
OUT_DIR    = Path.home() / "cappuccino" / "momentum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX",
    "LINK", "LTC", "BCH", "XLM", "SUI", "DOT", "HBAR",
]
LOOKBACK_BARS = 24        # 24h momentum and volume window
TOP_N         = 2         # long top-2 assets
MAX_WEIGHT    = 0.25      # 25% per asset → 50% max deployed
FEE_RT        = 0.0020    # 0.20% RT taker (Coinbase Derivatives)
REBAL_BARS    = 168       # weekly rebalance cadence
REGIME_MA     = 168       # BTC 168h MA — matches engine regime_ma_bars
REGIME_BUFFER = 0.005     # ±0.5% hysteresis band — matches engine regime_buffer
SHARPE_MIN    = 0.4       # OOS pass gate
N_FOLDS       = 3
HOURS_PER_YR  = 8760

# Fee schedules for platform comparison
FEE_SCHEDULES = {
    "Coinbase Futures taker RT (live)": 0.0020,   # 0.10% × 2
    "Coinbase Futures maker RT":        0.0019,   # 0.095% × 2
    "Baseline (0.12% RT)":              0.0012,
    "Kraken spot taker RT":             0.0052,   # 0.26% × 2
}


# ── Data ───────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (prices, volumes) DataFrames aligned on a common datetime index.
    Drops any bar where any asset has a NaN close or volume (forward-fill first
    to handle isolated missing bars, then drop if still NaN).
    """
    closes  = {}
    volumes = {}
    for asset in ASSETS:
        path = DATA_DIR / f"{asset}_1h.parquet"
        df   = pd.read_parquet(path)
        df   = df.set_index("datetime")
        df.index = pd.to_datetime(df.index, utc=True)
        closes[asset]  = df["close"].ffill()
        volumes[asset] = df["volume"].ffill()

    prices = pd.DataFrame(closes).sort_index()
    vols   = pd.DataFrame(volumes).sort_index()

    # Align to common index and drop rows still missing after ffill
    prices, vols = prices.align(vols, join="inner", axis=0)
    prices = prices.dropna()
    vols   = vols.loc[prices.index].fillna(0.0)
    return prices, vols


# ── Composite signal (vectorised — exact parity with engine.py) ─────────────
def _zscore_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise z-score across asset columns. Returns same shape."""
    mu  = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.subtract(mu, axis=0).divide(std, axis=0).fillna(0.0)


def compute_signal(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised composite signal.
    Factor 1 — ret_24h  : rolling 24h price return
    Factor 2 — vol_accel: rolling 24h volume sum / shifted-by-24h rolling 24h volume sum

    Each factor z-scored cross-sectionally (row-wise), averaged equally.
    Returns composite DataFrame, same shape as prices.
    """
    # Factor 1: 24h return
    ret_24h = prices.pct_change(LOOKBACK_BARS)

    # Factor 2: volume acceleration (matches engine formula exactly)
    vol_now  = volumes.rolling(LOOKBACK_BARS).sum()
    vol_prev = volumes.rolling(LOOKBACK_BARS).sum().shift(LOOKBACK_BARS)
    vol_accel = vol_now.divide(vol_prev.replace(0, np.nan))
    vol_accel.replace([np.inf, -np.inf], np.nan, inplace=True)

    z_ret    = _zscore_panel(ret_24h)
    z_vaccel = _zscore_panel(vol_accel)

    # Average the two z-scores — NaN in one factor still uses the other (fill 0)
    composite = (z_ret.fillna(0) + z_vaccel.fillna(0)) / 2.0

    # Zero out warm-up rows where ret_24h is still NaN
    warm_mask = ret_24h.isna().any(axis=1)
    composite.loc[warm_mask] = np.nan

    return composite


# ── Core backtest ─────────────────────────────────────────────────────────────
def backtest(
    prices:    pd.DataFrame,
    volumes:   pd.DataFrame,
    use_regime: bool = True,
    fee_rt:    float = FEE_RT,
) -> dict:
    """
    No lookahead guarantee:
      - Composite signal at bar t uses data up to t (no future data)
      - Weights determined at t → shifted +1 → realised return on bar t+1
      - Regime MA at t uses prices[t-REGIME_MA : t]

    Regime hysteresis:
      BEAR → BULL: BTC must close > MA × (1 + REGIME_BUFFER)
      BULL → BEAR: BTC must close < MA × (1 - REGIME_BUFFER)
    """
    asset_rets = prices.pct_change()

    # ── Composite signal & ranks ──────────────────────────────────────────────
    composite = compute_signal(prices, volumes)
    # Rank descending: rank 1 = highest composite score
    ranks = composite.rank(axis=1, ascending=False, method="first")

    # ── Equal-weight target for top-N (no inv_vol scaling — matches engine) ──
    raw_w = pd.DataFrame(
        np.where(ranks.values <= TOP_N, MAX_WEIGHT, 0.0),
        index=ranks.index,
        columns=ranks.columns,
    )
    # Zero out warm-up bars
    raw_w[composite.isna().any(axis=1)] = 0.0

    # ── Weekly rebalancing: carry weights forward between rebal bars ──────────
    rebal_idx = np.arange(0, len(raw_w), REBAL_BARS)
    rebal_w   = pd.DataFrame(0.0, index=raw_w.index, columns=raw_w.columns)
    for i, start in enumerate(rebal_idx):
        end = rebal_idx[i + 1] if i + 1 < len(rebal_idx) else len(raw_w)
        rebal_w.iloc[start:end] = raw_w.iloc[start].values

    # ── BTC regime filter with hysteresis ────────────────────────────────────
    btc_ma = prices["BTC"].rolling(REGIME_MA).mean()
    if use_regime:
        # Vectorised hysteresis via forward-scan (must stay sequential)
        in_regime_arr = np.ones(len(prices), dtype=bool)
        state = True    # start in-regime
        btc_arr = prices["BTC"].values
        ma_arr  = btc_ma.values
        buf     = REGIME_BUFFER
        for i in range(len(prices)):
            if np.isnan(ma_arr[i]):
                in_regime_arr[i] = True
                continue
            if state:
                if btc_arr[i] < ma_arr[i] * (1 - buf):
                    state = False
            else:
                if btc_arr[i] > ma_arr[i] * (1 + buf):
                    state = True
            in_regime_arr[i] = state

        in_regime = pd.Series(in_regime_arr, index=prices.index)
        # Zero out bear-regime weights
        rebal_w[~in_regime] = 0.0
    else:
        in_regime = pd.Series(True, index=prices.index)

    # ── Execution: weights shift +1 (fill at next bar open) ──────────────────
    weights = rebal_w.shift(1)

    gross_ret  = (weights * asset_rets).sum(axis=1)
    weight_chg = weights.diff().abs().sum(axis=1)
    cost       = weight_chg * fee_rt
    net_ret    = gross_ret - cost

    # Trim warm-up (need REGIME_MA bars for MA + LOOKBACK_BARS for signal)
    start = max(LOOKBACK_BARS * 2, REGIME_MA) + 1
    return {
        "net_ret"    : net_ret.iloc[start:],
        "gross_ret"  : gross_ret.iloc[start:],
        "weights"    : weights.iloc[start:],
        "weight_chg" : weight_chg.iloc[start:],
        "cost"       : cost.iloc[start:],
        "btc_price"  : prices["BTC"].iloc[start:],
        "btc_ma"     : btc_ma.iloc[start:] if use_regime else None,
        "composite"  : composite.iloc[start:],
    }


# ── Metrics ────────────────────────────────────────────────────────────────────
def metrics(r: pd.Series, label: str = "") -> dict:
    ann  = np.sqrt(HOURS_PER_YR)
    cum  = (1 + r).cumprod()
    yrs  = len(r) / HOURS_PER_YR
    cagr = (cum.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    sh   = r.mean() / r.std() * ann if r.std() > 0 else 0
    dd   = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    down = r[r < 0]
    so   = r.mean() / down.std() * ann if len(down) > 0 and down.std() > 0 else 0
    calmar = cagr / abs(dd) if dd != 0 else 0
    return dict(
        label=label, bars=len(r), years=round(yrs, 2),
        sharpe=round(sh, 3), sortino=round(so, 3), calmar=round(calmar, 3),
        cagr_pct=round(cagr, 2), max_dd_pct=round(dd, 2),
        total_ret_pct=round((cum.iloc[-1] - 1) * 100, 2),
    )


def extra_metrics(res: dict) -> dict:
    w  = res["weights"]
    wc = res["weight_chg"]
    invested     = (w.sum(axis=1) > 0).mean() * 100
    turnover_day = wc.mean() * 24
    fee_drag_yr  = wc.mean() * HOURS_PER_YR * FEE_RT * 100

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

    # Regime time
    if res["btc_ma"] is not None:
        regime_pct = (res["btc_price"] >= res["btc_ma"]).mean() * 100
    else:
        regime_pct = 100.0

    return dict(
        pct_time_invested=round(invested, 1),
        turnover_per_day=round(turnover_day, 4),
        fee_drag_pct_yr=round(fee_drag_yr, 2),
        avg_holding_hrs=avg_hold_hrs,
        pct_time_in_regime=round(regime_pct, 1),
    )


# ── Walk-forward ───────────────────────────────────────────────────────────────
def walk_forward(
    prices: pd.DataFrame, volumes: pd.DataFrame
) -> tuple[list[dict], dict]:
    n         = len(prices)
    fold_size = n // N_FOLDS
    fold_ms   = []
    oos_rets  = []

    print(f"Data:  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"Bars:  {n}  |  Fold: ~{fold_size} bars ({fold_size/HOURS_PER_YR:.2f} yrs each)\n")

    for i in range(N_FOLDS):
        fp = prices.iloc[i * fold_size : (i + 1) * fold_size]
        fv = volumes.iloc[i * fold_size : (i + 1) * fold_size]
        res = backtest(fp, fv)
        m   = metrics(res["net_ret"], f"Fold {i+1}")
        fold_ms.append(m)
        oos_rets.append(res["net_ret"])
        print(f"  Fold {i+1}  {fp.index[0].date()} → {fp.index[-1].date()}"
              f"  Sharpe={m['sharpe']:+.3f}  CAGR={m['cagr_pct']:+.1f}%"
              f"  MaxDD={m['max_dd_pct']:.1f}%  Calmar={m['calmar']:.2f}")

    combined = pd.concat(oos_rets)
    cm       = metrics(combined, "Combined OOS")
    return fold_ms, cm


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot(prices: pd.DataFrame, volumes: pd.DataFrame):
    res_regime   = backtest(prices, volumes, use_regime=True)
    res_noregime = backtest(prices, volumes, use_regime=False)

    nr   = res_regime["net_ret"]
    nr2  = res_noregime["net_ret"]
    cum  = (1 + nr).cumprod()
    cum2 = (1 + nr2).cumprod()
    dd   = ((cum - cum.cummax()) / cum.cummax()) * 100

    # Fold shading
    n         = len(prices)
    fold_size = n // N_FOLDS
    fold_spans = []
    for i in range(N_FOLDS):
        fp  = prices.iloc[i * fold_size : (i + 1) * fold_size]
        fv  = volumes.iloc[i * fold_size : (i + 1) * fold_size]
        res = backtest(fp, fv)
        idx = res["net_ret"].index
        if len(idx):
            fold_spans.append((idx[0], idx[-1]))

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    shade  = ["#e3f2fd", "#f3e5f5", "#e8f5e9"]
    for i, (s, e) in enumerate(fold_spans):
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(s, e, alpha=0.18, color=shade[i])
        ax1.axvline(s, color="#bdbdbd", lw=0.5, ls="--")

    ax1.plot(cum.index,  cum.values,  color="#1565c0", lw=1.3,
             label="With regime filter (168h MA)")
    ax1.plot(cum2.index, cum2.values, color="#90a4ae", lw=0.9, ls="--",
             label="No regime filter", alpha=0.7)
    ax1.axhline(1.0, color="#9e9e9e", lw=0.5, ls=":")
    ax1.set_ylabel("Cumulative Return (x)", fontsize=10)
    ax1.set_title(
        "Cross-Sectional Momentum v3  |  14-asset universe  |  "
        "ret_24h + vol_accel composite  |  Weekly rebal  |  BTC 168h MA regime",
        fontsize=11, pad=8,
    )
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

    out = OUT_DIR / "backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved to {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> bool:
    print("=" * 72)
    print("CROSS-SECTIONAL MOMENTUM BACKTEST  v3")
    print("Signal: ret_24h + vol_accel (z-score composite)  |  14-asset universe")
    print("=" * 72)

    prices, volumes = load_data()
    print(f"Loaded {len(prices)} bars x {len(prices.columns)} assets\n")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    print("WALK-FORWARD (3 sequential folds, no random splits)")
    print("-" * 72)
    fold_ms, combined = walk_forward(prices, volumes)

    # ── Full-sample ───────────────────────────────────────────────────────────
    res_full = backtest(prices, volumes)
    full_m   = metrics(res_full["net_ret"], "Full sample")
    extra    = extra_metrics(res_full)

    print("\n" + "=" * 72)
    print("FULL-SAMPLE METRICS")
    print("=" * 72)
    for k, v in {**full_m, **extra}.items():
        if k == "label":
            continue
        print(f"  {k:<32} {v}")

    # No-regime comparison
    res_raw = backtest(prices, volumes, use_regime=False)
    raw_m   = metrics(res_raw["net_ret"], "No regime filter")
    print(f"\n  {'--- without regime filter ---'}")
    print(f"  {'sharpe (no filter)':<32} {raw_m['sharpe']}")
    print(f"  {'cagr_pct (no filter)':<32} {raw_m['cagr_pct']}")
    print(f"  {'max_dd_pct (no filter)':<32} {raw_m['max_dd_pct']}")

    # ── Platform fee comparison ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("PLATFORM FEE COMPARISON  (168h MA regime, weekly rebal, composite signal)")
    print(f"{'='*72}")
    print(f"  {'Platform':<36} {'OOS Sharpe':>10} {'OOS CAGR':>10} {'Max DD':>9}  Verdict")
    print("  " + "-" * 70)
    for label, fr in FEE_SCHEDULES.items():
        fold_rets = []
        n  = len(prices)
        fs = n // N_FOLDS
        for i in range(N_FOLDS):
            fp = prices.iloc[i * fs : (i + 1) * fs]
            fv = volumes.iloc[i * fs : (i + 1) * fs]
            r  = backtest(fp, fv, fee_rt=fr)
            fold_rets.append(r["net_ret"])
        combined_r = pd.concat(fold_rets)
        m = metrics(combined_r)
        verdict = "PASS" if m["sharpe"] >= SHARPE_MIN else "FAIL"
        print(f"  {label:<36} {m['sharpe']:>+10.3f} {m['cagr_pct']:>+9.1f}%"
              f" {m['max_dd_pct']:>8.1f}%  {verdict}")

    # ── Walk-forward summary ──────────────────────────────────────────────────
    print(f"\n{'-'*72}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'-'*72}")
    for m in fold_ms:
        print(f"  {m['label']}: Sharpe={m['sharpe']:+.3f}  CAGR={m['cagr_pct']:+.1f}%"
              f"  MaxDD={m['max_dd_pct']:.1f}%  ({m['years']:.2f} yrs)")
    print(f"  Combined:  Sharpe={combined['sharpe']:+.3f}  CAGR={combined['cagr_pct']:+.1f}%"
          f"  MaxDD={combined['max_dd_pct']:.1f}%")

    print("\n" + "=" * 72)
    passed = combined["sharpe"] >= SHARPE_MIN
    if not passed:
        print(f"HARD STOP: Combined OOS Sharpe {combined['sharpe']:.3f} < {SHARPE_MIN}")
        print("DO NOT proceed to live trading.")
    else:
        print(f"PASS: Combined OOS Sharpe {combined['sharpe']:.3f} >= {SHARPE_MIN}")

    plot(prices, volumes)

    result = dict(
        version="v3",
        signal="ret_24h + vol_accel (z-score composite, equal weight)",
        universe=ASSETS,
        fold_metrics=fold_ms,
        combined_oos=combined,
        full_sample={**full_m, **extra},
        no_regime=raw_m,
        passed=passed,
    )
    out = OUT_DIR / "backtest_metrics.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Metrics saved to {out}")
    return passed


if __name__ == "__main__":
    main()
