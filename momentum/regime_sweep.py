"""
Regime MA Period Sweep
Tests BTC moving-average lengths from 2 days (48h) to 200 days (4800h).
Finds the MA period that maximises OOS Sharpe for the weekly-rebalance
cross-sectional momentum strategy.

Usage:
    python regime_sweep.py
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_DIR      = Path.home() / "cappuccino" / "data"
ASSETS        = ["BTC", "ETH", "DOGE", "ADA", "SOL"]
LOOKBACK_BARS = 24
TOP_N         = 2
MAX_WEIGHT    = 0.25
FEE_RT        = 0.0012   # 0.12% round-trip (matches backtest.py)
REBAL_BARS    = 168      # weekly
HOURS_PER_YR  = 8760
N_FOLDS       = 3

# MA periods to test (in hours)
# 48h=2d, 100h=4d, 168h=7d, 200h=8.3d, 336h=14d, 504h=21d,
# 720h=30d, 1200h=50d, 2400h=100d, 4800h=200d
MA_PERIODS = [48, 100, 168, 200, 336, 504, 720, 1200, 2400, 4800]


def load_prices() -> pd.DataFrame:
    frames = {}
    for asset in ASSETS:
        df = pd.read_parquet(DATA_DIR / f"{asset}_1h.parquet")
        s  = df.set_index("datetime")["close"]
        s.index = pd.to_datetime(s.index, utc=True)
        frames[asset] = s
    return pd.DataFrame(frames).sort_index().dropna()


def backtest_with_ma(prices: pd.DataFrame, regime_ma: int) -> dict:
    asset_rets = prices.pct_change()

    signal = prices.pct_change(LOOKBACK_BARS)
    ranks  = signal.rank(axis=1, ascending=False, method="first")
    raw_w  = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    raw_w[ranks <= TOP_N] = MAX_WEIGHT

    rebal_w = raw_w.copy()
    for i in range(1, len(rebal_w)):
        if i % REBAL_BARS != 0:
            rebal_w.iloc[i] = rebal_w.iloc[i - 1]

    btc_ma    = prices["BTC"].rolling(regime_ma).mean()
    in_regime = prices["BTC"] >= btc_ma
    for i in range(len(rebal_w)):
        if not in_regime.iloc[i]:
            rebal_w.iloc[i] = 0.0

    weights  = rebal_w.shift(1)
    gross_r  = (weights * asset_rets).sum(axis=1)
    cost     = weights.diff().abs().sum(axis=1) * FEE_RT
    net_r    = gross_r - cost

    start = max(LOOKBACK_BARS, regime_ma) + 1
    net_r = net_r.iloc[start:]
    w_trim = weights.iloc[start:]
    btc_trim = prices["BTC"].iloc[start:]
    ma_trim  = btc_ma.iloc[start:]

    return {
        "net_ret": net_r,
        "weights": w_trim,
        "btc_price": btc_trim,
        "btc_ma": ma_trim,
    }


def calc_metrics(r: pd.Series) -> dict:
    if len(r) < 10 or r.std() == 0:
        return dict(sharpe=0, cagr_pct=0, max_dd_pct=0)
    ann  = np.sqrt(HOURS_PER_YR)
    cum  = (1 + r).cumprod()
    yrs  = len(r) / HOURS_PER_YR
    cagr = (cum.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    sh   = r.mean() / r.std() * ann
    dd   = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    return dict(sharpe=round(sh, 3), cagr_pct=round(cagr, 2), max_dd_pct=round(dd, 2))


def oos_sweep(prices: pd.DataFrame) -> list[dict]:
    n         = len(prices)
    fold_size = n // N_FOLDS
    results   = []

    for ma in MA_PERIODS:
        fold_rets = []
        fold_sharpes = []
        for i in range(N_FOLDS):
            fp  = prices.iloc[i * fold_size : (i + 1) * fold_size]
            res = backtest_with_ma(fp, ma)
            if len(res["net_ret"]) > 10:
                fold_rets.append(res["net_ret"])
                m = calc_metrics(res["net_ret"])
                fold_sharpes.append(m["sharpe"])

        if not fold_rets:
            continue

        combined = pd.concat(fold_rets)
        cm = calc_metrics(combined)

        # Full-sample stats for supplementary info
        full_res = backtest_with_ma(prices, ma)
        nr_full  = full_res["net_ret"]
        invested = (full_res["weights"].sum(axis=1) > 0).mean() * 100
        in_bull  = (full_res["btc_price"] >= full_res["btc_ma"]).mean() * 100

        results.append({
            "ma_hours": ma,
            "ma_label": f"{ma}h ({ma/24:.0f}d)",
            "oos_sharpe": cm["sharpe"],
            "oos_cagr_pct": cm["cagr_pct"],
            "oos_max_dd_pct": cm["max_dd_pct"],
            "fold_sharpes": [round(s, 3) for s in fold_sharpes],
            "pct_time_invested": round(invested, 1),
            "pct_time_in_bull": round(in_bull, 1),
        })

    return results


def main():
    print("=" * 70)
    print("REGIME MA PERIOD SWEEP  |  Weekly rebal  |  Top-2 by 24h rank")
    print(f"Fee: {FEE_RT*100:.3f}% RT  |  Assets: {', '.join(ASSETS)}")
    print("=" * 70)

    prices = load_prices()
    print(f"Data: {prices.index[0].date()} → {prices.index[-1].date()}  "
          f"({len(prices):,} bars)\n")

    print("Running sweep...")
    results = oos_sweep(prices)

    # Sort by OOS Sharpe
    results.sort(key=lambda x: x["oos_sharpe"], reverse=True)

    print(f"\n{'MA Period':<16} {'OOS Sharpe':>10} {'OOS CAGR':>10} {'Max DD':>9}"
          f" {'Invested%':>10} {'Bull%':>8}  Fold Sharpes")
    print("─" * 90)
    for r in results:
        fold_str = "  ".join(f"{s:+.3f}" for s in r["fold_sharpes"])
        mark = " ◄ BEST" if r == results[0] else ""
        print(f"{r['ma_label']:<16} {r['oos_sharpe']:>+10.3f} {r['oos_cagr_pct']:>+9.1f}%"
              f" {r['oos_max_dd_pct']:>8.1f}%"
              f" {r['pct_time_invested']:>9.1f}%"
              f" {r['pct_time_in_bull']:>7.1f}%  [{fold_str}]{mark}")

    best = results[0]
    print(f"\n{'='*70}")
    print(f"BEST: {best['ma_label']}  →  OOS Sharpe {best['oos_sharpe']:+.3f}"
          f"  CAGR {best['oos_cagr_pct']:+.1f}%  MaxDD {best['oos_max_dd_pct']:.1f}%")
    print(f"      {best['pct_time_invested']:.1f}% time invested  |  "
          f"{best['pct_time_in_bull']:.1f}% of dataset in bull regime")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
