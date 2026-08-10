"""
Feature Builder — XGBoost Momentum
Loads 28-month OHLCV + OKX pipeline features, builds labeled feature matrix.

Usage:
    python feature_builder.py            # build and print diagnostics
    from feature_builder import build_features
    X, y, meta = build_features()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR  = Path(__file__).parent
CFG_FILE  = BASE_DIR / "config.yaml"

# ── Config ─────────────────────────────────────────────────────────────────────
def load_cfg() -> dict:
    with open(CFG_FILE) as f:
        return yaml.safe_load(f)


# ── OHLCV loading + per-asset feature engineering ─────────────────────────────
def load_ohlcv(asset: str, ohlcv_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(ohlcv_dir / f"{asset}_1h.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def add_ohlcv_features(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Compute per-asset technical features from OHLCV.
    All features are lag-1 shifted to prevent lookahead.
    """
    c = df["close"]
    v = df["volume"]
    h = df["high"]
    lo = df["low"]

    out = pd.DataFrame(index=df.index)

    # Returns at multiple horizons
    for lag in [1, 2, 4, 8, 12, 24, 48, 168]:
        out[f"{asset}_ret_{lag}h"] = c.pct_change(lag)

    # Volatility
    for window in [8, 24, 168]:
        out[f"{asset}_vol_{window}h"] = c.pct_change().rolling(window).std()

    # RSI-14
    delta   = c.diff()
    gain    = delta.clip(lower=0).rolling(14).mean()
    loss    = (-delta.clip(upper=0)).rolling(14).mean()
    rs      = gain / loss.replace(0, np.nan)
    out[f"{asset}_rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12   = c.ewm(span=12, adjust=False).mean()
    ema26   = c.ewm(span=26, adjust=False).mean()
    macd    = ema12 - ema26
    signal  = macd.ewm(span=9, adjust=False).mean()
    out[f"{asset}_macd"]        = macd
    out[f"{asset}_macd_signal"] = signal
    out[f"{asset}_macd_hist"]   = macd - signal

    # Bollinger band position (price relative to 20h bands)
    ma20     = c.rolling(20).mean()
    std20    = c.rolling(20).std()
    out[f"{asset}_bb_pos"] = (c - ma20) / std20.replace(0, np.nan)

    # Volume features
    out[f"{asset}_vol_ratio_24h"] = v / v.rolling(24).mean().replace(0, np.nan)
    out[f"{asset}_vol_ratio_168h"] = v / v.rolling(168).mean().replace(0, np.nan)

    # High-low range normalised by close
    out[f"{asset}_hl_range"] = (h - lo) / c.replace(0, np.nan)

    # Distance from rolling highs/lows
    out[f"{asset}_dist_high_24h"] = c / h.rolling(24).max() - 1
    out[f"{asset}_dist_low_24h"]  = c / lo.rolling(24).min() - 1
    out[f"{asset}_dist_high_168h"] = c / h.rolling(168).max() - 1
    out[f"{asset}_dist_low_168h"]  = c / lo.rolling(168).min() - 1

    # Volume trend — slope of volume over last 12h (rising = positive conviction)
    out[f"{asset}_vol_trend_12h"] = v.rolling(12).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] / (x.mean() + 1e-9),
        raw=True,
    )

    # Return skewness — tail asymmetry over 24h and 168h windows
    rets = c.pct_change()
    out[f"{asset}_skew_24h"]  = rets.rolling(24).skew()
    out[f"{asset}_skew_168h"] = rets.rolling(168).skew()

    # VWAP deviation — price vs volume-weighted average price
    # Rolling 24h and 168h VWAP. Positive = price above average paid → stretched.
    typical = (h + lo + c) / 3
    vwap_24h  = (typical * v).rolling(24).sum()  / v.rolling(24).sum().replace(0, np.nan)
    vwap_168h = (typical * v).rolling(168).sum() / v.rolling(168).sum().replace(0, np.nan)
    out[f"{asset}_vwap_dev_24h"]  = (c - vwap_24h)  / vwap_24h.replace(0, np.nan)
    out[f"{asset}_vwap_dev_168h"] = (c - vwap_168h) / vwap_168h.replace(0, np.nan)

    # Ichimoku components (no lookahead — standard periods: 9/26/52)
    tenkan  = (h.rolling(9).max()  + lo.rolling(9).min())  / 2   # conversion line
    kijun   = (h.rolling(26).max() + lo.rolling(26).min()) / 2   # base line
    senkou_a = (tenkan + kijun) / 2                               # cloud A (unshifted)
    senkou_b = (h.rolling(52).max() + lo.rolling(52).min()) / 2  # cloud B (unshifted)
    # TK spread: positive = tenkan > kijun = bullish short-term momentum
    out[f"{asset}_ichi_tk_spread"]      = (tenkan - kijun) / c.replace(0, np.nan)
    # Price vs Kijun: distance from medium-term equilibrium
    out[f"{asset}_ichi_price_kijun"]    = (c - kijun) / c.replace(0, np.nan)
    # Cloud from 26 bars ago (available in real-time — the cloud leads by 26 bars
    # in standard Ichimoku, so the bar-26 cloud aligns with today's price)
    out[f"{asset}_ichi_cloud_thick"]    = (senkou_a - senkou_b).shift(26) / c.replace(0, np.nan)
    out[f"{asset}_ichi_price_vs_cloud"] = (c - senkou_a.shift(26))        / c.replace(0, np.nan)

    # Hurst exponent (rescaled range, rolling 168h)
    # H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random walk.
    # Tells the model whether current momentum is likely to persist.
    def _hurst_rs(closes: np.ndarray) -> float:
        if len(closes) < 20 or np.any(closes <= 0):
            return np.nan
        log_rets = np.diff(np.log(closes))
        s = log_rets.std()
        if s == 0:
            return np.nan
        dev = np.cumsum(log_rets - log_rets.mean())
        return np.log((dev.max() - dev.min()) / s) / np.log(len(log_rets))

    out[f"{asset}_hurst_168h"] = c.rolling(168).apply(_hurst_rs, raw=True)

    # Target: next-bar direction (1 if next close > current close, 0 otherwise)
    out[f"{asset}_target"] = (c.shift(-1) > c).astype(int)

    # Shift all features by 1 (no lookahead — features known at bar close)
    feature_cols = [col for col in out.columns if not col.endswith("_target")]
    out[feature_cols] = out[feature_cols].shift(1)

    return out


# ── Pipeline feature loading ───────────────────────────────────────────────────
def load_pipeline_features(
    feature_parquet_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    assets: list[str],
) -> pd.DataFrame:
    """
    Find the most recent export parquet covering the requested range.
    Returns a DataFrame indexed by UTC hourly timestamps.
    """
    pattern = f"features_*_{'_'.join(assets)}.parquet"
    candidates = list(feature_parquet_dir.glob(pattern))
    if not candidates:
        # Also try any parquet with matching asset subset
        candidates = list(feature_parquet_dir.glob("features_*.parquet"))
    if not candidates:
        print("  WARNING: no pipeline feature parquets found — skipping enrichment")
        return pd.DataFrame()

    # Pick most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    print(f"  Pipeline features: {chosen.name}")
    df = pd.read_parquet(chosen)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


# ── Macro / external features ─────────────────────────────────────────────────
def load_macro_features(
    start: pd.Timestamp,
    end: pd.Timestamp,
    hourly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Fetch DXY (US Dollar Index), VIX, and Crypto Fear & Greed Index.
    All are daily; forward-filled and aligned to the hourly OHLCV index.
    Shifted by 1 bar so daily values are known at bar close (no lookahead).
    Returns empty DataFrame silently if all sources fail.
    """
    out = pd.DataFrame(index=hourly_index)

    # ── DXY and VIX via yfinance ───────────────────────────────────────────────
    try:
        import yfinance as yf
        start_s = (start - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        end_s   = (end   + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        for ticker, col in [("DX-Y.NYB", "macro_dxy"), ("^VIX", "macro_vix")]:
            raw = yf.download(ticker, start=start_s, end=end_s,
                              interval="1d", progress=False, auto_adjust=True)
            if raw.empty:
                continue
            s = raw["Close"].squeeze()
            s.index = pd.to_datetime(s.index).tz_localize("UTC")
            # Reindex to hourly, forward-fill gaps (weekends, holidays)
            out[col] = s.reindex(hourly_index, method="ffill")

        # Derived: 5-day DXY return (BTC is inversely correlated with dollar momentum)
        if "macro_dxy" in out.columns:
            out["macro_dxy_ret_5d"] = out["macro_dxy"].pct_change(5 * 24)

        # VIX regime: high VIX (>25) = risk-off, correlations with TradFi rise
        if "macro_vix" in out.columns:
            out["macro_vix_high"] = (out["macro_vix"] >= 25).astype(int)

        print(f"  Macro (DXY/VIX): loaded {out.notna().any(axis=1).sum():,} non-null rows")

    except Exception as e:
        print(f"  WARNING: yfinance macro fetch failed ({e}) — skipping DXY/VIX")

    # ── Crypto Fear & Greed Index (alternative.me — free, no key) ─────────────
    try:
        import requests
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1000&format=json",
            timeout=15,
        )
        if resp.ok:
            data = resp.json().get("data", [])
            fg = pd.DataFrame([
                {
                    "ts":    pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC"),
                    "value": int(d["value"]),
                }
                for d in data
            ]).set_index("ts").sort_index()
            out["macro_fear_greed"] = fg["value"].reindex(hourly_index, method="ffill")
            # Normalise 0–100 → 0–1 for the model
            out["macro_fear_greed_norm"] = out["macro_fear_greed"] / 100.0
            print(f"  Macro (Fear & Greed): {out['macro_fear_greed'].notna().sum():,} rows loaded")
    except Exception as e:
        print(f"  WARNING: Fear & Greed fetch failed ({e}) — skipping")

    # Shift by 1: daily values from yesterday are known at today's bar close
    return out.shift(1)


# ── Cross-asset features ───────────────────────────────────────────────────────
def add_cross_asset_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum strategy-aware cross-asset features.
    Uses BTC as the regime anchor.
    """
    out = pd.DataFrame(index=prices.index)

    # ── Calendar features (sin/cos encoded for cyclicality) ───────────────────
    # All timestamps are UTC. Crypto has strong hour-of-day and day-of-week
    # patterns: funding resets at 00/08/16 UTC, US open ~14:00 UTC, weekend dips.
    idx = prices.index
    hour = idx.hour
    dow  = idx.dayofweek  # 0=Monday … 6=Sunday
    out["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["cal_dow_sin"]  = np.sin(2 * np.pi * dow / 7)
    out["cal_dow_cos"]  = np.cos(2 * np.pi * dow / 7)
    # Binary flags for high-activity periods
    out["cal_us_session"]     = ((hour >= 13) & (hour <= 21)).astype(int)  # 13–21 UTC
    out["cal_asia_session"]   = ((hour >= 0)  & (hour <= 8)).astype(int)   # 00–08 UTC
    out["cal_funding_reset"]  = (hour % 8 == 0).astype(int)               # 00/08/16 UTC
    out["cal_weekend"]        = (dow >= 5).astype(int)                     # Sat/Sun

    # ── BTC trend strength: ADX-14 ────────────────────────────────────────────
    # ADX measures trend STRENGTH (not direction) — 0 = choppy, 100 = strong trend.
    # Computed from BTC OHLCV. No directional bias, no cold-start at 28-month scale.
    # We need high/low for ADX, so use BTC close as proxy for simplified True Range
    # via a pure-close approximation (Wilder's ATR variant on close changes):
    #   TR_close  = |close_t - close_t-1|
    #   +DM_close = max(close_t - close_t-1, 0)
    #   -DM_close = max(close_t-1 - close_t, 0)
    btc_c  = prices["BTC_close"]
    btc_d  = btc_c.diff()
    tr14   = btc_d.abs().ewm(span=14, adjust=False).mean()
    pdm14  = btc_d.clip(lower=0).ewm(span=14, adjust=False).mean()
    ndm14  = (-btc_d).clip(lower=0).ewm(span=14, adjust=False).mean()
    pdi14  = 100 * pdm14 / tr14.replace(0, np.nan)
    ndi14  = 100 * ndm14 / tr14.replace(0, np.nan)
    di_sum = (pdi14 + ndi14).replace(0, np.nan)
    dx     = 100 * (pdi14 - ndi14).abs() / di_sum
    out["btc_adx_14"]      = dx.ewm(span=14, adjust=False).mean()   # 0–100 trend strength
    out["btc_pdi_minus_ndi"] = pdi14 - ndi14                        # directional bias component

    # ── BTC volatility regime ─────────────────────────────────────────────────
    # Short/long realized vol ratio — captures vol compression before breakouts.
    # Ratio > 1 = vol expanding (trending/noisy), < 1 = vol compressed (coiling).
    btc_ret = btc_c.pct_change()
    vol_24h  = btc_ret.rolling(24).std()
    vol_168h = btc_ret.rolling(168).std()
    out["btc_vol_regime"] = vol_24h / vol_168h.replace(0, np.nan)

    # ── BTC price position within rolling range ───────────────────────────────
    # Percentile of current price within recent range — actual cycle position
    # without any MA cold-start bias.
    lo168  = btc_c.rolling(168).min()
    hi168  = btc_c.rolling(168).max()
    lo720  = btc_c.rolling(720).min()
    hi720  = btc_c.rolling(720).max()
    out["btc_pct_rank_168h"] = (btc_c - lo168) / (hi168 - lo168).replace(0, np.nan)
    out["btc_pct_rank_720h"] = (btc_c - lo720) / (hi720 - lo720).replace(0, np.nan)

    # Cross-sectional 24h return rank for each asset (normalised 0–1)
    ret_24h_cols = [c for c in prices.columns if c.endswith("_ret_24h")]
    if len(ret_24h_cols) == len(prices.columns.str.extract(r"^(\w+)_ret_24h")[0].dropna()):
        pass
    for col in ret_24h_cols:
        pass  # assembled below

    # Build a temp frame of 24h returns to rank
    ret_cols = {col.replace("_ret_24h", ""): prices[col]
                for col in prices.columns if col.endswith("_ret_24h")}
    if ret_cols:
        ret_frame = pd.DataFrame(ret_cols)
        ranks     = ret_frame.rank(axis=1, pct=True)   # 0–1 normalised rank
        for asset in ret_cols:
            out[f"{asset}_rank_24h"] = ranks[asset]

    # BTC dominance proxy: BTC 24h ret minus mean of others
    if "BTC" in ret_cols:
        others     = [c for c in ret_cols if c != "BTC"]
        mean_other = ret_frame[others].mean(axis=1)
        out["btc_dominance_spread"] = ret_frame["BTC"] - mean_other

    # ── Rolling BTC correlation per altcoin ───────────────────────────────────
    # How locked-in each altcoin is with BTC. High corr = less independent alpha.
    btc_ret = prices.get("BTC_ret_24h", pd.Series(dtype=float))
    for asset in ret_cols:
        if asset == "BTC":
            continue
        alt_ret = prices.get(f"{asset}_ret_24h", pd.Series(dtype=float))
        if not btc_ret.empty and not alt_ret.empty:
            # 48h and 168h rolling Pearson correlation
            for w in [48, 168]:
                out[f"{asset}_btc_corr_{w}h"] = (
                    btc_ret.rolling(w).corr(alt_ret)
                )

    # Shift by 1 (already shifted per-asset features feed here, but cross-asset
    # computes from those shifted features — no additional shift needed)
    return out


# ── Main builder ───────────────────────────────────────────────────────────────
def build_features(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Returns:
        X    — feature matrix (rows = bars × assets, stacked long format)
        y    — target series  (0/1, next-bar direction)
        meta — DataFrame with (asset, timestamp) for each row in X/y
    """
    if cfg is None:
        cfg = load_cfg()

    assets   = cfg["assets"]
    ohlcv_dir = Path(cfg["ohlcv_dir"]).expanduser()
    fpq_dir   = Path(cfg["feature_parquet_dir"]).expanduser()

    # ── 1. Per-asset OHLCV features ───────────────────────────────────────────
    print("\n[1] Loading OHLCV and computing per-asset features...")
    asset_dfs: dict[str, pd.DataFrame] = {}
    close_px: dict[str, pd.Series] = {}

    for asset in assets:
        raw   = load_ohlcv(asset, ohlcv_dir)
        feats = add_ohlcv_features(raw, asset)
        asset_dfs[asset] = feats
        close_px[asset]  = raw["close"]
        print(f"  {asset}: {len(raw)} bars, {len(feats.columns)} features")

    # Build a wide-format combined frame for cross-asset features
    # Rename close column for merging
    close_wide = pd.DataFrame({f"{a}_close": close_px[a] for a in assets})
    # Add 24h ret columns for cross-asset ranking
    ret24_wide = pd.DataFrame({
        f"{a}_ret_24h": asset_dfs[a][f"{a}_ret_24h"] for a in assets
    })
    wide = pd.concat([close_wide, ret24_wide], axis=1)

    # ── 2. Cross-asset features ───────────────────────────────────────────────
    print("\n[2] Computing cross-asset features...")
    cross_feats = add_cross_asset_features(wide)
    print(f"  Cross-asset features: {len(cross_feats.columns)}")

    # ── 3. Pipeline features ──────────────────────────────────────────────────
    print("\n[3] Loading pipeline (funding/OI/L/S) features...")
    # Get date range from OHLCV
    all_idx  = asset_dfs[assets[0]].index
    start_dt = all_idx[0]
    end_dt   = all_idx[-1]
    pipeline = load_pipeline_features(fpq_dir, start_dt, end_dt, assets)

    # ── 3b. Macro features ────────────────────────────────────────────────────
    print("\n[3b] Loading macro features (DXY, VIX, Fear & Greed)...")
    macro_feats = load_macro_features(start_dt, end_dt, asset_dfs[assets[0]].index)
    if not macro_feats.empty:
        print(f"  Macro features: {len(macro_feats.columns)} columns")
    else:
        print("  Macro features: none loaded (all sources offline)")

    # ── 4. Assemble per-asset labeled datasets ────────────────────────────────
    print("\n[4] Assembling feature matrix (long format, stacked assets)...")
    asset_frames = []

    for asset in assets:
        af = asset_dfs[asset].copy()

        # Merge cross-asset features
        af = af.join(cross_feats, how="left")

        # Merge macro features
        if not macro_feats.empty:
            af = af.join(macro_feats, how="left")

        # Merge pipeline features (merge_asof ± 30min tolerance)
        if not pipeline.empty:
            # Select this asset's pipeline columns
            pipe_cols  = [c for c in pipeline.columns if c.startswith(asset + "_")]
            if pipe_cols:
                pipe_sub = pipeline[pipe_cols].copy()
                pipe_sub.index = pipe_sub.index.tz_localize(None) if pipe_sub.index.tz else pipe_sub.index
                af_notz = af.copy()
                af_notz.index = af_notz.index.tz_localize(None) if af_notz.index.tz else af_notz.index

                merged = pd.merge_asof(
                    af_notz.reset_index().rename(columns={"datetime": "ts",
                                                           "index": "ts"}),
                    pipe_sub.reset_index().rename(columns={pipe_sub.index.name or "index": "ts"}),
                    on="ts",
                    tolerance=pd.Timedelta("30min"),
                    direction="backward",
                )
                merged = merged.set_index("ts")
                merged.index = pd.to_datetime(merged.index, utc=True)
                af = merged

        # Extract target
        target_col = f"{asset}_target"
        if target_col not in af.columns:
            print(f"  WARNING: no target column for {asset}, skipping")
            continue

        y_asset   = af[target_col]
        X_asset   = af.drop(columns=[c for c in af.columns if c.endswith("_target")])

        # Drop the last bar (target = NaN since we shift(-1) beyond the data end)
        valid     = y_asset.notna()
        X_asset   = X_asset[valid]
        y_asset   = y_asset[valid]

        X_asset["_asset"]     = asset
        X_asset["_timestamp"] = X_asset.index
        asset_frames.append((X_asset, y_asset))
        print(f"  {asset}: {len(X_asset)} labeled bars")

    # Stack
    X_all   = pd.concat([f[0] for f in asset_frames], axis=0)
    y_all   = pd.concat([f[1] for f in asset_frames], axis=0)

    meta    = X_all[["_asset", "_timestamp"]].copy()
    X_all   = X_all.drop(columns=["_asset", "_timestamp"])

    # ── 5. Diagnostics ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FEATURE MATRIX DIAGNOSTICS")
    print("="*60)
    print(f"Shape            : {X_all.shape[0]:,} rows × {X_all.shape[1]} columns")
    print(f"Date range       : {meta['_timestamp'].min()} → {meta['_timestamp'].max()}")
    print(f"Assets           : {meta['_asset'].unique().tolist()}")
    print(f"Rows per asset   : {meta['_asset'].value_counts().to_dict()}")
    print(f"Class balance    : {y_all.mean()*100:.1f}% positive labels (up bars)")

    null_counts = X_all.isnull().sum()
    high_null   = null_counts[null_counts > len(X_all) * 0.05].sort_values(ascending=False)
    if len(high_null) > 0:
        print(f"\nColumns with >5% nulls ({len(high_null)} cols):")
        for col, cnt in high_null.items():
            print(f"  {col}: {cnt:,} ({cnt/len(X_all)*100:.1f}%)")
    else:
        print(f"\nNull counts: all columns <5% null")

    low_null = null_counts[null_counts <= len(X_all) * 0.05]
    if len(low_null) > 0:
        print(f"Dense columns (<5% null): {len(low_null)}")

    print("\nFirst 3 rows:")
    print(X_all.head(3).to_string())
    print("\nLast 3 rows:")
    print(X_all.tail(3).to_string())
    print("="*60)

    return X_all, y_all, meta


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, meta = build_features()
    print(f"\nDone. X={X.shape}, y={y.shape}")
