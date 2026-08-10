"""
XGBoost Momentum Predictor
Loads all trained fold models, builds features for the latest bar,
and returns a LONG/HOLD recommendation per asset ranked by confidence.

Usage:
    python predict.py            # print recommendation table
    from predict import predict  # returns list[dict]
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

BASE_DIR = Path(__file__).parent
CFG_FILE = BASE_DIR / "config.yaml"


def load_cfg() -> dict:
    with open(CFG_FILE) as f:
        return yaml.safe_load(f)


def get_feature_cols(asset: str, all_cols: list[str]) -> list[str]:
    """Mirror of train.py — must stay in sync."""
    cross_asset_cols = [
        "btc_adx_14",
        "btc_pdi_minus_ndi",
        "btc_vol_regime",
        "btc_pct_rank_168h",
        "btc_pct_rank_720h",
        "btc_dominance_spread",
        # Macro / external — must match train.py
        "macro_dxy",
        "macro_dxy_ret_5d",
        "macro_vix",
        "macro_vix_high",
        "macro_fear_greed_norm",
    ]
    rank_cols = [c for c in all_cols if c.endswith("_rank_24h")]
    own_cols = [
        c for c in all_cols
        if c.startswith(f"{asset}_")
        and not c.endswith("_target")
        and not c.endswith("_rank_24h")
    ]
    seen: set[str] = set()
    selected: list[str] = []
    for c in own_cols + cross_asset_cols + rank_cols:
        if c in all_cols and c not in seen:
            selected.append(c)
            seen.add(c)
    return selected


def load_models(assets: list[str], model_dir: Path) -> dict[str, list[xgb.Booster]]:
    """Load all available fold models per asset."""
    models: dict[str, list[xgb.Booster]] = {}
    for asset in assets:
        fold_models = []
        for fold in range(1, 4):
            mpath = model_dir / f"{asset}_fold{fold}.json"
            if mpath.exists():
                b = xgb.Booster()
                b.load_model(str(mpath))
                fold_models.append(b)
        if fold_models:
            models[asset] = fold_models
        else:
            print(f"  WARNING: no models found for {asset}")
    return models


def predict(cfg: dict | None = None) -> list[dict]:
    """
    Returns a list of dicts, one per asset, sorted by confidence descending:
        {
            asset:       str,
            prob_up:     float,   # probability next bar closes higher (0–1)
            confidence:  float,   # distance from 0.5 — higher = more certain
            signal:      str,     # "LONG" or "HOLD"
            n_models:    int,     # number of fold models averaged
            bar_time:    str,     # UTC timestamp of the bar used for prediction
        }
    """
    if cfg is None:
        cfg = load_cfg()

    assets    = cfg["assets"]
    model_dir = BASE_DIR / cfg["model_dir"]

    # ── Build feature matrix ────────────────────────────────────────────────
    sys.path.insert(0, str(BASE_DIR))
    from feature_builder import build_features
    X_all, y_all, meta = build_features(cfg)

    meta  = meta.reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)

    # ── Load models ─────────────────────────────────────────────────────────
    models = load_models(assets, model_dir)

    # ── Predict per asset ───────────────────────────────────────────────────
    results = []

    for asset in assets:
        if asset not in models:
            continue

        asset_mask = meta["_asset"] == asset
        X_asset    = X_all[asset_mask].copy().reset_index(drop=True)
        ts_asset   = meta.loc[asset_mask, "_timestamp"].reset_index(drop=True)

        feat_cols = get_feature_cols(asset, list(X_asset.columns))
        X_asset   = X_asset[feat_cols]

        # Drop warm-up NaN rows (same as train.py)
        ret_1h_col = f"{asset}_ret_1h"
        valid_rows = X_asset[ret_1h_col].notna()
        X_asset    = X_asset[valid_rows].reset_index(drop=True)
        ts_asset   = ts_asset[valid_rows].reset_index(drop=True)

        if len(X_asset) == 0:
            print(f"  WARNING: no valid rows for {asset}")
            continue

        # Latest bar (last row)
        latest_X  = X_asset.iloc[[-1]]
        bar_time  = ts_asset.iloc[-1]

        Dlat = xgb.DMatrix(
            latest_X.values.astype(np.float32),
            feature_names=feat_cols,
            missing=np.nan,
        )

        # Average probability across all fold models
        fold_probs = [m.predict(Dlat)[0] for m in models[asset]]
        prob_up    = float(np.mean(fold_probs))
        confidence = abs(prob_up - 0.5)

        results.append({
            "asset":     asset,
            "prob_up":   round(prob_up, 4),
            "confidence": round(confidence, 4),
            "signal":    "LONG" if prob_up >= 0.5 else "HOLD",
            "n_models":  len(models[asset]),
            "bar_time":  bar_time.isoformat() if hasattr(bar_time, "isoformat") else str(bar_time),
        })

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def main():
    cfg     = load_cfg()
    top_n   = cfg.get("top_n", 2)  # how many to recommend LONG

    print("\n" + "=" * 60)
    print("XGBoost Momentum Predictor")
    print(f"Run at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 60)

    recs = predict(cfg)

    if not recs:
        print("No predictions generated — check models and data.")
        return

    bar_time = recs[0]["bar_time"]
    print(f"\nFeatures computed from bar: {bar_time}")
    print(f"Top-{top_n} by confidence → LONG recommendation\n")

    print(f"{'Rank':<5} {'Asset':<7} {'P(up)':<8} {'Confidence':<12} {'Signal':<8} {'Models'}")
    print("─" * 52)
    for i, r in enumerate(recs):
        rank_label = f"#{i+1}"
        signal_str = r["signal"]
        if i < top_n and r["signal"] == "LONG":
            signal_str = "★ LONG"
        print(f"{rank_label:<5} {r['asset']:<7} {r['prob_up']:<8.4f} "
              f"{r['confidence']:<12.4f} {signal_str:<8} {r['n_models']} folds")

    print("\n" + "─" * 52)
    top_longs = [r for r in recs[:top_n] if r["signal"] == "LONG"]
    if top_longs:
        print(f"RECOMMENDATION: LONG {', '.join(r['asset'] for r in top_longs)}")
    else:
        print("RECOMMENDATION: HOLD (no assets above 50% confidence)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
