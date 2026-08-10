"""
XGBoost Walk-Forward Trainer
Walk-forward validation: 3 folds, no random splits
One XGBoost classifier per asset.

Usage:
    python train.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, roc_auc_score)

BASE_DIR = Path(__file__).parent
CFG_FILE = BASE_DIR / "config.yaml"


# ── Booster wrapper for save/importance compatibility ─────────────────────────
class _BoosterWrapper:
    """Thin wrapper around xgb.Booster so importance extraction works uniformly."""
    def __init__(self, booster: xgb.Booster, feature_names: list[str]):
        self.booster       = booster
        self.feature_names = feature_names

    def get_booster(self) -> xgb.Booster:
        return self.booster

    def save_model(self, path: str):
        self.booster.save_model(path)


# ── Config ─────────────────────────────────────────────────────────────────────
def load_cfg() -> dict:
    with open(CFG_FILE) as f:
        return yaml.safe_load(f)


# ── Feature selection per asset ────────────────────────────────────────────────
def get_feature_cols(asset: str, all_cols: list[str]) -> list[str]:
    """
    For a given asset, select:
    - All columns starting with '{asset}_' (excluding the target column)
    - All cross-asset / regime columns (no asset prefix or shared prefix)
    """
    cross_asset_cols = [
        "btc_adx_14",
        "btc_pdi_minus_ndi",
        "btc_vol_regime",
        "btc_pct_rank_168h",
        "btc_pct_rank_720h",
        "btc_dominance_spread",
        # Macro / external
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
        and not c.endswith("_rank_24h")   # rank cols handled separately
    ]

    # Deduplicate while preserving order
    seen: set[str] = set()
    selected: list[str] = []
    for c in own_cols + cross_asset_cols + rank_cols:
        if c in all_cols and c not in seen:
            selected.append(c)
            seen.add(c)
    return selected


# ── Fold definitions ───────────────────────────────────────────────────────────
def build_folds(
    timestamps: pd.Series,
    start_dt: pd.Timestamp,
    fold_specs: list[dict],
) -> list[dict]:
    """
    Convert month-offset fold specs into (train_mask, test_mask) pairs.
    """
    folds = []
    for spec in fold_specs:
        train_end  = start_dt + pd.DateOffset(months=spec["train_end"])
        test_start = start_dt + pd.DateOffset(months=spec["test_start"])
        test_end   = start_dt + pd.DateOffset(months=spec["test_end"])

        train_mask = timestamps < train_end
        test_mask  = (timestamps >= test_start) & (timestamps < test_end)
        folds.append({
            "train_mask": train_mask,
            "test_mask":  test_mask,
            "label":      f"months {spec['train_end']}-{spec['test_end']}",
            "train_end":  train_end,
            "test_start": test_start,
            "test_end":   test_end,
        })
    return folds


# ── Train one fold ─────────────────────────────────────────────────────────────
def train_fold(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
    xgb_params: dict,
    asset: str,
    fold_idx: int,
) -> tuple[xgb.XGBClassifier, dict]:

    # XGBoost 3.x: no use_label_encoder, pass numpy to avoid pandas 2.x compat issues
    feat_names  = list(X_train.columns)
    Xtr_np = X_train.values.astype(np.float32)
    Xte_np = X_test.values.astype(np.float32)
    ytr_np = y_train.values.astype(np.int32)
    yte_np = y_test.values.astype(np.int32)

    Dtr = xgb.DMatrix(Xtr_np, label=ytr_np, feature_names=feat_names, missing=np.nan)
    Dte = xgb.DMatrix(Xte_np, label=yte_np, feature_names=feat_names, missing=np.nan)

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "max_depth":        xgb_params["max_depth"],
        "learning_rate":    xgb_params["learning_rate"],
        "subsample":        xgb_params["subsample"],
        "colsample_bytree": xgb_params["colsample_bytree"],
        "min_child_weight": xgb_params["min_child_weight"],
        "seed":             xgb_params.get("random_state", 42),
        "verbosity":        0,
    }

    callbacks = [xgb.callback.EarlyStopping(
        rounds=xgb_params["early_stopping_rounds"],
        metric_name="auc",
        maximize=True,
        save_best=True,
    )]

    booster = xgb.train(
        params,
        Dtr,
        num_boost_round=xgb_params["n_estimators"],
        evals=[(Dte, "eval")],
        callbacks=callbacks,
        verbose_eval=False,
    )

    # Wrap in sklearn-style object for compatibility with importance extraction
    model = _BoosterWrapper(booster, feat_names)

    proba  = booster.predict(Dte)
    preds  = (proba >= 0.5).astype(int)
    auc    = roc_auc_score(yte_np, proba)
    acc    = accuracy_score(yte_np, preds)
    prec   = precision_score(yte_np, preds, zero_division=0)
    rec    = recall_score(yte_np, preds, zero_division=0)
    n_est  = booster.num_boosted_rounds()

    metrics = {
        "auc":       round(auc,  4),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "best_iter": int(n_est),
    }
    return model, metrics


# ── Feature importances ────────────────────────────────────────────────────────
def aggregate_importances(
    models: dict[str, list[xgb.XGBClassifier]],
    feature_lists: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Mean gain importance across all assets and folds.
    Returns a DataFrame sorted by mean_gain descending.
    """
    rows = []
    for asset, fold_models in models.items():
        for m in fold_models:
            scores = m.get_booster().get_score(importance_type="gain")
            cols   = feature_lists[asset]
            for feat, gain in scores.items():
                rows.append({"feature": feat, "gain": gain, "asset": asset})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby("feature")["gain"]
              .mean()
              .rename("mean_gain")
              .reset_index()
              .sort_values("mean_gain", ascending=False)
              .reset_index(drop=True))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    cfg         = load_cfg()
    assets      = cfg["assets"]
    xgb_params  = cfg["xgb"]
    auc_thresh  = cfg["auc_threshold"]
    acc_thresh  = cfg["accuracy_threshold"]
    model_dir   = BASE_DIR / cfg["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("XGBoost Walk-Forward Training")
    print("=" * 65)
    print(f"Assets  : {assets}")
    print(f"Thresholds: AUC ≥ {auc_thresh} | Accuracy ≥ {acc_thresh}")
    print()

    # ── Load features ────────────────────────────────────────────────────────
    sys.path.insert(0, str(BASE_DIR))
    from feature_builder import build_features
    X_all, y_all, meta = build_features(cfg)

    meta = meta.reset_index(drop=True)
    X_all = X_all.reset_index(drop=True)
    y_all = y_all.reset_index(drop=True)

    # ── Walk-forward per asset ────────────────────────────────────────────────
    all_results:  dict[str, list[dict]]                 = {}
    trained_models: dict[str, list[xgb.XGBClassifier]] = {}
    feature_lists:  dict[str, list[str]]                = {}

    for asset in assets:
        print(f"\n{'─'*65}")
        print(f"Asset: {asset}")
        print(f"{'─'*65}")

        # Filter to this asset's rows
        asset_mask = meta["_asset"] == asset
        X_asset    = X_all[asset_mask].copy()
        y_asset    = y_all[asset_mask].copy()
        ts_asset   = meta.loc[asset_mask, "_timestamp"].reset_index(drop=True)

        X_asset    = X_asset.reset_index(drop=True)
        y_asset    = y_asset.reset_index(drop=True)

        # Select relevant features
        feat_cols  = get_feature_cols(asset, list(X_asset.columns))
        X_asset    = X_asset[feat_cols]
        feature_lists[asset] = feat_cols

        # Drop warm-up NaNs
        ret_1h_col = f"{asset}_ret_1h"
        valid_rows = X_asset[ret_1h_col].notna()
        X_asset    = X_asset[valid_rows].reset_index(drop=True)
        y_asset    = y_asset[valid_rows].reset_index(drop=True)
        ts_asset   = ts_asset[valid_rows].reset_index(drop=True)

        # No regime filter — train on all bars
        bull_mask = pd.Series(True, index=X_asset.index)

        # Build fold masks
        start_dt   = ts_asset.iloc[0]
        folds      = build_folds(ts_asset, start_dt, cfg["folds"])

        asset_results  = []
        asset_models   = []

        for fold_idx, fold in enumerate(folds):
            train_mask = fold["train_mask"]
            test_mask  = fold["test_mask"]

            # Training: bull-regime bars only
            X_tr = X_asset[train_mask & bull_mask]
            y_tr = y_asset[train_mask & bull_mask]
            # Test: full OOS period (mixed regime — mirrors live conditions)
            X_te = X_asset[test_mask]
            y_te = y_asset[test_mask]

            if len(X_tr) < 100 or len(X_te) < 10:
                print(f"  Fold {fold_idx+1}: insufficient data (train={len(X_tr)}, test={len(X_te)}) — skipping")
                continue

            model, metrics = train_fold(
                X_tr, y_tr, X_te, y_te,
                xgb_params, asset, fold_idx
            )

            train_period = f"{fold['train_end'].strftime('%Y-%m')} cutoff"
            test_period  = (f"{fold['test_start'].strftime('%Y-%m')} → "
                            f"{fold['test_end'].strftime('%Y-%m')}")
            print(f"  Fold {fold_idx+1}  train→{train_period}  test={test_period}")
            print(f"    AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
                  f"n_train={metrics['n_train']:,}  n_test={metrics['n_test']:,}  "
                  f"best_iter={metrics['best_iter']}")

            asset_results.append({
                "fold": fold_idx + 1,
                "test_period": test_period,
                **metrics,
            })
            asset_models.append(model)

            # Save model
            mpath = model_dir / f"{asset}_fold{fold_idx+1}.json"
            model.save_model(str(mpath))

        all_results[asset]     = asset_results
        trained_models[asset]  = asset_models

    # ── Aggregate OOS metrics ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("COMBINED OOS RESULTS")
    print(f"{'='*65}")

    all_aucs: list[float] = []
    all_accs: list[float] = []

    print(f"\n{'Asset':<6} {'Fold':<6} {'AUC':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'n_test':<8}")
    print("-" * 55)
    for asset in assets:
        for r in all_results.get(asset, []):
            print(f"{asset:<6} {r['fold']:<6} {r['auc']:<8.4f} {r['accuracy']:<8.4f} "
                  f"{r['precision']:<8.4f} {r['recall']:<8.4f} {r['n_test']:<8,}")
            all_aucs.append(r["auc"])
            all_accs.append(r["accuracy"])

    if not all_aucs:
        print("\nNo folds completed — check data or fold config.")
        sys.exit(1)

    combined_auc = float(np.mean(all_aucs))
    combined_acc = float(np.mean(all_accs))

    print(f"\nCombined mean OOS AUC     : {combined_auc:.4f}")
    print(f"Combined mean OOS Accuracy: {combined_acc:.4f}")

    # ── Hard stop check ───────────────────────────────────────────────────────
    auc_pass = combined_auc >= auc_thresh
    acc_pass = combined_acc >= acc_thresh

    if not auc_pass and not acc_pass:
        print(f"\n{'!'*65}")
        print("HARD STOP — Results below both thresholds")
        print(f"  AUC     : {combined_auc:.4f}  (threshold ≥ {auc_thresh}) ✗")
        print(f"  Accuracy: {combined_acc:.4f}  (threshold ≥ {acc_thresh}) ✗")
        print(f"{'!'*65}")
        print("\nThe XGBoost models show no predictive edge on this feature set.")
        print("Options to consider:")
        print("  1. Add more features (volume imbalance, order book, on-chain)")
        print("  2. Reduce prediction horizon (next-bar is very noisy at 1h)")
        print("  3. Target a different label (e.g., next 4h return direction)")
        print("  4. Restrict to high-confidence regime only (bull market only)")
        print("\nHow would you like to proceed?")
        _save_metrics(all_results, combined_auc, combined_acc, cfg, "FAIL")
        sys.exit(0)

    if not auc_pass:
        print(f"\nWARNING: AUC {combined_auc:.4f} < threshold {auc_thresh}")
    if not acc_pass:
        print(f"\nWARNING: Accuracy {combined_acc:.4f} < threshold {acc_thresh}")

    if auc_pass or acc_pass:
        print(f"\n{'='*65}")
        print("PASS — At least one threshold met. Computing feature importances.")
        print(f"{'='*65}\n")

        imp_df = aggregate_importances(trained_models, feature_lists)
        if not imp_df.empty:
            top20 = imp_df.head(20)
            print(f"{'Rank':<5} {'Feature':<40} {'Mean Gain':<12}")
            print("-" * 58)
            for i, row in top20.iterrows():
                print(f"{i+1:<5} {row['feature']:<40} {row['mean_gain']:,.1f}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    _save_metrics(all_results, combined_auc, combined_acc, cfg,
                  "PASS" if (auc_pass and acc_pass) else "PARTIAL",
                  imp_df.head(20).to_dict("records") if (auc_pass or acc_pass) else [])

    print(f"\nModels saved to: {model_dir}")
    print(f"Metrics saved to: {BASE_DIR / 'results/metrics.json'}")


def _save_metrics(
    all_results: dict,
    combined_auc: float,
    combined_acc: float,
    cfg: dict,
    verdict: str,
    top_features: list | None = None,
):
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict":   verdict,
        "combined_oos_auc":      combined_auc,
        "combined_oos_accuracy": combined_acc,
        "per_asset": all_results,
        "top_20_features_by_mean_gain": top_features or [],
        "config": cfg,
    }
    metrics_path = BASE_DIR / "results" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
