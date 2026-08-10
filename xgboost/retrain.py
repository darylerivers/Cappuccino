"""
Automated XGBoost Retraining Pipeline
Exports fresh features, retrains all fold models, validates AUC,
and auto-promotes if the new models meet the quality threshold.

Usage:
    python retrain.py            # full retrain + auto-promote
    python retrain.py --dry-run  # retrain but do not promote

Cron example (weekly, Sunday 03:00 UTC):
    0 3 * * 0 cd /home/mrc/cappuccino/xgboost && /opt/miniconda3/bin/python3 retrain.py >> /tmp/retrain.log 2>&1
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
XGB_DIR      = Path(__file__).parent
PROJ_DIR     = XGB_DIR.parent
PIPELINE_DIR = PROJ_DIR / "data_pipeline"
MODEL_DIR    = XGB_DIR / "results" / "models"
METRICS_FILE = XGB_DIR / "results" / "metrics.json"
BACKUP_DIR   = XGB_DIR / "results" / "models_backup"

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(XGB_DIR))

# Telegram alerting (optional — reads momentum config)
def _send_alert(msg: str):
    try:
        momentum_cfg = PROJ_DIR / "momentum" / "config.yaml"
        if not momentum_cfg.exists():
            return
        import yaml, requests
        cfg = yaml.safe_load(open(momentum_cfg))
        token = cfg.get("telegram_bot_token", "")
        chat  = cfg.get("telegram_chat_id", "")
        if not token or not chat:
            return
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg},
            timeout=10,
        )
    except Exception:
        pass


def load_current_auc() -> float:
    """Return the current production combined OOS AUC, or 0 if no metrics exist."""
    if not METRICS_FILE.exists():
        return 0.0
    with open(METRICS_FILE) as f:
        metrics = json.load(f)
    aucs = [
        fold.get("auc", 0)
        for asset_folds in metrics.values()
        for fold in (asset_folds if isinstance(asset_folds, list) else [asset_folds])
    ]
    return sum(aucs) / len(aucs) if aucs else 0.0


def export_features():
    """Regenerate the full features parquet from the latest pipeline data."""
    print("\n[1] Exporting fresh features parquet...")
    from export_features import export, ALL_CCYS
    now   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    # Use all available history (collector has been running since deployment)
    start = datetime(2023, 12, 1, tzinfo=timezone.utc)
    df    = export(start=start, end=now, ccys=ALL_CCYS)
    print(f"   Exported {len(df)} rows × {len(df.columns)} cols")
    return df


def retrain() -> float:
    """Run the walk-forward training pipeline. Returns new combined OOS AUC."""
    print("\n[2] Retraining XGBoost models (walk-forward, 3 folds)...")
    import train as train_module
    train_module.main()

    # Read the AUC from freshly written metrics
    new_auc = load_current_auc()
    print(f"   New combined OOS AUC: {new_auc:.4f}")
    return new_auc


def backup_models():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for f in MODEL_DIR.glob("*.json"):
        shutil.copy2(f, BACKUP_DIR / f.name)
    print(f"   Models backed up to {BACKUP_DIR}")


def promote_models():
    print("   Promoting new models (already written to results/models/ by train.py)")


def rollback_models():
    if not BACKUP_DIR.exists():
        print("   No backup to roll back to.")
        return
    for f in BACKUP_DIR.glob("*.json"):
        shutil.copy2(f, MODEL_DIR / f.name)
    print(f"   Rolled back models from {BACKUP_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Retrain and auto-promote XGBoost models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Retrain but do not promote; roll back to previous models")
    parser.add_argument("--min-auc-delta", type=float, default=-0.005,
                        help="Minimum AUC change vs production to promote (default: -0.005)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"XGBoost Automated Retraining — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*60}")

    current_auc = load_current_auc()
    print(f"\nCurrent production AUC: {current_auc:.4f}")

    # 1. Backup existing models before overwriting
    backup_models()

    # 2. Export fresh data
    try:
        export_features()
    except Exception as e:
        print(f"ERROR: Feature export failed: {e}")
        _send_alert(f"⚠️ XGBoost retrain FAILED (feature export): {e}")
        sys.exit(1)

    # 3. Retrain (overwrites model files)
    try:
        new_auc = retrain()
    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        rollback_models()
        _send_alert(f"⚠️ XGBoost retrain FAILED (training): {e}")
        sys.exit(1)

    # 4. Evaluate promotion
    delta = new_auc - current_auc
    print(f"\n[3] Evaluation:")
    print(f"   Current AUC:  {current_auc:.4f}")
    print(f"   New AUC:      {new_auc:.4f}")
    print(f"   Delta:        {delta:+.4f}  (threshold: {args.min_auc_delta:+.4f})")

    if args.dry_run:
        print("\n   DRY RUN — rolling back to previous models.")
        rollback_models()
        _send_alert(
            f"🔬 XGBoost retrain (dry run)\n"
            f"Current AUC: {current_auc:.4f} → New: {new_auc:.4f} ({delta:+.4f})\n"
            f"Models NOT promoted (dry run mode)."
        )
        return

    if delta >= args.min_auc_delta:
        promote_models()
        result_msg = f"✅ XGBoost retrain — PROMOTED\nAUC: {current_auc:.4f} → {new_auc:.4f} ({delta:+.4f})"
        print(f"\n   PROMOTED — new models are live.")
    else:
        rollback_models()
        result_msg = (
            f"⚠️ XGBoost retrain — NOT PROMOTED\n"
            f"AUC: {current_auc:.4f} → {new_auc:.4f} ({delta:+.4f})\n"
            f"Below threshold ({args.min_auc_delta:+.4f}). Previous models restored."
        )
        print(f"\n   NOT PROMOTED — AUC below threshold. Previous models restored.")

    _send_alert(result_msg)
    print(f"\n{'='*60}\nDone.\n")


if __name__ == "__main__":
    main()
