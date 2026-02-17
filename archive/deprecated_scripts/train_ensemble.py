#!/usr/bin/env python3
"""
Train ensemble model using top performing trials.

Uses voting/averaging strategy across multiple high-performing models.
"""

import argparse
import pickle
import sqlite3
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch

from config_main import TICKER_LIST
from drl_agents.elegantrl_models import MODELS
from environment_Alpaca import CryptoEnvAlpaca
from train.config import Arguments
from train.run import init_agent


class EnsembleTrainer:
    """Train and manage ensemble of top models."""

    def __init__(
        self,
        db_path: str = "databases/optuna_cappuccino.db",
        study_name: str = "cappuccino_3workers_20251102_2325",
        top_n: int = 20,
        percentile: float = 90,
        gpu_id: int = 0,
    ):
        self.db_path = db_path
        self.study_name = study_name
        self.top_n = top_n
        self.percentile = percentile
        self.gpu_id = gpu_id

        self.ensemble_models = []
        self.trial_ids = []

        print("=" * 80)
        print("ENSEMBLE TRAINING INITIALIZATION")
        print("=" * 80)

    def load_top_trials(self) -> List[Dict]:
        """Load top performing trials from database."""
        print(f"\n📊 Loading top {self.top_n} trials from database...")

        conn = sqlite3.connect(self.db_path)

        # Get top trials by value
        query = """
        SELECT
            t.trial_id,
            t.number,
            tv.value
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ?
            AND t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT ?
        """

        cursor = conn.cursor()
        cursor.execute(query, (self.study_name, self.top_n))
        results = cursor.fetchall()

        # Also get percentile threshold
        cursor.execute("""
            SELECT COUNT(*)
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            WHERE s.study_name = ?
                AND t.state = 'COMPLETE'
        """, (self.study_name,))
        total_count = cursor.fetchone()[0]
        offset = int(total_count * (100 - self.percentile) / 100)

        percentile_query = """
        SELECT value
        FROM trial_values tv
        JOIN trials t ON tv.trial_id = t.trial_id
        JOIN studies s ON t.study_id = s.study_id
        WHERE s.study_name = ?
            AND t.state = 'COMPLETE'
        ORDER BY value DESC
        LIMIT 1 OFFSET ?
        """
        cursor.execute(percentile_query, (self.study_name, offset))
        threshold_result = cursor.fetchone()
        threshold = threshold_result[0] if threshold_result else 0

        conn.close()

        print(f"✓ Found {len(results)} top trials")
        print(f"✓ Top 10% threshold: {threshold:.6f}")
        print(f"\nTop {min(5, len(results))} trials:")
        for i, (trial_id, number, value) in enumerate(results[:5], 1):
            print(f"  {i}. Trial #{number}: {value:.6f}")

        return [
            {'trial_id': trial_id, 'number': number, 'value': value}
            for trial_id, number, value in results
        ]

    def load_trial_model(self, trial_info: Dict):
        """Load a single trial's model."""
        trial_id = trial_info['trial_id']
        trial_number = trial_info['number']

        # Load trial object
        trial_dir = Path("train_results/cwd_tests")
        trial_folders = list(trial_dir.glob(f"trial_{trial_number}_*"))

        if not trial_folders:
            print(f"  ⚠️  Trial #{trial_number}: No folder found")
            return None

        trial_path = trial_folders[0] / "best_trial"
        if not trial_path.exists():
            print(f"  ⚠️  Trial #{trial_number}: No best_trial file")
            return None

        with trial_path.open("rb") as f:
            trial = pickle.load(f)

        # Load model
        model_name = trial.user_attrs.get("model_name", "ppo")
        name_folder = trial.user_attrs.get("name_folder")
        cwd_path = Path("train_results/cwd_tests") / name_folder / "stored_agent"

        if not cwd_path.exists():
            print(f"  ⚠️  Trial #{trial_number}: No stored_agent")
            return None

        # Get hyperparameters
        params = trial.params
        net_dimension = int(params["net_dimension"])

        # Create dummy env to get state/action dims
        from config_main import TECHNICAL_INDICATORS_LIST
        n_tickers = len(TICKER_LIST)
        n_tech = len(TECHNICAL_INDICATORS_LIST)

        # Create minimal environment
        dummy_price = np.ones((100, n_tickers), dtype=np.float32)
        dummy_tech = np.ones((100, n_tickers * n_tech), dtype=np.float32)

        env_config = {
            "price_array": dummy_price,
            "tech_array": dummy_tech,
            "if_train": False,
        }
        env_params = {
            "lookback": int(params["lookback"]),
            "norm_cash": float(2 ** params.get("norm_cash_exp", -11)),
            "norm_stocks": float(2 ** params.get("norm_stocks_exp", -8)),
            "norm_tech": float(2 ** params.get("norm_tech_exp", -14)),
            "norm_reward": float(2 ** params.get("norm_reward_exp", -9)),
            "norm_action": float(params["norm_action"]),
            "time_decay_floor": float(params.get("time_decay_floor", 0.0)),
            "min_cash_reserve": float(params.get("min_cash_reserve", 0.1)),
            "concentration_penalty": float(params.get("concentration_penalty", 0.05)),
        }

        env = CryptoEnvAlpaca(env_config, env_params, if_log=False, use_sentiment=False)

        # Initialize agent
        args = Arguments(agent=MODELS[model_name], env=env)
        args.cwd = str(cwd_path)
        args.if_remove = False
        args.net_dim = net_dimension

        agent = init_agent(args, gpu_id=self.gpu_id, env=env)

        print(f"  ✓ Trial #{trial_number}: Loaded (value={trial_info['value']:.6f})")

        return {
            'trial_number': trial_number,
            'agent': agent,
            'trial': trial,
            'value': trial_info['value'],
        }

    def build_ensemble(self):
        """Build ensemble from top trials."""
        print(f"\n🧠 Building ensemble with top {self.top_n} models...")

        top_trials = self.load_top_trials()

        loaded_count = 0
        for trial_info in top_trials:
            model = self.load_trial_model(trial_info)
            if model:
                self.ensemble_models.append(model)
                loaded_count += 1

        print(f"\n✓ Ensemble built: {loaded_count}/{self.top_n} models loaded")

        if loaded_count == 0:
            raise RuntimeError("Failed to load any models for ensemble")

        # Save ensemble info
        ensemble_dir = Path("train_results/ensemble")
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        ensemble_info = {
            'model_count': loaded_count,
            'trial_numbers': [m['trial_number'] for m in self.ensemble_models],
            'trial_values': [m['value'] for m in self.ensemble_models],
            'mean_value': np.mean([m['value'] for m in self.ensemble_models]),
            'created_at': str(Path.ctime(Path.cwd())),
        }

        with (ensemble_dir / "ensemble_info.pkl").open("wb") as f:
            pickle.dump(ensemble_info, f)

        print(f"✓ Ensemble info saved to {ensemble_dir / 'ensemble_info.pkl'}")
        print(f"\nEnsemble Statistics:")
        print(f"  Models: {ensemble_info['model_count']}")
        print(f"  Mean Value: {ensemble_info['mean_value']:.6f}")
        print(f"  Best Value: {max(ensemble_info['trial_values']):.6f}")
        print(f"  Worst Value: {min(ensemble_info['trial_values']):.6f}")

    def predict_ensemble(self, state: np.ndarray) -> np.ndarray:
        """Get ensemble prediction by averaging all model outputs."""
        predictions = []

        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)

        for model in self.ensemble_models:
            agent = model['agent']
            device = agent.device

            with torch.no_grad():
                action = agent.act(state_tensor.to(device))
                predictions.append(action.cpu().numpy()[0])

        # Average predictions
        ensemble_action = np.mean(predictions, axis=0)

        return ensemble_action


def main():
    parser = argparse.ArgumentParser(description="Train ensemble from top trials")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top models to use")
    parser.add_argument("--percentile", type=float, default=90, help="Percentile threshold")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--study-name", type=str, default="cappuccino_3workers_20251102_2325")
    args = parser.parse_args()

    trainer = EnsembleTrainer(
        top_n=args.top_n,
        percentile=args.percentile,
        gpu_id=args.gpu,
        study_name=args.study_name,
    )

    trainer.build_ensemble()

    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 80)
    print("\nYou can now use this ensemble for paper trading:")
    print("  python paper_trader_alpaca_polling.py --model-dir train_results/ensemble \\")
    print("    --tickers BTC/USD ETH/USD LTC/USD --timeframe 1h")


if __name__ == "__main__":
    main()
