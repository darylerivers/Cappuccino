"""
Adaptive Ensemble Agent with Game Theory-style Model Elimination

Each model votes on actions. Models are scored based on:
- Whether their vote direction matched the actual price movement
- Accumulated accuracy over time

The worst-performing model gets eliminated periodically and can be
replaced with a new best trial from training.
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ModelScore:
    """Track a model's voting performance."""
    trial_number: int
    correct_votes: int = 0
    total_votes: int = 0
    cumulative_pnl: float = 0.0  # P&L if we only followed this model
    last_votes: List[float] = field(default_factory=list)  # Recent vote history

    @property
    def accuracy(self) -> float:
        if self.total_votes == 0:
            return 0.5  # Neutral score for new models
        return self.correct_votes / self.total_votes

    @property
    def score(self) -> float:
        """Combined score: accuracy + recency-weighted performance."""
        base_score = self.accuracy
        # Bonus for recent correct predictions (last 10)
        if len(self.last_votes) >= 5:
            recent_accuracy = sum(self.last_votes[-10:]) / len(self.last_votes[-10:])
            base_score = 0.7 * base_score + 0.3 * recent_accuracy
        return base_score


class AdaptiveEnsembleAgent:
    """
    Ensemble that tracks model performance and eliminates poor performers.

    Features:
    - Each model votes on actions
    - Votes are weighted by model accuracy
    - Worst model eliminated after `elimination_interval` steps
    - Can auto-replace with better trials from ongoing training
    """

    def __init__(
        self,
        ensemble_dir: str,
        device: str = "cpu",
        elimination_interval: int = 24,  # Eliminate worst model every 24 hours
        min_models: int = 5,  # Never go below this many models
        score_file: str = "model_scores.json",
    ):
        self.ensemble_dir = Path(ensemble_dir)
        self.device = torch.device(device)
        self.elimination_interval = elimination_interval
        self.min_models = min_models
        self.score_file = self.ensemble_dir / score_file

        self.models: Dict[int, Tuple] = {}  # trial_num -> (actor, trial)
        self.scores: Dict[int, ModelScore] = {}
        self.step_count = 0
        self.last_prices: Optional[np.ndarray] = None
        self.last_ensemble_action: Optional[np.ndarray] = None
        self.last_individual_actions: Dict[int, np.ndarray] = {}

        self._load_models()
        self._load_scores()

    def _load_models(self):
        """Load all models in the ensemble."""
        from train.run import init_agent
        from train.config import Arguments
        from drl_agents.elegantrl_models import MODELS

        manifest_path = self.ensemble_dir / "ensemble_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            trial_numbers = manifest.get("trial_numbers", [])
        else:
            # Scan for model directories
            trial_numbers = []
            for model_dir in sorted(self.ensemble_dir.glob("model_*")):
                best_trial = model_dir / "best_trial"
                if best_trial.exists():
                    with open(best_trial, "rb") as f:
                        trial = pickle.load(f)
                    trial_numbers.append(trial.number)

        print(f"Loading {len(trial_numbers)} models for adaptive ensemble...")

        for i, model_dir in enumerate(sorted(self.ensemble_dir.glob("model_*"))):
            best_trial = model_dir / "best_trial"
            actor_path = model_dir / "actor.pth"

            if not best_trial.exists() or not actor_path.exists():
                continue

            with open(best_trial, "rb") as f:
                trial = pickle.load(f)

            trial_num = trial.number

            # Load actor weights directly
            state_dict = torch.load(actor_path, map_location=self.device, weights_only=True)

            # Store for later inference
            self.models[trial_num] = {
                "state_dict": state_dict,
                "trial": trial,
                "model_dir": model_dir,
                "net_dim": trial.params.get("net_dimension", 256),
            }

            # Initialize score if new
            if trial_num not in self.scores:
                self.scores[trial_num] = ModelScore(trial_number=trial_num)

            print(f"  Loaded model #{trial_num} (score: {self.scores[trial_num].score:.3f})")

        print(f"Ensemble ready with {len(self.models)} models")

    def _load_scores(self):
        """Load historical scores from file."""
        if self.score_file.exists():
            with open(self.score_file) as f:
                data = json.load(f)
            for trial_num, score_data in data.items():
                trial_num = int(trial_num)
                if trial_num in self.models:
                    self.scores[trial_num] = ModelScore(
                        trial_number=trial_num,
                        correct_votes=score_data.get("correct_votes", 0),
                        total_votes=score_data.get("total_votes", 0),
                        cumulative_pnl=score_data.get("cumulative_pnl", 0.0),
                        last_votes=score_data.get("last_votes", [])[-20:],
                    )

    def _save_scores(self):
        """Save scores to file."""
        data = {}
        for trial_num, score in self.scores.items():
            data[str(trial_num)] = {
                "correct_votes": score.correct_votes,
                "total_votes": score.total_votes,
                "cumulative_pnl": score.cumulative_pnl,
                "last_votes": score.last_votes[-20:],
                "accuracy": score.accuracy,
                "score": score.score,
            }
        with open(self.score_file, "w") as f:
            json.dump(data, f, indent=2)

    def _build_actor(self, state_dim: int, action_dim: int, net_dim: int):
        """Build actor network architecture."""
        import torch.nn as nn

        # Match the ElegantRL PPO actor architecture (3 linear layers)
        net = nn.Sequential(
            nn.Linear(state_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, net_dim),
            nn.ReLU(),
            nn.Linear(net_dim, action_dim),
        )
        return net

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble action by weighted voting.

        Each model votes, weighted by their accuracy score.
        Returns the weighted average action.
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        state_dim = state.shape[1]
        actions = {}
        weights = {}

        for trial_num, model_data in self.models.items():
            try:
                # Build actor and load weights
                # Infer action_dim from state_dict - check for both architectures
                state_dict = model_data["state_dict"]
                if "net.4.weight" in state_dict:
                    action_dim = state_dict["net.4.weight"].shape[0]
                elif "net.6.weight" in state_dict:
                    action_dim = state_dict["net.6.weight"].shape[0]
                else:
                    print(f"Unknown architecture for model #{trial_num}, keys: {list(state_dict.keys())}")
                    continue
                net_dim = model_data["net_dim"]

                actor = self._build_actor(state_dim, action_dim, net_dim).to(self.device)
                # Filter state_dict to only include keys starting with 'net.'
                filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith('net.')}
                actor.load_state_dict(filtered_state_dict)
                actor.eval()

                with torch.no_grad():
                    action = actor(state)
                    # Apply tanh to constrain to [-1, 1]
                    action = torch.tanh(action)

                actions[trial_num] = action.cpu().numpy()[0]
                weights[trial_num] = max(0.1, self.scores[trial_num].score)  # Min weight 0.1

            except Exception as e:
                print(f"Error getting action from model #{trial_num}: {e}")
                continue

        if not actions:
            raise RuntimeError("No models produced valid actions")

        # Store individual actions for later scoring
        self.last_individual_actions = actions.copy()

        # Weighted average
        total_weight = sum(weights.values())
        ensemble_action = np.zeros_like(list(actions.values())[0])

        for trial_num, action in actions.items():
            ensemble_action += (weights[trial_num] / total_weight) * action

        self.last_ensemble_action = ensemble_action
        return torch.from_numpy(ensemble_action).unsqueeze(0).to(self.device)

    def update_scores(self, current_prices: np.ndarray, rewards: float):
        """
        Update model scores based on whether their votes were correct.

        A vote is "correct" if:
        - Model voted BUY (action > 0) and price went up
        - Model voted SELL (action < 0) and price went down
        """
        if self.last_prices is None or self.last_individual_actions is None:
            self.last_prices = current_prices
            return

        # Calculate price changes
        price_changes = (current_prices - self.last_prices) / (self.last_prices + 1e-8)

        for trial_num, action in self.last_individual_actions.items():
            if trial_num not in self.scores:
                continue

            score = self.scores[trial_num]

            # For each asset, check if vote direction matched price direction
            correct = 0
            total = 0

            for i, (act, price_change) in enumerate(zip(action, price_changes)):
                if abs(act) > 0.1:  # Only count significant votes
                    total += 1
                    # Correct if both positive or both negative
                    if (act > 0 and price_change > 0) or (act < 0 and price_change < 0):
                        correct += 1

            if total > 0:
                accuracy = correct / total
                score.correct_votes += correct
                score.total_votes += total
                score.last_votes.append(accuracy)

                # Keep last 20 votes
                if len(score.last_votes) > 20:
                    score.last_votes = score.last_votes[-20:]

        self.last_prices = current_prices
        self.step_count += 1

        # Save scores periodically
        if self.step_count % 10 == 0:
            self._save_scores()

        # Check for elimination
        if self.step_count % self.elimination_interval == 0:
            self._maybe_eliminate_worst()

    def _maybe_eliminate_worst(self):
        """Eliminate the worst-performing model if we have enough."""
        if len(self.models) <= self.min_models:
            print(f"Cannot eliminate: only {len(self.models)} models (min: {self.min_models})")
            return

        # Find worst model by score
        worst_trial = min(self.scores.keys(), key=lambda t: self.scores[t].score)
        worst_score = self.scores[worst_trial]

        # Only eliminate if score is significantly below average
        avg_score = np.mean([s.score for s in self.scores.values()])

        if worst_score.score < avg_score - 0.1:  # 10% below average
            print(f"\n{'='*60}")
            print(f"ELIMINATING MODEL #{worst_trial}")
            print(f"  Score: {worst_score.score:.3f} (avg: {avg_score:.3f})")
            print(f"  Accuracy: {worst_score.accuracy:.1%}")
            print(f"  Total votes: {worst_score.total_votes}")
            print(f"{'='*60}\n")

            # Remove from active models
            del self.models[worst_trial]

            # Log elimination
            self._log_elimination(worst_trial, worst_score)
        else:
            print(f"No elimination: worst model #{worst_trial} score {worst_score.score:.3f} >= threshold")

    def _log_elimination(self, trial_num: int, score: ModelScore):
        """Log model elimination to file."""
        log_file = self.ensemble_dir / "elimination_log.json"

        log_data = []
        if log_file.exists():
            with open(log_file) as f:
                log_data = json.load(f)

        log_data.append({
            "timestamp": datetime.now().isoformat(),
            "trial_number": trial_num,
            "final_score": score.score,
            "accuracy": score.accuracy,
            "total_votes": score.total_votes,
            "remaining_models": len(self.models),
        })

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def get_voting_summary(self) -> str:
        """Get a summary of current model votes and scores."""
        lines = ["Model Voting Summary:"]
        lines.append("-" * 50)

        sorted_scores = sorted(
            self.scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )

        for trial_num, score in sorted_scores:
            if trial_num in self.models:
                status = "ACTIVE"
            else:
                status = "ELIMINATED"

            lines.append(
                f"  #{trial_num:>3}: Score={score.score:.3f} "
                f"Acc={score.accuracy:.1%} "
                f"Votes={score.total_votes:>4} "
                f"[{status}]"
            )

        return "\n".join(lines)

    def get_required_hyperparameters(self) -> Dict:
        """Get hyperparameters from the first model (for environment setup)."""
        if not self.models:
            raise RuntimeError("No models loaded")

        # Get params from first model's trial
        first_model = list(self.models.values())[0]
        trial = first_model["trial"]
        params = trial.params

        return {
            'lookback': int(params.get('lookback', 60)),
            'norm_cash': 2 ** float(params.get('norm_cash_exp', -11)),
            'norm_stocks': 2 ** float(params.get('norm_stocks_exp', -8)),
            'norm_tech': 2 ** float(params.get('norm_tech_exp', -14)),
            'norm_reward': 2 ** float(params.get('norm_reward_exp', -9)),
            'norm_action': float(params.get('norm_action', 100)),
            'time_decay_floor': float(params.get('time_decay_floor', 0.0)),
            'min_cash_reserve': float(params.get('min_cash_reserve', 0.1)),
            'concentration_penalty': float(params.get('concentration_penalty', 0.05)),
        }

    def add_new_model(self, trial_num: int, model_dir: Path):
        """Add a new model to the ensemble (e.g., from ongoing training)."""
        best_trial = model_dir / "best_trial"
        actor_path = model_dir / "actor.pth"

        if not best_trial.exists() or not actor_path.exists():
            print(f"Cannot add model #{trial_num}: missing files")
            return False

        with open(best_trial, "rb") as f:
            trial = pickle.load(f)

        state_dict = torch.load(actor_path, map_location=self.device, weights_only=True)

        self.models[trial_num] = {
            "state_dict": state_dict,
            "trial": trial,
            "model_dir": model_dir,
            "net_dim": trial.params.get("net_dimension", 256),
        }

        self.scores[trial_num] = ModelScore(trial_number=trial_num)

        print(f"Added new model #{trial_num} to ensemble")
        return True

    def check_for_reload(self) -> bool:
        """Check if models need to be reloaded (hot-reload support)."""
        reload_flag = self.ensemble_dir / ".reload_models"
        if not reload_flag.exists():
            return False

        print("\n" + "=" * 60)
        print("HOT-RELOAD: Updating ensemble models...")
        print("=" * 60)

        # Remove the flag first
        reload_flag.unlink()

        # Re-read manifest to get updated trial list
        manifest_path = self.ensemble_dir / "ensemble_manifest.json"
        if not manifest_path.exists():
            print("No manifest found, skipping reload")
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        new_trial_numbers = set(manifest.get("trial_numbers", []))
        current_trial_numbers = set(self.models.keys())

        # Find models to add
        to_add = new_trial_numbers - current_trial_numbers
        # Find models to remove
        to_remove = current_trial_numbers - new_trial_numbers

        if not to_add and not to_remove:
            print("No changes detected")
            return False

        # Remove old models
        for trial_num in to_remove:
            print(f"  Removing model #{trial_num}")
            del self.models[trial_num]
            # Keep scores for history

        # Add new models
        for model_dir in sorted(self.ensemble_dir.glob("model_*")):
            best_trial = model_dir / "best_trial"
            actor_path = model_dir / "actor.pth"

            if not best_trial.exists() or not actor_path.exists():
                continue

            try:
                with open(best_trial, "rb") as f:
                    trial = pickle.load(f)

                if trial.number in to_add:
                    state_dict = torch.load(actor_path, map_location=self.device, weights_only=True)
                    self.models[trial.number] = {
                        "state_dict": state_dict,
                        "trial": trial,
                        "model_dir": model_dir,
                        "net_dim": trial.params.get("net_dimension", 256),
                    }
                    if trial.number not in self.scores:
                        self.scores[trial.number] = ModelScore(trial_number=trial.number)
                    print(f"  Added model #{trial.number} (score: {self.scores[trial.number].score:.3f})")
            except Exception as e:
                print(f"  Error loading model from {model_dir}: {e}")

        print(f"Ensemble updated: {len(self.models)} models active")
        print("=" * 60 + "\n")

        return True
