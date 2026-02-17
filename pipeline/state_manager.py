"""
Pipeline State Manager
Manages pipeline_state.json for tracking trial progress through validation gates.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List


class PipelineStateManager:
    """Manages pipeline state persistence and queries."""

    def __init__(self, state_file: str = "deployments/pipeline_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load pipeline state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
                return self._init_state()
        return self._init_state()

    def _init_state(self) -> Dict:
        """Initialize empty state structure."""
        return {
            "trials": {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

    def save(self):
        """Save state to file."""
        self.state["metadata"]["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def get_trial(self, trial_number: int) -> Optional[Dict]:
        """Get trial state by number."""
        trial_key = str(trial_number)
        return self.state["trials"].get(trial_key)

    def add_trial(self, trial_number: int, value: float):
        """Add new trial to pipeline state."""
        trial_key = str(trial_number)

        if trial_key in self.state["trials"]:
            self.logger.info(f"Trial {trial_number} already in pipeline")
            return

        self.state["trials"][trial_key] = {
            "trial_number": trial_number,
            "value": value,
            "discovered_at": datetime.now().isoformat(),
            "current_stage": "pending",
            "stages": {
                "backtest": {"status": "pending"},
                "cge_stress": {"status": "pending"},
                "paper_trading": {"status": "pending"},
                "live_trading": {"status": "pending"}
            },
            "retry_counts": {
                "backtest": 0,
                "cge_stress": 0
            }
        }
        self.save()
        self.logger.info(f"Added trial {trial_number} to pipeline")

    def update_stage(self, trial_number: int, stage: str, status: str,
                     metrics: Optional[Dict] = None, error: Optional[str] = None):
        """Update a trial's stage status."""
        trial_key = str(trial_number)

        if trial_key not in self.state["trials"]:
            self.logger.warning(f"Trial {trial_number} not in pipeline")
            return

        trial = self.state["trials"][trial_key]

        # Update stage
        trial["stages"][stage] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }

        if metrics:
            trial["stages"][stage]["metrics"] = metrics

        if error:
            trial["stages"][stage]["error"] = error

        # Update current stage
        if status == "passed":
            # Move to next stage
            stage_order = ["backtest", "cge_stress", "paper_trading", "live_trading"]
            current_idx = stage_order.index(stage)
            if current_idx < len(stage_order) - 1:
                trial["current_stage"] = stage_order[current_idx + 1]
            else:
                trial["current_stage"] = "completed"
        elif status == "failed":
            trial["current_stage"] = f"{stage}_failed"

        self.save()
        self.logger.info(f"Trial {trial_number}: {stage} -> {status}")

    def increment_retry(self, trial_number: int, stage: str) -> int:
        """Increment retry counter for a stage."""
        trial_key = str(trial_number)

        if trial_key not in self.state["trials"]:
            return 0

        trial = self.state["trials"][trial_key]

        if stage not in trial["retry_counts"]:
            trial["retry_counts"][stage] = 0

        trial["retry_counts"][stage] += 1
        self.save()

        return trial["retry_counts"][stage]

    def get_trials_at_stage(self, stage: str) -> List[Dict]:
        """Get all trials at a specific stage."""
        trials = []
        for trial_key, trial_data in self.state["trials"].items():
            if trial_data.get("current_stage") == stage:
                trials.append(trial_data)
        return trials

    def get_trials_pending_backtest(self) -> List[Dict]:
        """Get trials pending backtest."""
        return self.get_trials_at_stage("pending")

    def get_trials_pending_cge(self) -> List[Dict]:
        """Get trials that passed backtest, pending CGE."""
        return self.get_trials_at_stage("cge_stress")

    def get_trials_pending_paper(self) -> List[Dict]:
        """Get trials ready for paper trading."""
        return self.get_trials_at_stage("paper_trading")

    def is_trial_in_pipeline(self, trial_number: int) -> bool:
        """Check if trial is already tracked."""
        return str(trial_number) in self.state["trials"]

    def get_all_trials(self) -> Dict[str, Dict]:
        """Get all trials."""
        return self.state["trials"]

    def cleanup_old_trials(self, keep_last_n: int = 100):
        """Remove old trials, keeping only the last N."""
        if len(self.state["trials"]) <= keep_last_n:
            return

        # Sort by discovery time
        trials_sorted = sorted(
            self.state["trials"].items(),
            key=lambda x: x[1].get("discovered_at", ""),
            reverse=True
        )

        # Keep only last N
        self.state["trials"] = dict(trials_sorted[:keep_last_n])
        self.save()
        self.logger.info(f"Cleaned up old trials, kept {keep_last_n}")
