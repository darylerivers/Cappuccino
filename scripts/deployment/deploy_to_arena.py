#!/usr/bin/env python3
"""
Deploy Model to Arena

Deploys a trained model from a completed trial to the live arena for evaluation.

Usage:
    python deploy_to_arena.py --trial 2 --study test_fixes_dec15
    python deploy_to_arena.py --trial 1226 --value 0.073
"""

import argparse
import sqlite3
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from model_arena import ModelArena


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def get_trial_info(trial_number: int, study_name: str, db_path: str = 'databases/optuna_cappuccino.db'):
    """Get trial information from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get trial value
    cursor.execute("""
    SELECT tv.value
    FROM trials t
    LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
    WHERE t.study_id = (SELECT study_id FROM studies WHERE study_name = ?)
      AND t.number = ?
      AND t.state = 'COMPLETE'
    """, (study_name, trial_number))

    result = cursor.fetchone()
    conn.close()

    if not result or result[0] is None:
        return None

    return result[0]


def main():
    parser = argparse.ArgumentParser(description="Deploy model to arena")
    parser.add_argument('--trial', type=int, required=True, help='Trial number to deploy')
    parser.add_argument('--study', type=str, default='test_fixes_dec15', help='Study name')
    parser.add_argument('--value', type=float, default=None, help='Training value (optional, will query from DB)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (default: 1h)')
    args = parser.parse_args()

    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}DEPLOY MODEL TO ARENA{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    # Get trial value if not provided
    if args.value is None:
        print(f"{Colors.BLUE}Fetching trial info from database...{Colors.END}")
        args.value = get_trial_info(args.trial, args.study)
        if args.value is None:
            print(f"{Colors.RED}ERROR: Trial {args.trial} not found in study {args.study}{Colors.END}")
            print(f"{Colors.YELLOW}Make sure the trial is complete and has a value.{Colors.END}")
            return 1

    # Check trial directory
    trial_dir = Path(f"train_results/cwd_tests/trial_{args.trial}_{args.timeframe}")

    if not trial_dir.exists():
        print(f"{Colors.RED}ERROR: Trial directory not found: {trial_dir}{Colors.END}")
        return 1

    actor_path = trial_dir / "actor.pth"
    if not actor_path.exists():
        print(f"{Colors.RED}ERROR: No actor.pth found in {trial_dir}{Colors.END}")
        return 1

    print(f"{Colors.CYAN}Trial Information:{Colors.END}")
    print(f"  Study: {args.study}")
    print(f"  Trial: {args.trial}")
    print(f"  Training value: {args.value:.6f}")
    print(f"  Directory: {trial_dir}")
    print(f"  Actor file: {actor_path} ({actor_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print()

    # Initialize arena
    print(f"{Colors.BLUE}Loading arena...{Colors.END}")
    arena = ModelArena(
        tickers=['BTC', 'ETH', 'LTC', 'BCH', 'LINK', 'UNI', 'AAVE'],
        max_models=20,
        state_dir=Path("arena_state"),
    )

    # Check if model already in arena
    model_id = f"trial_{args.trial}"
    if model_id in arena.portfolios:
        print(f"{Colors.YELLOW}WARNING: {model_id} is already in the arena!{Colors.END}")
        print(f"  Current return: {arena.portfolios[model_id].get_return_pct():.2f}%")
        print(f"  Time in arena: {arena.portfolios[model_id].hours_in_arena():.1f}h")
        print()

        response = input(f"{Colors.CYAN}Replace existing model? (y/n): {Colors.END}")
        if response.lower() != 'y':
            print(f"{Colors.YELLOW}Deployment cancelled.{Colors.END}")
            return 0

        # Remove existing model
        del arena.portfolios[model_id]
        if model_id in arena.loaded_models:
            del arena.loaded_models[model_id]
        print(f"{Colors.GREEN}Removed existing {model_id} from arena{Colors.END}")

    # Add model to arena
    print(f"{Colors.BLUE}Deploying {model_id} to arena...{Colors.END}")
    success = arena.add_model(trial_dir, args.trial, args.value)

    if success:
        print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
        print(f"{Colors.GREEN}✓ DEPLOYMENT SUCCESSFUL!{Colors.END}")
        print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")
        print(f"Model {model_id} is now live in the arena!")
        print(f"  Training value: {args.value:.6f}")
        print(f"  Initial capital: $1000")
        print(f"  Status: EVAL (needs {arena.min_evaluation_hours}h for promotion eligibility)")
        print()
        print(f"Monitor performance:")
        print(f"  python model_arena.py --mode status")
        print(f"  tail -f arena_state/leaderboard.txt")
        print()
    else:
        print(f"\n{Colors.RED}{'='*80}{Colors.END}")
        print(f"{Colors.RED}✗ DEPLOYMENT FAILED{Colors.END}")
        print(f"{Colors.RED}{'='*80}{Colors.END}\n")
        print(f"Check logs for details.")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
