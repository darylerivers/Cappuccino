#!/usr/bin/env python3
"""
Model Inspector - Extract training configuration from saved checkpoints.

This tool analyzes a trained model to determine:
- Number of assets/tickers
- State dimension
- Network architecture
- Which tickers were used during training
"""

import argparse
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import pandas as pd


def inspect_checkpoint(checkpoint_path: Path) -> Dict:
    """Extract configuration from model checkpoint."""
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    info = {}

    # Extract dimensions from actor network
    if 'net.0.weight' in state:
        net_dim = state['net.0.weight'].shape[0]
        state_dim = state['net.0.weight'].shape[1]
        info['net_dimension'] = net_dim
        info['state_dimension'] = state_dim

    # Extract action dimension (number of tickers)
    if 'net.4.weight' in state:
        n_actions = state['net.4.weight'].shape[0]
        info['n_actions'] = n_actions
        info['n_tickers'] = n_actions  # For continuous control, actions = tickers

    # Action standard deviation
    if 'a_std_log' in state:
        info['action_std_shape'] = tuple(state['a_std_log'].shape)

    return info


def infer_config_from_state_dim(state_dim: int, n_tickers: int, n_tech: int = 11) -> Dict:
    """
    Infer training configuration from state dimension.

    State = prices + stocks + tech + cash
    State = n_tickers + n_tickers + (n_tickers * n_tech * lookback) + 1

    Solve for lookback:
    state_dim = 2*n_tickers + (n_tickers * n_tech * lookback) + 1
    lookback = (state_dim - 2*n_tickers - 1) / (n_tickers * n_tech)
    """
    lookback = (state_dim - 2*n_tickers - 1) / (n_tickers * n_tech)

    if not lookback.is_integer() or lookback < 1:
        # Try different n_tech values
        for alt_n_tech in [7, 9, 11, 13, 15]:
            alt_lookback = (state_dim - 2*n_tickers - 1) / (n_tickers * alt_n_tech)
            if alt_lookback.is_integer() and alt_lookback >= 1:
                return {
                    'lookback': int(alt_lookback),
                    'n_tech_indicators': alt_n_tech,
                    'n_tickers': n_tickers,
                    'confidence': 'inferred'
                }

        return {
            'lookback': None,
            'n_tech_indicators': n_tech,
            'n_tickers': n_tickers,
            'confidence': 'failed',
            'error': f'Could not infer integer lookback from state_dim={state_dim}'
        }

    return {
        'lookback': int(lookback),
        'n_tech_indicators': n_tech,
        'n_tickers': n_tickers,
        'confidence': 'high'
    }


def get_tickers_from_trial(trial_pickle: Path, db_path: Optional[Path] = None) -> Optional[List[str]]:
    """Extract ticker list from trial metadata."""
    try:
        with open(trial_pickle, 'rb') as f:
            trial = pickle.load(f)

        # Check user attributes for ticker list
        if 'tickers' in trial.user_attrs:
            return trial.user_attrs['tickers']

        # If not in user attrs, try to find from database
        if db_path and db_path.exists():
            conn = sqlite3.connect(str(db_path))

            # Get study name
            study_name = None
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.study_name
                FROM studies s
                JOIN trials t ON s.study_id = t.study_id
                WHERE t.trial_id = ?
            """, (trial._trial_id,))
            result = cursor.fetchone()
            if result:
                study_name = result[0]

            conn.close()

        return None

    except Exception as e:
        print(f"Warning: Could not extract tickers from trial: {e}")
        return None


def inspect_model(model_dir: Path, db_path: Optional[Path] = None) -> Dict:
    """
    Fully inspect a model directory and extract all training configuration.

    Returns complete config needed to run paper trading.
    """
    config = {}

    # Find checkpoint
    actor_path = model_dir / "stored_agent" / "actor.pth"
    if not actor_path.exists():
        actor_path = model_dir / "actor.pth"

    if not actor_path.exists():
        raise FileNotFoundError(f"No actor checkpoint found in {model_dir}")

    # Inspect checkpoint
    checkpoint_info = inspect_checkpoint(actor_path)
    config.update(checkpoint_info)

    # Try to load trial metadata
    trial_path = model_dir / "best_trial"
    if trial_path.exists():
        try:
            with open(trial_path, 'rb') as f:
                trial = pickle.load(f)

            # Extract trial parameters
            config['trial_params'] = dict(trial.params)
            config['trial_user_attrs'] = dict(trial.user_attrs)

            # Get tickers if available
            tickers = get_tickers_from_trial(trial_path, db_path)
            if tickers:
                config['tickers'] = tickers

        except Exception as e:
            print(f"Warning: Could not load trial metadata: {e}")

    # Infer lookback and other params
    if 'state_dimension' in config and 'n_tickers' in config:
        inferred = infer_config_from_state_dim(
            config['state_dimension'],
            config['n_tickers']
        )
        config['inferred_config'] = inferred

    return config


def print_model_info(config: Dict, verbose: bool = False):
    """Pretty print model configuration."""
    print("="*80)
    print("MODEL CONFIGURATION")
    print("="*80)

    print(f"\n📊 Architecture:")
    print(f"  Network dimension: {config.get('net_dimension', 'Unknown')}")
    print(f"  State dimension:   {config.get('state_dimension', 'Unknown')}")
    print(f"  Action dimension:  {config.get('n_actions', 'Unknown')}")
    print(f"  Number of tickers: {config.get('n_tickers', 'Unknown')}")

    if 'inferred_config' in config:
        inf = config['inferred_config']
        confidence_emoji = "✅" if inf['confidence'] == 'high' else "⚠️"
        print(f"\n🔍 Inferred Configuration ({confidence_emoji} {inf['confidence']}):")
        print(f"  Lookback:          {inf.get('lookback', 'Unknown')}")
        print(f"  Tech indicators:   {inf.get('n_tech_indicators', 'Unknown')}")

        if 'error' in inf:
            print(f"  Error: {inf['error']}")

    if 'tickers' in config:
        print(f"\n💰 Tickers:")
        for i, ticker in enumerate(config['tickers'], 1):
            print(f"  {i}. {ticker}")

    if 'trial_params' in config and verbose:
        print(f"\n⚙️  Trial Parameters:")
        for key, value in sorted(config['trial_params'].items()):
            print(f"  {key:30s} = {value}")

    print("\n" + "="*80)


def generate_paper_trading_command(config: Dict, model_dir: Path) -> str:
    """Generate the correct paper trading command for this model."""

    # Get tickers
    if 'tickers' in config:
        tickers = ' '.join(config['tickers'])
    elif 'trial_user_attrs' in config and 'tickers' in config['trial_user_attrs']:
        tickers = ' '.join(config['trial_user_attrs']['tickers'])
    else:
        # Use default training tickers
        from config_main import TICKER_LIST
        n_tickers = config.get('n_tickers', len(TICKER_LIST))
        tickers = ' '.join(TICKER_LIST[:n_tickers])

    # Get other params
    lookback = config.get('inferred_config', {}).get('lookback', 3)

    cmd = f"""python paper_trader_alpaca_polling.py \\
    --model-dir {model_dir} \\
    --tickers {tickers} \\
    --poll-interval 60 \\
    --history-hours 24"""

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model checkpoint and extract training configuration"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to model directory"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("databases/optuna_cappuccino.db"),
        help="Path to Optuna database"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all trial parameters"
    )
    parser.add_argument(
        "--generate-command",
        action="store_true",
        help="Generate paper trading command"
    )

    args = parser.parse_args()

    # Inspect model
    config = inspect_model(args.model_dir, args.db)

    # Print info
    print_model_info(config, verbose=args.verbose)

    # Generate command if requested
    if args.generate_command:
        print("\n📝 Paper Trading Command:")
        print("="*80)
        print(generate_paper_trading_command(config, args.model_dir))
        print("="*80)

    return config


if __name__ == "__main__":
    main()
