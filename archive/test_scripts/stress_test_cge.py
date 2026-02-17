#!/usr/bin/env python3
"""
Generic CGE Stress Test Module
Provides reusable function for running CGE stress tests on any model.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pickle
import torch

# Import Cappuccino components
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from function_finance_metrics import sharpe_iid, compute_eqw


def run_cge_stress_test(model_dir: str, trial_number: int, num_scenarios: int = 200,
                       output_file: str = None, gpu_id: int = -1) -> pd.DataFrame:
    """
    Run CGE stress test on a model.

    Args:
        model_dir: Path to model directory containing actor.pth
        trial_number: Trial number for logging
        num_scenarios: Number of CGE scenarios to test
        output_file: Path to save CSV results (optional)
        gpu_id: GPU ID to use (-1 for CPU)

    Returns:
        DataFrame with results for each scenario
    """
    print(f"\n{'='*70}")
    print(f"CGE STRESS TEST - Trial {trial_number}")
    print(f"{'='*70}")
    print(f"Model: {model_dir}")
    print(f"Scenarios: {num_scenarios}")

    # Check model exists
    model_path = Path(model_dir)
    actor_file = model_path / "actor.pth"
    if not actor_file.exists():
        print(f"ERROR: Model file not found: {actor_file}")
        return None

    # Load trial parameters
    params = load_trial_params(model_dir, trial_number)
    if not params:
        print(f"WARNING: Could not load trial params, using defaults")
        params = get_default_params()

    # Setup environment parameters
    env_params = {
        'min_cash_reserve': params.get('min_cash_reserve', 0.1),
        'concentration_penalty': params.get('concentration_penalty', 0.05),
        'max_drawdown_penalty': params.get('max_drawdown_penalty', 0.0),
        'volatility_penalty': params.get('volatility_penalty', 0.0),
        'trailing_stop_pct': params.get('trailing_stop_pct', 0.0),
        'norm_cash': 2 ** params.get('norm_cash_exp', -11),
        'norm_stocks': 2 ** params.get('norm_stocks_exp', -8),
        'norm_tech': 2 ** params.get('norm_tech_exp', -14),
        'norm_reward': 2 ** params.get('norm_reward_exp', -9),
        'norm_action': params.get('norm_action', 100),
        'time_decay_floor': params.get('time_decay_floor', 0.0),
        'lookback': params.get('lookback', 60),
    }

    # Load CGE scenarios
    cge_data_path = Path('data/cge_synthetic')
    if not cge_data_path.exists():
        print(f"ERROR: CGE data path not found: {cge_data_path}")
        return None

    print(f"\nLoading {num_scenarios} CGE scenarios...")
    scenarios = load_cge_scenarios(cge_data_path, num_scenarios)
    if not scenarios:
        print("ERROR: Failed to load scenarios")
        return None

    print(f"✓ Loaded {len(scenarios)} scenarios")

    # Load model
    print("\nLoading model...")
    agent = load_model(model_dir, params, gpu_id)
    if not agent:
        print("ERROR: Failed to load model")
        return None

    print("✓ Model loaded successfully")

    # Run stress tests
    print(f"\nRunning stress tests...")
    results = []

    for i, scenario in enumerate(scenarios):
        if (i + 1) % 20 == 0:
            print(f"Progress: {i+1}/{len(scenarios)} scenarios completed")

        # Run simulation
        metrics = run_scenario(agent, scenario, env_params)
        if metrics:
            metrics['scenario_id'] = i
            results.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print(f"\n{'='*70}")
    print("STRESS TEST RESULTS")
    print(f"{'='*70}")
    print(f"Total scenarios: {len(df)}")
    print(f"Profitable: {(df['total_return'] > 0).sum()} ({(df['total_return'] > 0).mean()*100:.1f}%)")
    print(f"Median Sharpe: {df['sharpe_ratio'].median():.3f}")
    print(f"Mean return: {df['total_return'].mean()*100:.2f}%")
    print(f"Max drawdown (worst): {df['max_drawdown'].max()*100:.1f}%")
    print(f"Catastrophic failures (<-90%): {(df['total_return'] < -0.9).sum()}")
    print(f"{'='*70}\n")

    # Save results
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"✓ Results saved to {output_file}")

    return df


def load_trial_params(model_dir: str, trial_number: int):
    """Load trial parameters from best_trial file or database."""
    try:
        # Try loading from best_trial pickle
        best_trial_path = Path(model_dir) / "best_trial"
        if best_trial_path.exists():
            with open(best_trial_path, 'rb') as f:
                trial = pickle.load(f)
                return trial.params
    except:
        pass

    # Try loading from database
    try:
        import sqlite3
        conn = sqlite3.connect('databases/optuna_cappuccino.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params tp
            JOIN trials t ON tp.trial_id = t.trial_id
            WHERE t.number = ?
        """, (trial_number,))

        params = {}
        for name, value in cursor.fetchall():
            params[name] = value
        conn.close()

        if params:
            return params
    except:
        pass

    return None


def get_default_params():
    """Get default parameters if trial params can't be loaded."""
    return {
        'lookback': 60,
        'norm_cash_exp': -11,
        'norm_stocks_exp': -8,
        'norm_tech_exp': -14,
        'norm_reward_exp': -9,
        'norm_action': 100,
        'time_decay_floor': 0.0,
        'min_cash_reserve': 0.1,
        'concentration_penalty': 0.05,
        'max_drawdown_penalty': 0.0,
        'volatility_penalty': 0.0,
        'trailing_stop_pct': 0.0,
    }


def load_cge_scenarios(cge_path: Path, num_scenarios: int):
    """Load CGE scenario data."""
    scenarios = []

    for i in range(num_scenarios):
        try:
            prices = np.load(cge_path / f'synthetic_{i:04d}.npy')

            # Load metadata
            meta_file = cge_path / f'synthetic_meta_{i:04d}.json'
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            scenarios.append({
                'id': i,
                'prices': prices,
                'metadata': metadata
            })

        except Exception as e:
            print(f"Warning: Failed to load scenario {i}: {e}")
            continue

    return scenarios


def load_model(model_dir: str, params: dict, gpu_id: int = -1):
    """Load trained model."""
    try:
        from train.run import init_agent
        from train.config import Arguments
        from drl_agents.elegantrl_models import MODELS

        # Create dummy environment for agent initialization
        env_config = {
            "price_array": np.zeros((1000, 7)),
            "tech_array": np.zeros((1000, 7, 44)),
            "if_train": False,
        }

        env = CryptoEnvAlpaca(env_config, params, if_log=False)

        # Initialize agent
        net_dim = int(params.get('net_dimension', 256))
        model_name = "ppo"

        args = Arguments(agent=MODELS[model_name], env=env)
        args.cwd = str(model_dir)
        args.if_remove = False
        args.net_dim = net_dim

        agent = init_agent(args, gpu_id=gpu_id, env=env)
        agent.act.eval()

        return agent

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def run_scenario(agent, scenario, env_params):
    """Run model on a single scenario."""
    try:
        # Create environment with scenario data
        env_config = {
            "price_array": scenario['prices'],
            "tech_array": np.zeros((len(scenario['prices']), 7, 44)),  # Placeholder
            "if_train": False,
        }

        env = CryptoEnvAlpaca(env_config, env_params, if_log=False)

        # Run episode
        state = env.reset()
        done = False
        portfolio_values = [env.initial_total_asset]

        while not done:
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action = agent.act(state_tensor).cpu().numpy()[0]
            state, reward, done, _ = env.step(action)
            portfolio_values.append(env.total_asset)

        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(24 * 365)

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values[-1],
            'num_steps': len(portfolio_values),
        }

    except Exception as e:
        print(f"Error running scenario: {e}")
        return None
