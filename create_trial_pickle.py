#!/usr/bin/env python3
"""Create a properly picklable trial object for deployment."""

import pickle
import sqlite3
import sys
import json
from pathlib import Path


class DeploymentTrial:
    """Picklable trial object for paper trader."""
    def __init__(self, trial_data):
        self.number = trial_data['number']
        self.params = trial_data['params']
        self.user_attrs = trial_data['user_attrs']
        self.value = trial_data['sharpe']


def get_trial_from_optuna(db_path, study_name, trial_number):
    """Load trial parameters from Optuna database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    result = cursor.fetchone()
    if not result:
        raise ValueError(f"Study '{study_name}' not found")
    study_id = result[0]

    cursor.execute("""
        SELECT t.trial_id, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.number = ?
    """, (study_id, trial_number))

    trial_data = cursor.fetchone()
    if not trial_data:
        raise ValueError(f"Trial #{trial_number} not found")

    trial_id, sharpe = trial_data

    cursor.execute("SELECT param_name, param_value FROM trial_params WHERE trial_id = ?", (trial_id,))

    params = {}
    for name, value in cursor.fetchall():
        if name in ['lookback', 'net_dimension', 'worker_num', 'thread_num', 'batch_size',
                    'base_target_step', 'base_break_step', 'ppo_epochs', 'eval_time_gap']:
            params[name] = int(value)
        elif name in ['use_ft_encoder', 'ft_use_pretrained', 'ft_freeze_encoder', 'use_lr_schedule']:
            params[name] = bool(int(value))
        else:
            params[name] = float(value)

    cursor.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?", (trial_id,))

    user_attrs = {}
    for key, value_json in cursor.fetchall():
        try:
            user_attrs[key] = json.loads(value_json) if value_json else None
        except:
            user_attrs[key] = value_json

    conn.close()

    return {'trial_id': trial_id, 'number': trial_number, 'sharpe': sharpe, 'params': params, 'user_attrs': user_attrs}


if __name__ == '__main__':
    trial_num = 250
    output_dir = Path('train_results/deployment_trial250_20260207_175728')
    db_path = 'databases/optuna_cappuccino.db'

    print(f"Loading Trial #{trial_num}...")
    trial_data = get_trial_from_optuna(db_path, 'cappuccino_ft_transformer', trial_num)
    print(f"  Sharpe: {trial_data['sharpe']:.6f}")

    trial_obj = DeploymentTrial(trial_data)

    output_file = output_dir / 'best_trial'
    with open(output_file, 'wb') as f:
        pickle.dump(trial_obj, f)

    print(f"✓ Saved to: {output_file}")

    with open(output_file, 'rb') as f:
        loaded = pickle.load(f)
    print(f"✓ Verified: Trial #{loaded.number}, Sharpe {loaded.value:.6f}")
