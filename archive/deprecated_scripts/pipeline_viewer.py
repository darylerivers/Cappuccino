#!/usr/bin/env python3
"""
Detailed Pipeline Viewer - Shows trial progress through all pipeline stages
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def get_trial_states():
    """Get all trials and their pipeline states."""
    state_file = Path("pipeline/state/pipeline_state.json")
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {}

def get_trials_from_db(db_path):
    """Get all trials from Optuna database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT t.number, t.state, tv.value, t.datetime_start, t.datetime_complete
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        ORDER BY t.number DESC
    """)

    trials = {}
    for number, state, value, start, end in cursor.fetchall():
        trials[number] = {
            'state': state,
            'value': value,
            'started': start,
            'completed': end
        }

    conn.close()
    return trials

def format_timestamp(ts_str):
    """Format timestamp string."""
    if not ts_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts_str.replace(' ', 'T'))
        return dt.strftime("%H:%M:%S")
    except:
        return ts_str[:8] if len(ts_str) > 8 else ts_str

def get_pipeline_stage(trial_num, pipeline_state):
    """Get current pipeline stage for a trial."""
    trial_str = str(trial_num)
    if trial_str not in pipeline_state:
        return "not_in_pipeline", {}

    trial_data = pipeline_state[trial_str]
    stage = trial_data.get('stage', 'unknown')
    status = trial_data.get('status', 'unknown')

    return f"{stage}:{status}", trial_data

def main():
    # Configuration
    db_path = "/tmp/optuna_working.db"

    # Get data
    trials = get_trials_from_db(db_path)
    pipeline_state = get_trial_states()

    # Header
    print("=" * 120)
    print("PIPELINE VIEWER - Trial Progress Dashboard")
    print("=" * 120)
    print(f"Database: {db_path}")
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    # Stage counts
    stage_counts = defaultdict(int)
    for trial_num in trials.keys():
        stage, _ = get_pipeline_stage(trial_num, pipeline_state)
        if trials[trial_num]['state'] == 'COMPLETE':
            stage_counts[stage] += 1

    print("\nPipeline Stage Summary:")
    print(f"  Not in pipeline: {stage_counts.get('not_in_pipeline', 0)}")
    print(f"  Backtest (pending): {stage_counts.get('backtest:pending', 0)}")
    print(f"  Backtest (passed): {stage_counts.get('backtest:passed', 0)}")
    print(f"  Backtest (failed): {stage_counts.get('backtest:failed', 0)}")
    print(f"  CGE stress (pending): {stage_counts.get('cge_stress:pending', 0)}")
    print(f"  CGE stress (passed): {stage_counts.get('cge_stress:passed', 0)}")
    print(f"  CGE stress (failed): {stage_counts.get('cge_stress:failed', 0)}")
    print(f"  Deployed: {stage_counts.get('deployed:success', 0)}")

    # Detailed trial listing
    print("\n" + "=" * 120)
    print(f"{'Trial':<8} {'Optuna':<10} {'Value':<12} {'Completed':<10} {'Pipeline Stage':<25} {'Last Update':<20}")
    print("=" * 120)

    for trial_num in sorted(trials.keys(), reverse=True):
        trial = trials[trial_num]
        stage, stage_data = get_pipeline_stage(trial_num, pipeline_state)

        # Format values
        optuna_state = trial['state']
        value = f"{trial['value']:.6f}" if trial['value'] is not None else "N/A"
        completed = format_timestamp(trial['completed'])
        last_update = format_timestamp(stage_data.get('last_updated', ''))

        # Color coding (basic)
        stage_display = stage
        if 'passed' in stage:
            stage_display = f"✓ {stage}"
        elif 'failed' in stage:
            stage_display = f"✗ {stage}"
        elif 'pending' in stage:
            stage_display = f"⏳ {stage}"

        print(f"{trial_num:<8} {optuna_state:<10} {value:<12} {completed:<10} {stage_display:<25} {last_update:<20}")

    print("=" * 120)

    # Recent activity
    print("\nRecent Pipeline Activity (last 5 updates):")
    all_updates = []
    for trial_num, trial_data in pipeline_state.items():
        if 'last_updated' in trial_data:
            all_updates.append((trial_num, trial_data))

    all_updates.sort(key=lambda x: x[1].get('last_updated', ''), reverse=True)

    for trial_num, trial_data in all_updates[:5]:
        stage = trial_data.get('stage', 'unknown')
        status = trial_data.get('status', 'unknown')
        timestamp = format_timestamp(trial_data.get('last_updated', ''))
        message = trial_data.get('message', '')

        print(f"  Trial {trial_num}: {stage}/{status} at {timestamp}")
        if message:
            print(f"    └─ {message}")

    print("\n" + "=" * 120)
    print("Legend:")
    print("  ✓ = Passed  ✗ = Failed  ⏳ = Pending")
    print("  Stages: not_in_pipeline → backtest → cge_stress → deployed")
    print("=" * 120)

if __name__ == "__main__":
    main()
