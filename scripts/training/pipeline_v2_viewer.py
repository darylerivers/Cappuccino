#!/usr/bin/env python3
"""
Pipeline V2 Viewer - Shows trial progress through Pipeline V2 stages
"""
import sqlite3
from datetime import datetime
from collections import defaultdict
from path_detector import PathDetector

def get_trials_from_optuna(db_path):
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

def get_pipeline_status(pipeline_db):
    """Get pipeline status from V2 database."""
    try:
        conn = sqlite3.connect(pipeline_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all trials and their stages
        cursor.execute("""
            SELECT
                t.trial_number,
                t.status as trial_status,
                t.value,
                t.updated_at,
                GROUP_CONCAT(s.stage_name || ':' || s.status) as stages
            FROM trials t
            LEFT JOIN stages s ON t.trial_id = s.trial_id
            GROUP BY t.trial_number
            ORDER BY t.trial_number DESC
        """)

        pipeline_trials = {}
        for row in cursor.fetchall():
            trial_num = row['trial_number']
            pipeline_trials[trial_num] = {
                'status': row['trial_status'],
                'updated_at': row['updated_at'],
                'stages': row['stages'] or ''
            }

        # Get deployment info
        cursor.execute("""
            SELECT
                t.trial_number,
                d.process_id,
                d.status as deploy_status,
                d.started_at
            FROM trials t
            JOIN deployments d ON t.trial_id = d.id
            WHERE d.status = 'running'
        """)

        deployments = {}
        for row in cursor.fetchall():
            deployments[row['trial_number']] = {
                'pid': row['process_id'],
                'status': row['deploy_status'],
                'started': row['started_at']
            }

        conn.close()
        return pipeline_trials, deployments

    except Exception as e:
        return {}, {}

def format_timestamp(ts_str):
    """Format timestamp string."""
    if not ts_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(ts_str.replace(' ', 'T'))
        return dt.strftime("%H:%M:%S")
    except:
        return ts_str[:8] if len(ts_str) > 8 else ts_str

def get_stage_display(trial_num, pipeline_trials):
    """Get display string for pipeline stage."""
    if trial_num not in pipeline_trials:
        return "not_in_pipeline", "N/A"

    data = pipeline_trials[trial_num]
    status = data['status']
    stages = data['stages']

    # Determine display
    if status == 'deployed':
        return 'deployed', format_timestamp(data['updated_at'])
    elif status == 'failed':
        return 'failed', format_timestamp(data['updated_at'])
    elif 'backtest:passed' in stages:
        return 'backtest:passed', format_timestamp(data['updated_at'])
    elif 'backtest:failed' in stages:
        return 'backtest:failed', format_timestamp(data['updated_at'])
    elif 'backtest:pending' in stages or 'backtest:running' in stages:
        return 'backtest:pending', format_timestamp(data['updated_at'])
    else:
        return status, format_timestamp(data['updated_at'])

def main():
    # Auto-detect database paths
    detector = PathDetector()
    optuna_db = detector.find_optuna_db()
    pipeline_db = detector.find_pipeline_db()

    # Get data
    optuna_trials = get_trials_from_optuna(optuna_db)
    pipeline_trials, deployments = get_pipeline_status(pipeline_db)

    # Header
    print("=" * 120)
    print("PIPELINE V2 VIEWER - Trial Progress Dashboard")
    print("=" * 120)
    print(f"Optuna DB:   {optuna_db}")
    print(f"Pipeline DB: {pipeline_db}")
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    # Stage counts
    stage_counts = defaultdict(int)
    for trial_num in optuna_trials.keys():
        if optuna_trials[trial_num]['state'] == 'COMPLETE':
            stage, _ = get_stage_display(trial_num, pipeline_trials)
            stage_counts[stage] += 1

    print("\nPipeline Stage Summary:")
    print(f"  Not in pipeline:     {stage_counts.get('not_in_pipeline', 0)}")
    print(f"  Pending:             {stage_counts.get('pending', 0)}")
    print(f"  Processing:          {stage_counts.get('processing', 0)}")
    print(f"  Backtest pending:    {stage_counts.get('backtest:pending', 0)}")
    print(f"  Backtest passed:     {stage_counts.get('backtest:passed', 0)}")
    print(f"  Backtest failed:     {stage_counts.get('backtest:failed', 0)}")
    print(f"  Deployed:            {stage_counts.get('deployed', 0)}")
    print(f"  Failed:              {stage_counts.get('failed', 0)}")

    # Active deployments
    if deployments:
        print(f"\n  Active paper traders: {len(deployments)}")
        for trial_num, deploy_info in sorted(deployments.items()):
            print(f"    Trial {trial_num}: PID {deploy_info['pid']}")

    # Detailed trial listing
    print("\n" + "=" * 120)
    print(f"{'Trial':<8} {'Optuna':<10} {'Value':<12} {'Completed':<10} {'Pipeline Stage':<25} {'Last Update':<20}")
    print("=" * 120)

    for trial_num in sorted(optuna_trials.keys(), reverse=True):
        trial = optuna_trials[trial_num]
        stage, last_update = get_stage_display(trial_num, pipeline_trials)

        # Format values
        optuna_state = trial['state']
        value = f"{trial['value']:.6f}" if trial['value'] is not None else "N/A"
        completed = format_timestamp(trial['completed'])

        # Color coding
        stage_display = stage
        if stage == 'deployed':
            stage_display = f"✓ {stage}"
        elif 'passed' in stage:
            stage_display = f"✓ {stage}"
        elif 'failed' in stage or stage == 'failed':
            stage_display = f"✗ {stage}"
        elif 'pending' in stage or stage == 'processing':
            stage_display = f"⏳ {stage}"

        # Add deployment indicator
        if trial_num in deployments:
            stage_display += f" (PID {deployments[trial_num]['pid']})"

        print(f"{trial_num:<8} {optuna_state:<10} {value:<12} {completed:<10} {stage_display:<40} {last_update:<20}")

    print("=" * 120)

    # Recent activity
    print("\nRecent Pipeline Activity (last 5 updates):")
    recent = sorted(
        [(num, data) for num, data in pipeline_trials.items()],
        key=lambda x: x[1]['updated_at'] or '',
        reverse=True
    )[:5]

    if recent:
        for trial_num, data in recent:
            status = data['status']
            timestamp = format_timestamp(data['updated_at'])
            print(f"  Trial {trial_num}: {status} at {timestamp}")
    else:
        print("  No recent activity")

    print("\n" + "=" * 120)
    print("Legend:")
    print("  ✓ = Passed/Deployed  ✗ = Failed  ⏳ = Pending/Processing")
    print("  Stages: not_in_pipeline → backtest → deployed")
    print("=" * 120)

if __name__ == "__main__":
    main()
