#!/usr/bin/env python3
"""
Training Dashboard - Comprehensive Real-time Monitoring

Shows:
- Optuna study progress and best trials
- FT-Transformer vs Baseline comparison
- Worker status and resource usage
- Recent trial performance
- GPU utilization
"""

import sqlite3
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

# Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def run_cmd(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        return result.stdout.strip()
    except:
        return ""


def get_optuna_stats(db_path, study_name):
    """Get Optuna study statistics from database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study_id
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
        study_id = result[0]

        # Get trial statistics (join with trial_values for objective values)
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN t.state = 'COMPLETE' THEN 1 ELSE 0 END) as complete,
                SUM(CASE WHEN t.state = 'RUNNING' THEN 1 ELSE 0 END) as running,
                SUM(CASE WHEN t.state = 'FAIL' THEN 1 ELSE 0 END) as failed,
                MAX(tv.value) as best_value,
                AVG(tv.value) as avg_value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ?
        """, (study_id,))

        total, complete, running, failed, best_value, avg_value = cursor.fetchone()

        # Get best trials with their parameters
        cursor.execute("""
            SELECT t.number, tv.value, t.trial_id
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value IS NOT NULL
            ORDER BY tv.value DESC
            LIMIT 5
        """, (study_id,))

        best_trials_raw = cursor.fetchall()

        # Fetch parameters for each best trial
        best_trials = []
        for trial_num, value, trial_id in best_trials_raw:
            # Get parameters for this trial
            cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ?
            """, (trial_id,))
            params_raw = cursor.fetchall()

            # Convert to dict
            params = {}
            for param_name, param_value in params_raw:
                # Try to determine if it's a boolean or number
                if param_name in ['use_ft_encoder', 'ft_use_pretrained', 'ft_freeze_encoder']:
                    params[param_name] = bool(int(param_value)) if param_value is not None else False
                else:
                    params[param_name] = param_value

            # Store as JSON string for compatibility with existing code
            best_trials.append((trial_num, value, json.dumps(params)))

        # Count FT-Transformer trials
        cursor.execute("""
            SELECT COUNT(DISTINCT t.trial_id)
            FROM trials t
            JOIN trial_params tp ON t.trial_id = tp.trial_id
            WHERE t.study_id = ? AND tp.param_name = 'use_ft_encoder' AND tp.param_value = 1.0
        """, (study_id,))
        ft_count = cursor.fetchone()[0]

        conn.close()

        return {
            'total': total or 0,
            'complete': complete or 0,
            'running': running or 0,
            'failed': failed or 0,
            'best_value': best_value,
            'avg_value': avg_value,
            'best_trials': best_trials,
            'ft_count': ft_count,
            'baseline_count': (total or 0) - ft_count
        }
    except Exception as e:
        print(f"Error querying database: {e}")
        return None


def get_worker_status():
    """Get status of training workers."""
    output = run_cmd("ps aux | grep '1_optimize_unified.py' | grep -v grep")
    if not output:
        return []

    workers = []
    for line in output.split('\n'):
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 11:
            workers.append({
                'pid': parts[1],
                'cpu': parts[2],
                'mem': parts[3],
                'mem_mb': int(float(parts[5]) / 1024),
                'time': parts[9]
            })
    return workers


def get_gpu_status():
    """Get GPU utilization."""
    output = run_cmd("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits")
    if output:
        parts = output.split(',')
        if len(parts) >= 4:
            return {
                'util': parts[0].strip(),
                'mem_used': parts[1].strip(),
                'mem_total': parts[2].strip(),
                'temp': parts[3].strip()
            }
    return None


def get_recent_trials_from_logs(log_dir, n=10):
    """Get recent trial information from logs."""
    recent_trials = []

    log_files = sorted(Path(log_dir).glob("training_worker_*_ft_transformer.log"))

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Find trial numbers
            import re
            trials = re.findall(r'Trial #(\d+):', content)

            # Check if FT-Transformer is mentioned
            ft_trials = set()
            for match in re.finditer(r'Trial #(\d+):', content):
                trial_num = match.group(1)
                # Check if "Using FT-Transformer" appears after this trial number
                pos = match.end()
                next_content = content[pos:pos+5000]
                if 'Using FT-Transformer' in next_content:
                    ft_trials.add(trial_num)

            if trials:
                for trial_num in trials[-n:]:
                    recent_trials.append({
                        'number': int(trial_num),
                        'worker': log_file.name,
                        'ft': trial_num in ft_trials
                    })
        except:
            pass

    return sorted(recent_trials, key=lambda x: x['number'], reverse=True)[:n]


def print_dashboard(study_name, db_path, log_dir):
    """Print comprehensive dashboard."""

    # Clear screen (optional)
    # print("\033[2J\033[H")

    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}TRAINING DASHBOARD - {study_name}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")
    print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Optuna Statistics
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}OPTUNA STUDY STATISTICS{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")

    stats = get_optuna_stats(db_path, study_name)
    if stats:
        print(f"Total Trials:      {stats['total']}")
        print(f"  ├─ Complete:     {Colors.GREEN}{stats['complete']}{Colors.END}")
        print(f"  ├─ Running:      {Colors.YELLOW}{stats['running']}{Colors.END}")
        print(f"  └─ Failed:       {Colors.RED}{stats['failed']}{Colors.END}")
        print()
        print(f"Trial Distribution:")
        print(f"  ├─ FT-Transformer:  {Colors.CYAN}{stats['ft_count']}{Colors.END} ({stats['ft_count']/max(stats['total'],1)*100:.1f}%)")
        print(f"  └─ Baseline (MLP):  {stats['baseline_count']} ({stats['baseline_count']/max(stats['total'],1)*100:.1f}%)")
        print()
        if stats['best_value'] is not None:
            print(f"Best Sharpe Ratio: {Colors.GREEN}{stats['best_value']:.6f}{Colors.END}")
        if stats['avg_value'] is not None:
            print(f"Avg Sharpe Ratio:  {stats['avg_value']:.6f}")
    else:
        print(f"{Colors.YELLOW}No study data available yet{Colors.END}")

    print()

    # Best Trials
    if stats and stats['best_trials']:
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}{Colors.BOLD}TOP 5 TRIALS{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{'Trial':<8} {'Sharpe':<12} {'Type':<15} {'Key Params'}")
        print(f"{'-'*80}")

        for trial_num, value, params_json in stats['best_trials']:
            try:
                params = json.loads(params_json)
                use_ft = params.get('use_ft_encoder', False)
                trial_type = f"{Colors.CYAN}FT-Trans{Colors.END}" if use_ft else "Baseline"

                # Extract key params
                lr = params.get('learning_rate', 0)
                net_dim = params.get('net_dimension', 0)
                lookback = params.get('lookback', 0)

                key_params = f"LR:{lr:.2e} NetDim:{net_dim} LB:{lookback}"

                print(f"#{trial_num:<7} {value:<12.6f} {trial_type:<22} {key_params}")
            except:
                print(f"#{trial_num:<7} {value:<12.6f} {Colors.YELLOW}(parsing error){Colors.END}")

        print()

    # Worker Status
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}TRAINING WORKERS{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")

    workers = get_worker_status()
    if workers:
        print(f"Active Workers: {len(workers)}")
        print(f"\n{'PID':<10} {'CPU%':<8} {'RAM (MB)':<12} {'Runtime':<10}")
        print(f"{'-'*80}")
        for w in workers:
            cpu_color = Colors.GREEN if float(w['cpu']) > 50 else Colors.YELLOW
            print(f"{w['pid']:<10} {cpu_color}{w['cpu']+'%':<8}{Colors.END} {w['mem_mb']:<12} {w['time']:<10}")
    else:
        print(f"{Colors.RED}No workers running{Colors.END}")

    print()

    # GPU Status
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}GPU STATUS{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")

    gpu = get_gpu_status()
    if gpu:
        util_color = Colors.GREEN if int(gpu['util']) > 80 else Colors.YELLOW
        mem_pct = int(gpu['mem_used']) / int(gpu['mem_total']) * 100
        mem_color = Colors.YELLOW if mem_pct > 80 else Colors.GREEN
        temp_color = Colors.GREEN if int(gpu['temp']) < 75 else Colors.YELLOW

        print(f"Utilization:  {util_color}{gpu['util']}%{Colors.END}")
        print(f"VRAM Usage:   {mem_color}{gpu['mem_used']} MB / {gpu['mem_total']} MB{Colors.END} ({mem_pct:.1f}%)")
        print(f"Temperature:  {temp_color}{gpu['temp']}°C{Colors.END}")
    else:
        print(f"{Colors.YELLOW}GPU info not available{Colors.END}")

    print()

    # Recent Trials
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}RECENT TRIALS (Last 10){Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")

    recent = get_recent_trials_from_logs(log_dir)
    if recent:
        print(f"{'Trial':<10} {'Type':<15} {'Worker'}")
        print(f"{'-'*80}")
        for trial in recent[:10]:
            trial_type = f"{Colors.CYAN}FT-Trans{Colors.END}" if trial['ft'] else "Baseline"
            print(f"#{trial['number']:<9} {trial_type:<22} {trial['worker']}")
    else:
        print(f"{Colors.YELLOW}No trial information available in logs{Colors.END}")

    print()

    # Paper Trader Status
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}PAPER TRADER STATUS{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")

    trader_status = run_cmd("ps aux | grep 'paper_trader' | grep -v grep")
    if trader_status:
        print(f"{Colors.GREEN}✓ Running{Colors.END}")
        lines = trader_status.split('\n')
        if lines:
            parts = lines[0].split()
            if len(parts) >= 10:
                print(f"  PID: {parts[1]}")
                print(f"  CPU: {parts[2]}%")
                print(f"  RAM: {int(float(parts[5])/1024)} MB")
    else:
        print(f"{Colors.RED}✗ Not Running{Colors.END}")

    print()

    # Quick Commands
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}QUICK COMMANDS{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print("Monitor FT trials:  tail -f logs/training_worker_*.log | grep 'FT-Transformer'")
    print("View study in DB:   sqlite3 databases/optuna_cappuccino.db")
    print("Check GPU:          nvidia-smi")
    print("Stop training:      pkill -f '1_optimize_unified.py'")
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Training Dashboard')
    parser.add_argument('--study', type=str, default='cappuccino_ft_transformer',
                        help='Optuna study name')
    parser.add_argument('--db', type=str, default='databases/optuna_cappuccino.db',
                        help='Optuna database path')
    parser.add_argument('--logs', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--watch', action='store_true',
                        help='Refresh every 30 seconds')

    args = parser.parse_args()

    if args.watch:
        import time
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                print_dashboard(args.study, args.db, args.logs)
                print(f"{Colors.YELLOW}Refreshing in 30 seconds... (Ctrl+C to exit){Colors.END}")
                time.sleep(30)
        except KeyboardInterrupt:
            print(f"\n{Colors.GREEN}Dashboard closed.{Colors.END}\n")
    else:
        print_dashboard(args.study, args.db, args.logs)


if __name__ == '__main__':
    main()
