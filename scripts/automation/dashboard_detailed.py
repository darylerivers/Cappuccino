#!/usr/bin/env python3
"""
Detailed Training Dashboard for Evaluation
Comprehensive analytics, visualizations, and insights
"""
import sqlite3
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
import subprocess
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DB_PATH = "databases/optuna_cappuccino.db"
STUDY_NAME = "cappuccino_auto_20260216_0340"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def get_study_stats():
    """Get comprehensive study statistics."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT study_id FROM studies WHERE study_name = ?", (STUDY_NAME,))
    result = c.fetchone()
    if not result:
        return None
    study_id = result[0]

    # Overall stats
    c.execute("""
        SELECT
            COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as completed,
            COUNT(CASE WHEN t.state = 'RUNNING' THEN 1 END) as running,
            COUNT(CASE WHEN t.state = 'FAIL' THEN 1 END) as failed,
            AVG(CASE WHEN t.state = 'COMPLETE' THEN tv.value END) as avg_sharpe,
            MAX(tv.value) as max_sharpe,
            MIN(CASE WHEN t.state = 'COMPLETE' THEN tv.value END) as min_sharpe,
            DATETIME(MIN(t.datetime_start)) as first_trial,
            DATETIME(MAX(CASE WHEN t.state = 'COMPLETE' THEN t.datetime_complete END)) as last_complete
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ?
    """, (study_id,))

    stats = dict(zip(['completed', 'running', 'failed', 'avg_sharpe', 'max_sharpe',
                      'min_sharpe', 'first_trial', 'last_complete'], c.fetchone()))

    # Grade distribution
    c.execute("""
        SELECT
            CASE
                WHEN tv.value >= 0.30 THEN 'S'
                WHEN tv.value >= 0.20 THEN 'A'
                WHEN tv.value >= 0.15 THEN 'B'
                WHEN tv.value >= 0.10 THEN 'C'
                WHEN tv.value >= 0.05 THEN 'D'
                ELSE 'F'
            END as grade,
            COUNT(*) as count
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE'
        GROUP BY grade
        ORDER BY grade
    """, (study_id,))

    stats['grades'] = {row[0]: row[1] for row in c.fetchall()}

    # Top 10 trials
    c.execute("""
        SELECT
            t.number,
            tv.value as sharpe,
            DATETIME(t.datetime_complete) as completed,
            CAST((JULIANDAY(t.datetime_complete) - JULIANDAY(t.datetime_start)) * 24 * 60 AS INT) as duration_min
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE'
        ORDER BY tv.value DESC
        LIMIT 10
    """, (study_id,))

    stats['top_trials'] = [dict(zip(['number', 'sharpe', 'completed', 'duration_min'], row))
                           for row in c.fetchall()]

    # Recent performance (last 20 trials)
    c.execute("""
        SELECT
            AVG(value) as recent_avg,
            MAX(value) as recent_max,
            MIN(value) as recent_min
        FROM (
            SELECT tv.value as value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY t.datetime_complete DESC
            LIMIT 20
        )
    """, (study_id,))

    recent = c.fetchone()
    stats['recent_avg'] = recent[0]
    stats['recent_max'] = recent[1]
    stats['recent_min'] = recent[2]

    # Hyperparameter analysis (top 10%)
    threshold = stats['max_sharpe'] * 0.9 if stats['max_sharpe'] else 0.15

    c.execute("""
        SELECT
            param_name,
            AVG(CAST(param_value AS REAL)) as avg_value,
            MIN(CAST(param_value AS REAL)) as min_value,
            MAX(CAST(param_value AS REAL)) as max_value
        FROM trial_params tp
        JOIN trials t ON tp.trial_id = t.trial_id
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value >= ?
          AND param_name IN ('batch_size', 'learning_rate', 'gamma', 'net_dimension',
                             'lookback', 'worker_num', 'base_target_step', 'base_break_step')
        GROUP BY param_name
        ORDER BY param_name
    """, (study_id, threshold))

    stats['best_hyperparams'] = {row[0]: {
        'avg': row[1], 'min': row[2], 'max': row[3]
    } for row in c.fetchall()}

    conn.close()
    return stats

def print_ascii_histogram(data, width=50, title=""):
    """Print ASCII histogram."""
    if not data:
        return

    max_val = max(data.values())
    if max_val == 0:
        return

    print(f"\n{Colors.CYAN}{title}{Colors.END}")
    for label, count in sorted(data.items()):
        bar_len = int((count / max_val) * width)
        bar = '█' * bar_len
        pct = (count / sum(data.values())) * 100
        print(f"  {label}: {bar} {count} ({pct:.1f}%)")

def print_sharpe_distribution(stats):
    """Print Sharpe ratio distribution."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT study_id FROM studies WHERE study_name = ?", (STUDY_NAME,))
    study_id = c.fetchone()[0]

    # Get Sharpe distribution in buckets
    c.execute("""
        SELECT
            CASE
                WHEN tv.value < 0.05 THEN '< 0.05'
                WHEN tv.value < 0.10 THEN '0.05-0.10'
                WHEN tv.value < 0.15 THEN '0.10-0.15'
                WHEN tv.value < 0.17 THEN '0.15-0.17'
                WHEN tv.value < 0.18 THEN '0.17-0.18'
                ELSE '>= 0.18'
            END as bucket,
            COUNT(*) as count
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE'
        GROUP BY bucket
        ORDER BY bucket
    """, (study_id,))

    buckets = {row[0]: row[1] for row in c.fetchall()}
    conn.close()

    print_ascii_histogram(buckets, width=40, title="📊 Sharpe Ratio Distribution")

def clear_screen():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def print_header(refresh_interval=None):
    """Print dashboard header."""
    print("\n" + "="*80)
    print(f"{Colors.BOLD}{Colors.HEADER}        CAPPUCCINO DETAILED TRAINING DASHBOARD{Colors.END}")
    print(f"{Colors.CYAN}                  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    if refresh_interval:
        print(f"{Colors.YELLOW}                  Auto-refresh: every {refresh_interval}s (Ctrl+C to exit){Colors.END}")
    print("="*80)

def print_system_status():
    """Print system resource status."""
    try:
        # RAM
        result = subprocess.run(['free', '-h'], capture_output=True, text=True)
        ram_line = [l for l in result.stdout.split('\n') if 'Mem:' in l][0].split()
        ram_used, ram_total = ram_line[2], ram_line[1]

        # GPU
        result = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True)
        gpu_line = [l for l in result.stdout.split('\n') if 'GPU use' in l]
        gpu_pct = gpu_line[0].split()[-1] if gpu_line else 'N/A'

        # Workers
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        workers = len([l for l in result.stdout.split('\n') if '1_optimize_unified' in l and 'grep' not in l])

        print(f"\n{Colors.BOLD}💻 System Resources:{Colors.END}")
        print(f"   RAM: {ram_used} / {ram_total}  |  GPU: {gpu_pct}%  |  Workers: {workers}")
    except:
        pass

def print_study_overview(stats):
    """Print study overview."""
    print(f"\n{Colors.BOLD}📊 Study Overview: {STUDY_NAME}{Colors.END}")
    print(f"   Trials: {stats['completed']} completed, {stats['running']} running, {stats['failed']} failed")
    print(f"   Duration: {stats['first_trial']} → {stats['last_complete']}")

    if stats['completed'] > 0:
        print(f"\n{Colors.BOLD}📈 Performance Metrics:{Colors.END}")
        print(f"   Overall:  Avg {stats['avg_sharpe']:.4f}  |  Max {stats['max_sharpe']:.4f}  |  Min {stats['min_sharpe']:.4f}")
        print(f"   Recent:   Avg {stats['recent_avg']:.4f}  |  Max {stats['recent_max']:.4f}  |  Min {stats['recent_min']:.4f}")

        # Trend indicator
        if stats['recent_avg'] > stats['avg_sharpe']:
            trend = f"{Colors.GREEN}↑ Improving{Colors.END}"
        elif stats['recent_avg'] < stats['avg_sharpe']:
            trend = f"{Colors.RED}↓ Declining{Colors.END}"
        else:
            trend = f"{Colors.YELLOW}→ Stable{Colors.END}"
        print(f"   Trend: {trend} (recent vs overall)")

def print_grade_distribution(stats):
    """Print grade distribution with visualization."""
    grades = stats['grades']
    total = sum(grades.values())

    print(f"\n{Colors.BOLD}🎯 Grade Distribution:{Colors.END}")

    grade_info = {
        'S': ('≥0.30', Colors.HEADER),
        'A': ('≥0.20', Colors.GREEN),
        'B': ('≥0.15', Colors.CYAN),
        'C': ('≥0.10', Colors.YELLOW),
        'D': ('≥0.05', Colors.YELLOW),
        'F': ('<0.05', Colors.RED)
    }

    for grade in ['S', 'A', 'B', 'C', 'D', 'F']:
        count = grades.get(grade, 0)
        pct = (count / total * 100) if total > 0 else 0
        threshold, color = grade_info[grade]
        bar_len = int(pct / 2)  # 50 chars = 100%
        bar = '█' * bar_len
        print(f"   {color}{grade}{Colors.END} ({threshold}): {bar} {count} ({pct:.1f}%)")

def print_top_trials(stats):
    """Print top performing trials."""
    print(f"\n{Colors.BOLD}🏆 Top 10 Trials:{Colors.END}")
    print(f"   {'Rank':<6} {'Trial':<8} {'Sharpe':<10} {'Duration':<12} {'Completed'}")
    print(f"   {'-'*70}")

    for i, trial in enumerate(stats['top_trials'], 1):
        grade_color = Colors.GREEN if trial['sharpe'] >= 0.20 else Colors.CYAN if trial['sharpe'] >= 0.15 else Colors.YELLOW
        medal = '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '  '

        duration_str = f"{trial['duration_min']//60}h {trial['duration_min']%60}m" if trial['duration_min'] else 'N/A'
        completed_str = trial['completed'].split()[1] if trial['completed'] else 'N/A'

        print(f"   {medal} #{i:<3} #{trial['number']:<7} {grade_color}{trial['sharpe']:.4f}{Colors.END}     "
              f"{duration_str:<12} {completed_str}")

def print_hyperparameter_analysis(stats):
    """Print hyperparameter analysis for top trials."""
    if not stats['best_hyperparams']:
        return

    print(f"\n{Colors.BOLD}🔬 Optimal Hyperparameters (Top 10% Trials):{Colors.END}")
    print(f"   {'Parameter':<20} {'Average':<15} {'Range'}")
    print(f"   {'-'*70}")

    param_display = {
        'batch_size': 'Batch Size',
        'learning_rate': 'Learning Rate',
        'gamma': 'Gamma (Discount)',
        'net_dimension': 'Network Dimension',
        'lookback': 'Lookback Period',
        'worker_num': 'Worker Processes',
        'base_target_step': 'Target Steps',
        'base_break_step': 'Break Steps'
    }

    for param, values in sorted(stats['best_hyperparams'].items()):
        name = param_display.get(param, param)
        avg = values['avg']
        min_val = values['min']
        max_val = values['max']

        if param == 'learning_rate':
            print(f"   {name:<20} {avg:.2e}         {min_val:.2e} - {max_val:.2e}")
        elif param in ['gamma']:
            print(f"   {name:<20} {avg:.4f}          {min_val:.4f} - {max_val:.4f}")
        else:
            print(f"   {name:<20} {avg:.1f}           {min_val:.0f} - {max_val:.0f}")

def print_training_velocity(stats):
    """Print training velocity and efficiency."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT study_id FROM studies WHERE study_name = ?", (STUDY_NAME,))
    study_id = c.fetchone()[0]

    # Calculate trials per hour
    c.execute("""
        SELECT
            CAST((JULIANDAY(MAX(datetime_complete)) - JULIANDAY(MIN(datetime_start))) * 24 AS REAL) as hours,
            COUNT(*) as trials
        FROM trials
        WHERE study_id = ? AND state = 'COMPLETE'
    """, (study_id,))

    result = c.fetchone()
    if result[0] and result[0] > 0:
        hours = result[0]
        trials = result[1]
        rate = trials / hours

        print(f"\n{Colors.BOLD}⚡ Training Velocity:{Colors.END}")
        print(f"   Total Runtime: {hours:.1f} hours")
        print(f"   Completion Rate: {rate:.1f} trials/hour")
        print(f"   Avg Trial Duration: {60/rate:.1f} minutes/trial")

    # Average duration by state
    c.execute("""
        SELECT
            AVG(CAST((JULIANDAY(datetime_complete) - JULIANDAY(datetime_start)) * 24 * 60 AS REAL)) as avg_min
        FROM trials
        WHERE study_id = ? AND state = 'COMPLETE'
    """, (study_id,))

    avg_duration = c.fetchone()[0]
    if avg_duration:
        print(f"   Measured Avg: {avg_duration:.1f} min/trial")

    conn.close()

def print_convergence_analysis(stats):
    """Analyze convergence trend."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT study_id FROM studies WHERE study_name = ?", (STUDY_NAME,))
    study_id = c.fetchone()[0]

    # Get performance over time (in chunks of 20 trials)
    c.execute("""
        SELECT
            CAST(t.number / 20 AS INT) as chunk,
            AVG(tv.value) as avg_sharpe,
            MAX(tv.value) as max_sharpe
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE'
        GROUP BY chunk
        ORDER BY chunk
    """, (study_id,))

    chunks = list(c.fetchall())
    conn.close()

    if len(chunks) >= 2:
        print(f"\n{Colors.BOLD}📉 Convergence Analysis (20-trial windows):{Colors.END}")

        # Show first, middle, last
        indices = [0, len(chunks)//2, -1] if len(chunks) > 2 else [0, -1]

        for i in indices:
            chunk_num = chunks[i][0]
            avg = chunks[i][1]
            max_val = chunks[i][2]
            trial_range = f"#{chunk_num*20}-{(chunk_num+1)*20}"
            label = "Early" if i == 0 else "Mid" if i == len(chunks)//2 else "Recent"
            print(f"   {label:8} {trial_range:12} Avg: {avg:.4f}  Max: {max_val:.4f}")

        # Trend
        early_avg = chunks[0][1]
        recent_avg = chunks[-1][1]
        improvement = ((recent_avg - early_avg) / early_avg * 100) if early_avg > 0 else 0

        if improvement > 5:
            print(f"\n   {Colors.GREEN}✓ Strong improvement: +{improvement:.1f}%{Colors.END}")
        elif improvement > 0:
            print(f"\n   {Colors.CYAN}→ Slight improvement: +{improvement:.1f}%{Colors.END}")
        else:
            print(f"\n   {Colors.YELLOW}→ Converged (±{abs(improvement):.1f}%){Colors.END}")

def display_dashboard(refresh_interval=None):
    """Display the dashboard once."""
    if refresh_interval:
        clear_screen()

    print_header(refresh_interval)
    print_system_status()

    stats = get_study_stats()
    if not stats:
        print(f"\n{Colors.RED}Error: Study '{STUDY_NAME}' not found{Colors.END}\n")
        return False

    print_study_overview(stats)
    print_grade_distribution(stats)
    print_sharpe_distribution(stats)
    print_top_trials(stats)
    print_hyperparameter_analysis(stats)
    print_training_velocity(stats)
    print_convergence_analysis(stats)

    print("\n" + "="*80)
    print(f"{Colors.CYAN}Commands:{Colors.END}")
    if refresh_interval:
        print(f"  Press Ctrl+C to exit auto-refresh mode")
    else:
        print(f"  python scripts/automation/dashboard_detailed.py --loop      # Auto-refresh mode")
        print(f"  python scripts/automation/dashboard_detailed.py --refresh N # Auto-refresh every N seconds")
        print(f"  python scripts/automation/trial_dashboard.py                # Live dashboard")
        print(f"  python monitor_progress.py                                  # Progress bars")
    print("="*80 + "\n")

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Detailed Training Dashboard with optional auto-refresh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dashboard_detailed.py              # One-time view
  python dashboard_detailed.py --loop       # Auto-refresh every 30s
  python dashboard_detailed.py --refresh 10 # Auto-refresh every 10s
        """
    )
    parser.add_argument('--loop', '-l', action='store_true',
                       help='Enable auto-refresh mode (default: 30s interval)')
    parser.add_argument('--refresh', '-r', type=int, metavar='SECONDS',
                       help='Auto-refresh interval in seconds (implies --loop)')

    args = parser.parse_args()

    # Determine refresh interval
    if args.refresh:
        refresh_interval = args.refresh
    elif args.loop:
        refresh_interval = 30
    else:
        refresh_interval = None

    if refresh_interval:
        # Auto-refresh mode
        print(f"{Colors.CYAN}Starting auto-refresh mode (every {refresh_interval}s)...{Colors.END}")
        print(f"{Colors.YELLOW}Press Ctrl+C to exit{Colors.END}\n")
        time.sleep(2)

        try:
            while True:
                if not display_dashboard(refresh_interval):
                    break

                # Countdown timer
                for remaining in range(refresh_interval, 0, -1):
                    sys.stdout.write(f"\r{Colors.CYAN}Next refresh in {remaining}s...{Colors.END} ")
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the line

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Auto-refresh stopped by user{Colors.END}\n")
    else:
        # One-time display
        display_dashboard()

if __name__ == "__main__":
    main()
