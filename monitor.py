#!/usr/bin/env python3
"""
Real-time Training Monitor for Cappuccino

Usage:
    python monitor.py                    # Show current status
    python monitor.py --watch            # Continuous monitoring
    python monitor.py --study-name prod  # Specific study
"""

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'


def get_gpu_stats():
    """Get GPU statistics."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'utilization': int(stats[0]),
                'memory_used': int(stats[1]),
                'memory_total': int(stats[2]),
                'temperature': int(stats[3]),
                'power': float(stats[4])
            }
    except Exception as e:
        return None
    return None


def get_db_stats(db_path, study_name):
    """Get statistics from Optuna database."""
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study info
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        study_row = cursor.fetchone()
        if not study_row:
            return None

        study_id = study_row[0]

        # Count trials by state
        cursor.execute("""
            SELECT state, COUNT(*)
            FROM trials
            WHERE study_id = ?
            GROUP BY state
        """, (study_id,))

        states = {}
        for state, count in cursor.fetchall():
            states[state] = count

        # Get best trial
        cursor.execute("""
            SELECT t.number, tv.value, t.datetime_complete
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT 1
        """, (study_id,))

        best = cursor.fetchone()

        # Get latest trial
        cursor.execute("""
            SELECT number, state, datetime_start
            FROM trials
            WHERE study_id = ?
            ORDER BY number DESC
            LIMIT 1
        """, (study_id,))

        latest = cursor.fetchone()

        conn.close()

        return {
            'states': states,
            'best_trial': best,
            'latest_trial': latest
        }
    except Exception as e:
        print(f"Error reading database: {e}")
        return None


def get_trial_details(db_path, study_id, trial_number):
    """Get detailed trial information including parameters."""
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get trial params
        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = (
                SELECT trial_id FROM trials
                WHERE study_id = ? AND number = ?
            )
        """, (study_id, trial_number))

        params = {}
        for name, value in cursor.fetchall():
            params[name] = value

        conn.close()
        return params
    except Exception as e:
        return None


def get_all_completed_trials(db_path, study_name):
    """Get all completed trials for analysis."""
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study ID
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        study_row = cursor.fetchone()
        if not study_row:
            return None

        study_id = study_row[0]

        # Get all completed trials
        cursor.execute("""
            SELECT t.number, tv.value, t.datetime_complete
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY t.number ASC
        """, (study_id,))

        trials = []
        for number, value, completed in cursor.fetchall():
            trials.append({
                'number': number,
                'value': value,
                'completed': completed
            })

        conn.close()
        return trials
    except Exception as e:
        return None


def analyze_convergence(trials):
    """Econometric analysis of optimization convergence."""
    if not trials or len(trials) < 5:
        return None

    values = np.array([t['value'] for t in trials])
    trial_numbers = np.array([t['number'] for t in trials])

    # Basic statistics
    stats = {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'range': np.max(values) - np.min(values),
    }

    # Moving averages
    if len(values) >= 5:
        ma_5 = np.convolve(values, np.ones(5)/5, mode='valid')
        stats['ma5_current'] = ma_5[-1] if len(ma_5) > 0 else None
        stats['ma5_trend'] = ma_5[-1] - ma_5[0] if len(ma_5) > 0 else None

    if len(values) >= 10:
        ma_10 = np.convolve(values, np.ones(10)/10, mode='valid')
        stats['ma10_current'] = ma_10[-1] if len(ma_10) > 0 else None
        stats['ma10_trend'] = ma_10[-1] - ma_10[0] if len(ma_10) > 0 else None

    # Volatility (rolling standard deviation)
    if len(values) >= 10:
        recent_10 = values[-10:]
        stats['volatility_recent'] = np.std(recent_10)
        stats['volatility_overall'] = np.std(values)

    # Trend analysis - split into periods
    if len(values) >= 15:
        third = len(values) // 3
        early = values[:third]
        middle = values[third:2*third]
        recent = values[-third:]

        stats['early_mean'] = np.mean(early)
        stats['middle_mean'] = np.mean(middle)
        stats['recent_mean'] = np.mean(recent)
        stats['trend_improvement'] = stats['recent_mean'] - stats['early_mean']

    # Best trial tracking
    best_idx = np.argmax(values)
    stats['best_trial'] = trial_numbers[best_idx]
    stats['best_value'] = values[best_idx]
    stats['trials_since_best'] = len(values) - best_idx - 1

    # Recent performance vs best
    if len(values) >= 10:
        recent_10_mean = np.mean(values[-10:])
        stats['recent_10_mean'] = recent_10_mean
        stats['recent_vs_best'] = recent_10_mean - stats['best_value']

    # Spearman rank correlation (monotonic trend test)
    if SCIPY_AVAILABLE and len(values) >= 10:
        corr, p_value = scipy_stats.spearmanr(trial_numbers, values)
        stats['spearman_rho'] = corr
        stats['spearman_p'] = p_value
        stats['trend_significant'] = p_value < 0.05

    # Autocorrelation (are consecutive trials correlated?)
    if len(values) >= 15:
        # Lag-1 autocorrelation
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
        stats['autocorr_lag1'] = autocorr

    # Stationarity check - Augmented Dickey-Fuller test
    if SCIPY_AVAILABLE and len(values) >= 20:
        try:
            # Simple trend test - compare first half vs second half
            half = len(values) // 2
            first_half = values[:half]
            second_half = values[half:]
            t_stat, p_val = scipy_stats.ttest_ind(first_half, second_half)
            stats['stationarity_p'] = p_val
            stats['mean_shift_detected'] = p_val < 0.05
        except:
            pass

    return stats


def get_recommendation(stats, total_trials):
    """Generate recommendation based on econometric analysis."""
    if not stats:
        return "Insufficient data for analysis", "yellow"

    recommendations = []
    color = "green"

    # Check convergence
    if stats.get('trials_since_best', 0) > 50:
        recommendations.append("⚠️  No improvement in 50+ trials - consider stopping")
        color = "yellow"
    elif stats.get('trials_since_best', 0) > 30:
        recommendations.append("⚠️  Optimization may be plateauing")
        color = "yellow"

    # Check trend
    if 'trend_improvement' in stats:
        if stats['trend_improvement'] < -0.01:
            recommendations.append("📉 Performance declining over time")
            color = "red"
        elif stats['trend_improvement'] > 0.01:
            recommendations.append("📈 Performance improving over time")

    # Check recent performance
    if 'recent_vs_best' in stats:
        if stats['recent_vs_best'] > -0.005:
            recommendations.append("✓ Recent trials competitive with best")
        else:
            recommendations.append("Recent trials underperforming vs best")
            if color == "green":
                color = "yellow"

    # Check volatility
    if 'volatility_recent' in stats and 'volatility_overall' in stats:
        if stats['volatility_recent'] > stats['volatility_overall'] * 1.5:
            recommendations.append("High recent volatility - parameter space may be unstable")

    # Spearman correlation interpretation
    if 'spearman_rho' in stats and stats.get('trend_significant', False):
        if stats['spearman_rho'] > 0.3:
            recommendations.append("✓ Strong positive trend detected (ρ={:.3f})".format(stats['spearman_rho']))
        elif stats['spearman_rho'] < -0.3:
            recommendations.append("⚠️  Negative trend detected (ρ={:.3f})".format(stats['spearman_rho']))
            color = "red"

    # General recommendation
    if not recommendations:
        if total_trials < 50:
            recommendations.append("✓ Continue optimization - more trials needed")
        else:
            recommendations.append("✓ Progress steady - continue optimization")

    return " | ".join(recommendations) if recommendations else "Continue optimization", color


def get_recent_trials(db_path, study_name, n=10):
    """Get last N completed trials with their stats."""
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study ID
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        study_row = cursor.fetchone()
        if not study_row:
            return None

        study_id = study_row[0]

        # Get recent completed trials
        cursor.execute("""
            SELECT t.number, tv.value, t.datetime_complete
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE'
            ORDER BY t.number DESC
            LIMIT ?
        """, (study_id, n))

        trials = []
        for number, value, completed in cursor.fetchall():
            trials.append({
                'number': number,
                'value': value,
                'completed': completed
            })

        conn.close()
        return list(reversed(trials))  # Oldest to newest
    except Exception as e:
        return None


def print_status(db_path='databases/optuna_cappuccino.db', study_name='cappuccino_production'):
    """Print current training status."""

    # Clear screen
    os.system('clear' if os.name != 'nt' else 'cls')

    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}CAPPUCCINO TRAINING MONITOR{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Study: {study_name}\n")

    # GPU Stats
    gpu = get_gpu_stats()
    if gpu:
        print(f"{Colors.CYAN}GPU Status:{Colors.END}")

        # Color code based on utilization
        util_color = Colors.GREEN if gpu['utilization'] > 50 else Colors.YELLOW if gpu['utilization'] > 20 else Colors.RED
        print(f"  Utilization: {util_color}{gpu['utilization']}%{Colors.END}")

        mem_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
        print(f"  Memory: {gpu['memory_used']} / {gpu['memory_total']} MiB ({mem_pct:.1f}%)")

        # Color code temperature
        temp_color = Colors.RED if gpu['temperature'] > 80 else Colors.YELLOW if gpu['temperature'] > 70 else Colors.GREEN
        print(f"  Temperature: {temp_color}{gpu['temperature']}°C{Colors.END}")
        print(f"  Power: {gpu['power']:.1f}W")
    else:
        print(f"{Colors.RED}GPU: Not available{Colors.END}")

    print()

    # Database Stats
    db_stats = get_db_stats(db_path, study_name)
    if db_stats:
        print(f"{Colors.CYAN}Training Progress:{Colors.END}")

        states = db_stats['states']
        total = sum(states.values())
        complete = states.get('COMPLETE', 0)
        running = states.get('RUNNING', 0)
        failed = states.get('FAIL', 0)
        pruned = states.get('PRUNED', 0)

        print(f"  Total Trials: {total}")
        print(f"  {Colors.GREEN}Complete: {complete}{Colors.END}")
        if running > 0:
            print(f"  {Colors.YELLOW}Running: {running}{Colors.END}")
        if failed > 0:
            print(f"  {Colors.RED}Failed: {failed}{Colors.END}")
        if pruned > 0:
            print(f"  Pruned: {pruned}")

        print()

        # Best trial with details
        if db_stats['best_trial']:
            trial_num, value, completed = db_stats['best_trial']
            print(f"{Colors.GREEN}{Colors.BOLD}{'='*80}{Colors.END}")
            print(f"{Colors.GREEN}{Colors.BOLD}BEST TRIAL #{trial_num}{Colors.END}")
            print(f"{Colors.GREEN}{Colors.BOLD}{'='*80}{Colors.END}")
            print(f"  Objective Value (Sharpe Δ): {Colors.GREEN}{value:.6f}{Colors.END}")
            if completed:
                print(f"  Completed: {completed}")

            # Get trial parameters using Optuna
            try:
                if OPTUNA_AVAILABLE:
                    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
                    best_trial = study.best_trial

                    print(f"\n  {Colors.CYAN}Key Parameters:{Colors.END}")
                    important_params = ['learning_rate', 'batch_size', 'gamma', 'net_dimension',
                                       'worker_num', 'base_target_step', 'base_break_step']
                    for param in important_params:
                        if param in best_trial.params:
                            val = best_trial.params[param]
                            if isinstance(val, float) and val < 0.01:
                                print(f"    {param}: {val:.6f}")
                            else:
                                print(f"    {param}: {val}")
            except:
                pass

            print()

        # Latest trial
        if db_stats['latest_trial']:
            trial_num, state, started = db_stats['latest_trial']
            print(f"{Colors.CYAN}Current Trial:{Colors.END}")
            print(f"  Trial #{trial_num}")
            print(f"  State: {state}")
            if started:
                start_time = datetime.fromisoformat(started.replace(' ', 'T'))
                elapsed = datetime.now() - start_time
                print(f"  Running for: {elapsed}")

        print()

        # Last 10 trials summary
        recent_trials = get_recent_trials(db_path, study_name, 10)
        if recent_trials:
            print(f"{Colors.CYAN}{Colors.BOLD}LAST 10 COMPLETED TRIALS:{Colors.END}")
            print(f"{Colors.CYAN}{'─'*80}{Colors.END}")
            print(f"  {'Trial':<8} {'Objective':<15} {'Completed'}")
            print(f"  {'-'*8} {'-'*15} {'-'*25}")

            for trial in recent_trials:
                value_color = Colors.GREEN if trial['value'] > 0 else Colors.RED
                obj_str = f"{value_color}{trial['value']:+.6f}{Colors.END}"
                completed_str = trial['completed'] if trial['completed'] else 'N/A'
                print(f"  #{trial['number']:<7} {obj_str:<24} {completed_str}")

            # Statistics
            values = [t['value'] for t in recent_trials]
            avg = sum(values) / len(values)
            best_recent = max(values)
            worst_recent = min(values)

            print(f"\n  Stats: Avg={avg:+.6f} | Best={best_recent:+.6f} | Worst={worst_recent:+.6f}")
            print()

        # Econometric Analysis
        all_trials = get_all_completed_trials(db_path, study_name)
        if all_trials and len(all_trials) >= 5:
            econ_stats = analyze_convergence(all_trials)
            if econ_stats:
                print(f"{Colors.MAGENTA}{Colors.BOLD}ECONOMETRIC ANALYSIS & TREND DETECTION:{Colors.END}")
                print(f"{Colors.MAGENTA}{'─'*80}{Colors.END}")

                # Overall statistics
                print(f"\n  {Colors.CYAN}Overall Statistics:{Colors.END}")
                print(f"    Mean: {econ_stats['mean']:+.6f} ± {econ_stats['std']:.6f}")
                print(f"    Median: {econ_stats['median']:+.6f}")
                print(f"    Range: [{econ_stats['min']:+.6f}, {econ_stats['max']:+.6f}] (Δ={econ_stats['range']:.6f})")

                # Moving averages
                if 'ma5_current' in econ_stats and econ_stats['ma5_current'] is not None:
                    ma5_trend_color = Colors.GREEN if econ_stats.get('ma5_trend', 0) > 0 else Colors.RED
                    print(f"\n  {Colors.CYAN}Moving Averages:{Colors.END}")
                    print(f"    MA(5):  {econ_stats['ma5_current']:+.6f}  (trend: {ma5_trend_color}{econ_stats.get('ma5_trend', 0):+.6f}{Colors.END})")

                if 'ma10_current' in econ_stats and econ_stats['ma10_current'] is not None:
                    ma10_trend_color = Colors.GREEN if econ_stats.get('ma10_trend', 0) > 0 else Colors.RED
                    print(f"    MA(10): {econ_stats['ma10_current']:+.6f}  (trend: {ma10_trend_color}{econ_stats.get('ma10_trend', 0):+.6f}{Colors.END})")

                # Period analysis
                if 'early_mean' in econ_stats:
                    print(f"\n  {Colors.CYAN}Period Analysis:{Colors.END}")
                    print(f"    Early trials:  {econ_stats['early_mean']:+.6f}")
                    print(f"    Middle trials: {econ_stats['middle_mean']:+.6f}")
                    print(f"    Recent trials: {econ_stats['recent_mean']:+.6f}")

                    trend_color = Colors.GREEN if econ_stats['trend_improvement'] > 0 else Colors.RED
                    trend_symbol = "📈" if econ_stats['trend_improvement'] > 0 else "📉"
                    print(f"    Overall trend: {trend_symbol} {trend_color}{econ_stats['trend_improvement']:+.6f}{Colors.END}")

                # Convergence metrics
                print(f"\n  {Colors.CYAN}Convergence:{Colors.END}")
                print(f"    Best trial found: #{int(econ_stats['best_trial'])} (value: {econ_stats['best_value']:+.6f})")
                print(f"    Trials since best: {int(econ_stats['trials_since_best'])}")

                if 'recent_10_mean' in econ_stats:
                    recent_vs_best_color = Colors.GREEN if econ_stats['recent_vs_best'] > -0.005 else Colors.YELLOW
                    print(f"    Recent 10 avg: {econ_stats['recent_10_mean']:+.6f}")
                    print(f"    Recent vs Best: {recent_vs_best_color}{econ_stats['recent_vs_best']:+.6f}{Colors.END}")

                # Volatility
                if 'volatility_recent' in econ_stats:
                    print(f"\n  {Colors.CYAN}Volatility Analysis:{Colors.END}")
                    print(f"    Recent (10 trials): {econ_stats['volatility_recent']:.6f}")
                    print(f"    Overall: {econ_stats['volatility_overall']:.6f}")

                    vol_ratio = econ_stats['volatility_recent'] / econ_stats['volatility_overall']
                    vol_color = Colors.RED if vol_ratio > 1.5 else Colors.YELLOW if vol_ratio > 1.2 else Colors.GREEN
                    print(f"    Volatility ratio: {vol_color}{vol_ratio:.2f}x{Colors.END}")

                # Statistical tests
                if 'spearman_rho' in econ_stats:
                    print(f"\n  {Colors.CYAN}Statistical Tests:{Colors.END}")
                    sig_marker = "***" if econ_stats['spearman_p'] < 0.01 else "**" if econ_stats['spearman_p'] < 0.05 else "*" if econ_stats['spearman_p'] < 0.1 else "ns"
                    trend_dir = "Positive" if econ_stats['spearman_rho'] > 0 else "Negative"
                    rho_color = Colors.GREEN if econ_stats['spearman_rho'] > 0 else Colors.RED

                    print(f"    Spearman ρ: {rho_color}{econ_stats['spearman_rho']:+.3f}{Colors.END} (p={econ_stats['spearman_p']:.4f}) {sig_marker}")
                    print(f"    Interpretation: {trend_dir} monotonic trend {'(significant)' if econ_stats.get('trend_significant', False) else '(not significant)'}")

                if 'autocorr_lag1' in econ_stats:
                    autocorr_color = Colors.YELLOW if abs(econ_stats['autocorr_lag1']) > 0.5 else Colors.GREEN
                    print(f"    Autocorr(1): {autocorr_color}{econ_stats['autocorr_lag1']:+.3f}{Colors.END}")
                    if abs(econ_stats['autocorr_lag1']) > 0.5:
                        print(f"      → High autocorrelation detected (trials not independent)")

                if 'mean_shift_detected' in econ_stats:
                    shift_color = Colors.YELLOW if econ_stats['mean_shift_detected'] else Colors.GREEN
                    shift_text = "Detected" if econ_stats['mean_shift_detected'] else "Not detected"
                    print(f"    Mean shift: {shift_color}{shift_text}{Colors.END} (p={econ_stats.get('stationarity_p', 0):.4f})")

                # Recommendation
                recommendation, rec_color = get_recommendation(econ_stats, total)
                color_code = Colors.GREEN if rec_color == "green" else Colors.YELLOW if rec_color == "yellow" else Colors.RED

                print(f"\n  {Colors.BOLD}Recommendation:{Colors.END}")
                print(f"    {color_code}{recommendation}{Colors.END}")
                print()
    else:
        print(f"{Colors.YELLOW}No training data found{Colors.END}")
        print(f"Database: {db_path}")
        print(f"Study: {study_name}")

    print(f"{Colors.HEADER}{'='*80}{Colors.END}")

    # Process info
    try:
        import subprocess
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )

        python_procs = [line for line in result.stdout.split('\n')
                       if 'python' in line and '1_optimize_unified.py' in line]

        if python_procs:
            print(f"\n{Colors.CYAN}Training Processes:{Colors.END}")
            for proc in python_procs:
                parts = proc.split()
                cpu = parts[2] if len(parts) > 2 else '?'
                mem = parts[3] if len(parts) > 3 else '?'
                print(f"  CPU: {cpu}% | MEM: {mem}%")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Monitor Cappuccino training')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring (updates every 5s)')
    parser.add_argument('--study-name', default='cappuccino_production', help='Study name to monitor')
    parser.add_argument('--db', default='databases/optuna_cappuccino.db', help='Database path')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')

    args = parser.parse_args()

    if args.watch:
        print(f"{Colors.CYAN}Starting continuous monitoring (Ctrl+C to stop)...{Colors.END}\n")
        try:
            while True:
                print_status(args.db, args.study_name)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Monitoring stopped{Colors.END}")
    else:
        print_status(args.db, args.study_name)


if __name__ == '__main__':
    main()
