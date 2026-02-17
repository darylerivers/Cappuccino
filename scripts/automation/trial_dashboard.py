#!/usr/bin/env python3
"""
Trial Dashboard - Monitor Training and Archived Trials

Real-time dashboard showing:
- Current training progress with VIN codes
- Top archived trials
- Paper trading performance
- System resources
"""

import os
import sys
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import subprocess

# Add parent to path
PARENT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from utils.trial_manager import TrialManager
from utils.trial_naming import generate_trial_vin, sharpe_to_grade


class TrialDashboard:
    """Real-time trial monitoring dashboard."""

    def __init__(self, base_dir: Path = None):
        """Initialize dashboard."""
        self.base_dir = Path(base_dir) if base_dir else PARENT_DIR
        self.trial_manager = TrialManager(base_dir=self.base_dir)
        # Try optuna database first, fallback to pipeline_v2.db
        optuna_db = self.base_dir / "databases/optuna_cappuccino.db"
        self.db_path = optuna_db if optuna_db.exists() else self.base_dir / "pipeline_v2.db"

    def get_current_study(self) -> str:
        """Get the current study name, auto-detecting from running workers if possible."""
        # First, try to detect from running workers
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if '1_optimize_unified.py' in line and '--study-name' in line:
                    # Extract study name from command line
                    parts = line.split('--study-name')
                    if len(parts) > 1:
                        study_name = parts[1].strip().split()[0]
                        # Update .current_study file for consistency
                        current_study_file = self.base_dir / ".current_study"
                        with open(current_study_file, 'w') as f:
                            f.write(study_name)
                        return study_name
        except Exception:
            pass

        # Fallback to .current_study file
        current_study_file = self.base_dir / ".current_study"
        if current_study_file.exists():
            return current_study_file.read_text().strip()
        return None

    def get_running_workers(self) -> List[Dict]:
        """Get currently running training workers."""
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            workers = []
            for line in result.stdout.split('\n'):
                if '1_optimize_unified.py' in line and 'grep' not in line:
                    parts = line.split()
                    workers.append({
                        'pid': int(parts[1]),
                        'cpu': float(parts[2]),
                        'mem': float(parts[3]),
                        'time': parts[9]
                    })
            return workers
        except Exception as e:
            return []

    def get_recent_trials(self, study_name: str = None, limit: int = 10) -> List[Dict]:
        """Get recent trials with VIN codes."""
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get study
        if study_name:
            cursor.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (study_name,)
            )
        else:
            cursor.execute(
                "SELECT study_id FROM studies ORDER BY study_id DESC LIMIT 1"
            )

        study_row = cursor.fetchone()
        if not study_row:
            conn.close()
            return []

        study_id = study_row[0]

        # Get recent trials with their values (Optuna schema)
        # Include RUNNING trials, not just COMPLETE
        cursor.execute("""
            SELECT t.trial_id, t.number, tv.value, t.state, t.datetime_complete, t.datetime_start
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state IN ('COMPLETE', 'RUNNING')
            ORDER BY t.trial_id DESC
            LIMIT ?
        """, (study_id, limit))

        trials = []
        for row in cursor.fetchall():
            trial_id, number, sharpe, state, complete_time, start_time = row

            # Skip failed trials, but include running ones
            if state not in ('COMPLETE', 'RUNNING'):
                continue

            # Get params
            cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ?
            """, (trial_id,))

            params = {}
            for param_name, param_value in cursor.fetchall():
                # param_value might be string or already a number
                if isinstance(param_value, (int, float)):
                    params[param_name] = param_value
                else:
                    try:
                        params[param_name] = float(param_value) if '.' in str(param_value) else int(param_value)
                    except (ValueError, AttributeError, TypeError):
                        params[param_name] = param_value

            # Generate VIN (use 0.0 for running trials without values)
            if state == 'RUNNING' and sharpe is None:
                # For running trials, use placeholder
                vin = f"PPO-?-RUNNING-Trial{number}"
                grade = '?'
                display_sharpe = None
            else:
                vin, grade, _ = generate_trial_vin(
                    'ppo',
                    sharpe if sharpe is not None else 0.0,
                    params,
                    datetime.fromisoformat(complete_time) if complete_time else datetime.now()
                )
                display_sharpe = sharpe

            trials.append({
                'number': number,
                'vin': vin,
                'grade': grade,
                'sharpe': display_sharpe,
                'complete_time': complete_time,
                'state': state,
                'start_time': start_time
            })

        conn.close()
        return trials

    def get_paper_trading_status(self) -> Dict:
        """Get paper trading performance with benchmarks."""
        # Try to read from paper trader logs
        log_files = list((self.base_dir / "logs").glob("paper_trader_*.log"))
        if not log_files:
            return None

        # Get most recent log
        latest_log = max(log_files, key=lambda p: p.stat().st_mtime)

        try:
            # Read last 100 lines for latest status
            result = subprocess.run(
                ["tail", "-100", str(latest_log)],
                capture_output=True,
                text=True
            )

            # Parse for portfolio value and positions
            status = {
                'log_file': latest_log.name,
                'last_update': datetime.fromtimestamp(latest_log.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'initial_capital': 500.0  # Standard starting capital
            }

            for line in result.stdout.split('\n'):
                if 'Portfolio:' in line or 'Portfolio Value:' in line:
                    try:
                        value = line.split('$')[1].split()[0]
                        status['portfolio_value'] = float(value)
                    except:
                        pass
                elif 'Positions:' in line:
                    try:
                        pos_count = int(line.split(':')[1].strip().split()[0])
                        status['positions'] = pos_count
                    except:
                        pass
                elif 'Cash:' in line:
                    try:
                        cash = line.split('$')[1].split()[0]
                        status['cash'] = float(cash)
                    except:
                        pass

            # Try to find CSV data for benchmark calculations
            csv_files = list(self.base_dir.glob("**/paper_trading*/*bars*.csv"))
            if not csv_files:
                csv_files = list(self.base_dir.glob("**/paper_trading*/*history*.csv"))

            if csv_files and status.get('portfolio_value'):
                latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                try:
                    # Quick benchmark calculation from CSV
                    import pandas as pd
                    import numpy as np

                    df = pd.read_csv(latest_csv)
                    if 'close' in df.columns:
                        # Get BTC prices
                        btc_data = df[df.iloc[:, 0].str.contains('BTC', case=False, na=False)]
                        if len(btc_data) > 1:
                            btc_start = btc_data['close'].iloc[0]
                            btc_end = btc_data['close'].iloc[-1]
                            btc_return = ((btc_end / btc_start) - 1) * 100
                            status['btc_return'] = btc_return

                            # Calculate strategy return
                            strategy_return = ((status['portfolio_value'] / status['initial_capital']) - 1) * 100
                            status['strategy_return'] = strategy_return
                            status['alpha_vs_btc'] = strategy_return - btc_return

                            # Simple equal-weight: average of all tickers
                            all_closes = df.groupby(df.index // 7)['close'].mean()  # 7 tickers
                            if len(all_closes) > 1:
                                eqw_return = ((all_closes.iloc[-1] / all_closes.iloc[0]) - 1) * 100
                                status['eqw_return'] = eqw_return
                                status['alpha_vs_eqw'] = strategy_return - eqw_return
                except Exception as e:
                    pass  # Benchmarks optional

            return status
        except Exception as e:
            return None

    def display_dashboard(self):
        """Display the dashboard."""
        os.system('clear')

        print("=" * 80)
        print(f"  CAPPUCCINO TRAINING DASHBOARD".center(80))
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print("=" * 80)
        print()

        # Current study with statistics
        study = self.get_current_study()
        if study:
            print(f"📊 Current Study: {study}")

            # Get quick stats
            if self.db_path.exists():
                try:
                    conn = sqlite3.connect(str(self.db_path))
                    cursor = conn.cursor()
                    cursor.execute('SELECT study_id FROM studies WHERE study_name = ?', (study,))
                    study_row = cursor.fetchone()
                    if study_row:
                        study_id = study_row[0]
                        cursor.execute('''
                            SELECT
                                COUNT(*) as completed,
                                AVG(tv.value) as avg_sharpe,
                                MAX(tv.value) as max_sharpe
                            FROM trials t
                            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
                            WHERE t.study_id = ? AND t.state = "COMPLETE"
                        ''', (study_id,))
                        completed, avg_sharpe, max_sharpe = cursor.fetchone()
                        if completed and completed > 0:
                            print(f"   Completed: {completed} trials | Avg Sharpe: {avg_sharpe:.4f} | Max: {max_sharpe:.4f}")
                    conn.close()
                except:
                    pass
        else:
            print(f"📊 Current Study: None")
        print()

        # Running workers
        workers = self.get_running_workers()
        print(f"👥 Training Workers: {len(workers)}")
        if workers:
            for i, worker in enumerate(workers, 1):
                print(f"   Worker {i}: PID {worker['pid']} | CPU {worker['cpu']}% | MEM {worker['mem']}% | Time {worker['time']}")
        else:
            print("   No workers running")
        print()

        # Recent trials
        print("🎯 Recent Trials:")
        recent = self.get_recent_trials(study, limit=10)
        if recent:
            print(f"   {'Trial':<8} {'VIN Code':<45} {'Grade':<7} {'Status':<12} {'Sharpe':<10}")
            print(f"   {'-'*8} {'-'*45} {'-'*7} {'-'*12} {'-'*10}")
            for trial in recent:
                grade_icon = {
                    'S': '🏆', 'A': '⭐', 'B': '✅', 'C': '🔵', 'D': '⚠️', 'F': '❌', '?': '⏳'
                }.get(trial['grade'], '?')

                # Show state for running trials
                if trial['state'] == 'RUNNING':
                    status = "🔄 RUNNING"
                    sharpe_display = "..."
                else:
                    status = "✓ Complete"
                    sharpe_display = f"{trial['sharpe']:>8.4f}" if trial['sharpe'] is not None else "N/A"

                print(f"   #{trial['number']:<7} {trial['vin']:<45} {grade_icon} {trial['grade']:<5} {status:<12} {sharpe_display}")
        else:
            print("   No trials yet (waiting for workers to start)")
        print()

        # Top archived trials
        print("🏆 Top Archived Trials:")
        archived = self.trial_manager.list_archived_trials(limit=5)
        if archived:
            for i, trial in enumerate(archived, 1):
                grade_icon = {
                    'S': '🏆', 'A': '⭐', 'B': '✅', 'C': '🔵', 'D': '⚠️', 'F': '❌'
                }.get(trial['grade'], '?')
                print(f"   {i}. {grade_icon} {trial['vin']:<45} Sharpe: {trial['sharpe']:.4f}")
        else:
            print("   No archived trials yet")
        print()

        # Paper trading status with benchmarks
        print("📈 Paper Trading:")
        pt_status = self.get_paper_trading_status()
        if pt_status:
            pv = pt_status.get('portfolio_value')
            print(f"   Portfolio Value: ${pv:.2f}" if pv else "   Portfolio Value: N/A")

            # Show return if available
            if 'strategy_return' in pt_status:
                sr = pt_status['strategy_return']
                print(f"   Strategy Return: {sr:+.2f}%")

            # Show benchmarks side by side
            if 'btc_return' in pt_status and 'eqw_return' in pt_status:
                print(f"   ┌─ Benchmarks ──────────────────────────────┐")
                print(f"   │ BTC-Only:     {pt_status['btc_return']:+7.2f}%  │  Alpha: {pt_status.get('alpha_vs_btc', 0):+.2f}%")
                print(f"   │ Equal-Weight: {pt_status['eqw_return']:+7.2f}%  │  Alpha: {pt_status.get('alpha_vs_eqw', 0):+.2f}%")
                print(f"   └───────────────────────────────────────────┘")

            print(f"   Positions: {pt_status.get('positions', 'N/A')}")

            # Time since last update with alarm
            last_update_str = pt_status.get('last_update', '')
            if last_update_str:
                try:
                    last_update = datetime.strptime(last_update_str, '%Y-%m-%d %H:%M:%S')
                    time_since = datetime.now() - last_update
                    hours = int(time_since.total_seconds() // 3600)
                    minutes = int((time_since.total_seconds() % 3600) // 60)

                    # Alarm if out of sync (more than 2 hours)
                    if hours >= 2:
                        print(f"   ⚠️  SYNC ALARM: Last update {hours}h {minutes}m ago - STALE DATA!")
                    else:
                        print(f"   Last Update: {last_update_str} ({hours}h {minutes}m ago)")
                except:
                    print(f"   Last Update: {last_update_str}")
            else:
                print(f"   Last Update: Unknown")
        else:
            print("   ⚠️  No active paper trading - start a trader to see benchmarks")
        print()

        print("=" * 80)
        print("Commands: Ctrl+C to exit | Refreshes every 30s")
        print("=" * 80)

    def run(self, refresh_interval: int = 30):
        """Run the dashboard with auto-refresh."""
        try:
            while True:
                self.display_dashboard()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Trial Dashboard")
    parser.add_argument('--refresh', type=int, default=30, help='Refresh interval in seconds (default: 30)')
    parser.add_argument('--once', action='store_true', help='Display once and exit (no auto-refresh)')

    args = parser.parse_args()

    dashboard = TrialDashboard()

    if args.once:
        dashboard.display_dashboard()
    else:
        dashboard.run(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()
