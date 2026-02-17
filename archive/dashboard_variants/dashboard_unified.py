#!/usr/bin/env python3
"""
Unified Cappuccino Dashboard - Switch Between Multiple Views

Views:
  [1] Overview - All systems status (training, trading, advisor, system health)
  [2] Training Details - Detailed training metrics and analysis
  [3] Paper Trading - Detailed trading performance
  [4] System Health - GPU, CPU, Memory, Disk

Controls:
  1-4: Switch views
  R: Force refresh
  Q: Quit
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import sqlite3
    import pandas as pd
    import psutil
    import curses
except ImportError:
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "psutil"], check=True)
    import sqlite3
    import pandas as pd
    import psutil
    import curses


class UnifiedDashboard:
    """Unified dashboard with multiple switchable views."""

    def __init__(self, compact: bool = False):
        self.compact = compact
        self.db_path = "databases/optuna_cappuccino.db"
        self.current_view = 1  # Start with overview
        self.auto_refresh = True
        self.refresh_interval = 5

    def colorize(self, text: str, color: str) -> str:
        """Add color to text."""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m',
        }
        return f"{colors.get(color, '')}{text}{colors['end']}"

    def box(self, title: str, content: List[str], width: int = 100) -> str:
        """Create a nice box around content."""
        lines = []
        lines.append("┌─ " + self.colorize(title, 'bold') + " " + "─" * (width - len(title) - 5) + "┐")

        for line in content:
            clean_line = re.sub(r'\033\[[0-9;]+m', '', line)
            padding = width - len(clean_line) - 4
            lines.append("│ " + line + " " * padding + " │")

        lines.append("└" + "─" * (width - 2) + "┘")
        return "\n".join(lines)

    # ==================== DATA COLLECTION METHODS ====================

    def get_training_status(self) -> Dict:
        """Get training workers status."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "1_optimize_unified.py"],
                capture_output=True,
                text=True
            )
            pids = result.stdout.strip().split('\n') if result.stdout.strip() else []

            workers = []
            for pid in pids:
                if pid:
                    ps_result = subprocess.run(
                        ["ps", "-p", pid, "-o", "pid,%cpu,%mem,etime"],
                        capture_output=True,
                        text=True
                    )
                    if ps_result.returncode == 0:
                        lines = ps_result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            parts = lines[1].split()
                            workers.append({
                                'pid': parts[0],
                                'cpu': parts[1],
                                'mem': parts[2],
                                'time': parts[3],
                            })

            trial_info = self.get_trial_counts()

            return {
                'running': len(workers) > 0,
                'workers': workers,
                'trials': trial_info,
            }
        except Exception as e:
            return {'running': False, 'error': str(e)}

    def get_trial_counts(self) -> Dict:
        """Get trial counts from database."""
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT
                COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as completed,
                COUNT(CASE WHEN t.state = 'RUNNING' THEN 1 END) as running,
                COUNT(CASE WHEN t.state = 'FAIL' THEN 1 END) as failed,
                MAX(tv.value) as best_value
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            """
            df = pd.read_sql_query(query, conn)

            one_hour_ago = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            recent_query = """
            SELECT COUNT(*) as recent
            FROM trials
            WHERE state = 'COMPLETE' AND datetime_complete > ?
            """
            cursor = conn.cursor()
            cursor.execute(recent_query, (one_hour_ago,))
            recent_count = cursor.fetchone()[0]

            top10_query = """
            SELECT value
            FROM trial_values tv
            JOIN trials t ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY value DESC
            LIMIT 1 OFFSET (SELECT COUNT(*) / 10 FROM trials WHERE state = 'COMPLETE')
            """
            cursor.execute(top10_query)
            top10_row = cursor.fetchone()

            conn.close()

            result = df.iloc[0].to_dict() if not df.empty else {}
            result['recent_completed'] = recent_count
            result['top_10_percent_threshold'] = top10_row[0] if top10_row else None

            return result
        except Exception:
            return {}

    def get_paper_trading_status(self) -> Dict:
        """Get paper trading status."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "paper_trader_alpaca_polling.py"],
                capture_output=True,
                text=True
            )
            pid = result.stdout.strip()

            if not pid:
                return {'running': False}

            ps_result = subprocess.run(
                ["ps", "-p", pid, "-o", "pid,%cpu,%mem,etime"],
                capture_output=True,
                text=True
            )

            info = {'running': True, 'pid': pid}
            if ps_result.returncode == 0:
                lines = ps_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    parts = lines[1].split()
                    info.update({
                        'cpu': parts[1],
                        'mem': parts[2],
                        'time': parts[3],
                    })

            # Get latest CSV info
            current_session = Path("paper_trades/alpaca_session.csv")
            if current_session.exists():
                info['csv_file'] = current_session.name

                try:
                    with current_session.open() as f:
                        line_count = sum(1 for _ in f) - 1
                    info['trades'] = line_count

                    if line_count > 0:
                        df = pd.read_csv(current_session)
                        if not df.empty:
                            latest = df.iloc[-1]
                            info['latest_timestamp'] = latest['timestamp']
                            info['total_asset'] = float(latest['total_asset'])
                            info['cash'] = float(latest['cash'])

                            positions = {}
                            for col in df.columns:
                                if col.startswith('holding_'):
                                    ticker = col.replace('holding_', '')
                                    holding = float(latest[col])
                                    if holding > 0.0001:
                                        price_col = f'price_{ticker}'
                                        if price_col in df.columns:
                                            price = float(latest[price_col])
                                            positions[ticker] = {
                                                'shares': holding,
                                                'price': price,
                                                'value': holding * price
                                            }

                            info['positions'] = positions

                            initial_total = float(df.iloc[0]['total_asset']) if len(df) > 0 else 1000.0
                            pnl = info['total_asset'] - initial_total
                            pnl_pct = (pnl / initial_total) * 100
                            info['pnl'] = pnl
                            info['pnl_pct'] = pnl_pct
                except Exception:
                    pass

            return info
        except Exception:
            return {'running': False}

    def get_system_health(self) -> Dict:
        """Get system health metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            health = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
            }

            # GPU info
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    gpu_data = result.stdout.strip().split(',')
                    if len(gpu_data) >= 4:
                        health['gpu_util'] = float(gpu_data[0])
                        health['gpu_mem_used'] = float(gpu_data[1]) / 1024  # Convert to GB
                        health['gpu_mem_total'] = float(gpu_data[2]) / 1024
                        health['gpu_temp'] = float(gpu_data[3])
                        health['gpu_available'] = True
                    else:
                        health['gpu_available'] = False
                else:
                    health['gpu_available'] = False
            except Exception:
                health['gpu_available'] = False

            return health
        except Exception as e:
            return {'error': str(e)}

    def get_study_overview(self) -> List[Dict]:
        """Get all studies overview."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT
                s.study_name,
                COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as completed,
                MAX(tv.value) as best_value,
                AVG(tv.value) as avg_value
            FROM studies s
            LEFT JOIN trials t ON s.study_id = t.study_id
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            GROUP BY s.study_id
            ORDER BY best_value DESC
            """
            studies_info = pd.read_sql_query(query, conn)
            conn.close()
            return studies_info.to_dict('records') if not studies_info.empty else []
        except Exception:
            return []

    def get_convergence_data(self) -> Dict:
        """Get convergence statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT tv.value, t.datetime_complete
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY t.datetime_complete
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                return {}

            df['rolling_max'] = df['value'].expanding().max()
            recent_100 = df.tail(100)
            recent_500 = df.tail(500)

            return {
                'total_trials': len(df),
                'current_best': df['value'].max(),
                'avg_last_100': recent_100['value'].mean() if len(recent_100) > 0 else 0,
                'avg_last_500': recent_500['value'].mean() if len(recent_500) > 0 else 0,
                'std_last_100': recent_100['value'].std() if len(recent_100) > 0 else 0,
                'improvement_rate': (recent_100['value'].mean() - df.head(100)['value'].mean()) / abs(df.head(100)['value'].mean()) * 100 if len(df) > 100 else 0,
            }
        except Exception:
            return {}

    def get_top_trials(self, limit: int = 10) -> List[Dict]:
        """Get top performing trials."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT
                t.trial_id,
                t.number,
                tv.value,
                tua.value_json as timeframe,
                tua2.value_json as folder
            FROM trials t
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            LEFT JOIN trial_user_attributes tua ON t.trial_id = tua.trial_id AND tua.key = 'timeframe'
            LEFT JOIN trial_user_attributes tua2 ON t.trial_id = tua2.trial_id AND tua2.key = 'name_folder'
            WHERE t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT ?
            """
            top_trials = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()

            trials_list = top_trials.to_dict('records')
            for trial in trials_list:
                if trial.get('timeframe'):
                    trial['timeframe'] = trial['timeframe'].strip('"')
                if trial.get('folder'):
                    trial['folder'] = trial['folder'].strip('"')

            return trials_list
        except Exception:
            return []

    # ==================== VIEW RENDERING METHODS ====================

    def render_view_1_overview(self, width: int) -> List[str]:
        """Render overview with all systems."""
        lines = []

        # Training section
        training = self.get_training_status()
        if training.get('running'):
            workers = training.get('workers', [])
            lines.append(self.colorize(f"TRAINING: RUNNING ({len(workers)} workers)", 'green'))
            trials = training.get('trials', {})
            total = trials.get('completed', 0) + trials.get('running', 0) + trials.get('failed', 0)
            lines.append(f"  Trials: {total} | Completed: {trials.get('completed', 0)} | Best: {trials.get('best_value', 0):.6f}")
        else:
            lines.append(self.colorize("TRAINING: NOT RUNNING", 'red'))

        lines.append("")

        # Paper trading section
        trading = self.get_paper_trading_status()
        if trading.get('running'):
            lines.append(self.colorize(f"PAPER TRADING: RUNNING (PID {trading['pid']})", 'green'))
            lines.append(f"  CPU {trading.get('cpu', '?')}% | Mem {trading.get('mem', '?')}% | Uptime {trading.get('time', '?')}")
            if 'total_asset' in trading:
                lines.append(f"  Total: ${trading['total_asset']:.2f} | Cash: ${trading['cash']:.2f} | P&L: ${trading.get('pnl', 0):.2f} ({trading.get('pnl_pct', 0):+.2f}%)")
        else:
            lines.append(self.colorize("PAPER TRADING: NOT RUNNING", 'red'))

        lines.append("")

        # System health
        health = self.get_system_health()
        lines.append("SYSTEM HEALTH")
        lines.append(f"  CPU: {health.get('cpu_percent', 0):.1f}% | Memory: {health.get('memory_percent', 0):.1f}% | Disk: {health.get('disk_percent', 0):.1f}%")
        if health.get('gpu_available'):
            lines.append(f"  GPU: {health['gpu_util']:.1f}% | GPU Mem: {health['gpu_mem_used']:.1f}/{health['gpu_mem_total']:.1f} GB | Temp: {health['gpu_temp']:.0f}°C")

        return lines

    def render_view_2_training_details(self, width: int) -> List[str]:
        """Render detailed training view."""
        lines = []

        # Study overview
        studies = self.get_study_overview()
        lines.append(self.colorize("STUDY OVERVIEW", 'cyan'))
        for study in studies[:5]:
            lines.append(f"  {study['study_name']:<40} | Trials: {study['completed']:>4} | Best: {study['best_value']:.6f}")
        lines.append("")

        # Convergence
        conv = self.get_convergence_data()
        if conv:
            lines.append(self.colorize("CONVERGENCE", 'cyan'))
            lines.append(f"  Total: {conv['total_trials']} | Best: {self.colorize(f\"{conv['current_best']:.6f}\", 'green')} | Avg(100): {conv['avg_last_100']:.6f}")
            improvement_color = 'green' if conv['improvement_rate'] > 0 else 'red'
            lines.append(f"  Improvement: {self.colorize(f\"{conv['improvement_rate']:+.2f}%\", improvement_color)} | Std: {conv['std_last_100']:.6f}")
            lines.append("")

        # Top trials
        top = self.get_top_trials(8)
        lines.append(self.colorize("TOP TRIALS", 'cyan'))
        for i, trial in enumerate(top, 1):
            color = 'green' if i <= 3 else 'cyan' if i <= 5 else 'white'
            timeframe = trial.get('timeframe', 'N/A') or 'N/A'
            lines.append(f"  {i}. Trial #{trial['number']:<5} | {self.colorize(f\"{trial['value']:.6f}\", color)} | {timeframe}")

        return lines

    def render_view_3_trading_details(self, width: int) -> List[str]:
        """Render detailed trading view."""
        lines = []

        trading = self.get_paper_trading_status()
        if not trading.get('running'):
            lines.append(self.colorize("PAPER TRADING: NOT RUNNING", 'red'))
            return lines

        lines.append(self.colorize(f"PAPER TRADING - PID {trading['pid']}", 'green'))
        lines.append(f"CPU: {trading.get('cpu', '?')}% | Memory: {trading.get('mem', '?')}% | Uptime: {trading.get('time', '?')}")
        lines.append("")

        if 'total_asset' in trading:
            lines.append("PORTFOLIO")
            lines.append(f"  Total Value: ${trading['total_asset']:.2f}")
            lines.append(f"  Cash: ${trading['cash']:.2f}")
            lines.append(f"  P&L: {self.colorize(f\"${trading.get('pnl', 0):+.2f}\", 'green' if trading.get('pnl', 0) > 0 else 'red')} ({trading.get('pnl_pct', 0):+.2f}%)")
            lines.append("")

            if 'positions' in trading and trading['positions']:
                lines.append("POSITIONS")
                for ticker, pos in trading['positions'].items():
                    lines.append(f"  {ticker}: {pos['shares']:.4f} @ ${pos['price']:.2f} = ${pos['value']:.2f}")

            if 'trades' in trading:
                lines.append("")
                lines.append(f"Total Trades: {trading['trades']}")

        return lines

    def render_view_4_system_health(self, width: int) -> List[str]:
        """Render system health view."""
        lines = []

        health = self.get_system_health()

        lines.append(self.colorize("CPU", 'cyan'))
        lines.append(f"  Utilization: {health.get('cpu_percent', 0):.1f}%")
        lines.append("")

        lines.append(self.colorize("MEMORY", 'cyan'))
        lines.append(f"  Used: {health.get('memory_used_gb', 0):.1f} / {health.get('memory_total_gb', 0):.1f} GB ({health.get('memory_percent', 0):.1f}%)")
        lines.append("")

        lines.append(self.colorize("DISK", 'cyan'))
        lines.append(f"  Used: {health.get('disk_used_gb', 0):.1f} / {health.get('disk_total_gb', 0):.1f} GB ({health.get('disk_percent', 0):.1f}%)")
        lines.append("")

        if health.get('gpu_available'):
            lines.append(self.colorize("GPU", 'cyan'))
            lines.append(f"  Utilization: {health['gpu_util']:.1f}%")
            lines.append(f"  Memory: {health['gpu_mem_used']:.1f} / {health['gpu_mem_total']:.1f} GB")
            lines.append(f"  Temperature: {health['gpu_temp']:.0f}°C")
        else:
            lines.append(self.colorize("GPU: Not Available", 'yellow'))

        return lines

    def render_dashboard(self) -> str:
        """Render the complete dashboard."""
        width = 100
        output = []

        # Header
        output.append("=" * width)
        output.append(self.colorize("CAPPUCCINO UNIFIED DASHBOARD", 'bold').center(width + 10))
        output.append("=" * width)
        output.append("")

        # View tabs
        tabs = []
        for i in range(1, 5):
            view_names = {1: "Overview", 2: "Training", 3: "Trading", 4: "System"}
            if i == self.current_view:
                tabs.append(self.colorize(f"[{i}] {view_names[i]}", 'green'))
            else:
                tabs.append(f"[{i}] {view_names[i]}")

        output.append("  ".join(tabs) + "  " + self.colorize("[R]efresh", 'yellow') + "  " + self.colorize("[Q]uit", 'red'))
        output.append("")

        # Render current view
        if self.current_view == 1:
            content = self.render_view_1_overview(width)
        elif self.current_view == 2:
            content = self.render_view_2_training_details(width)
        elif self.current_view == 3:
            content = self.render_view_3_trading_details(width)
        elif self.current_view == 4:
            content = self.render_view_4_system_health(width)
        else:
            content = ["Invalid view"]

        output.append(self.box(f"VIEW {self.current_view}", content, width))
        output.append("")
        output.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(output)

    def run_interactive(self):
        """Run dashboard with keyboard controls."""
        import select
        import termios
        import tty

        def getch_nonblocking():
            """Get character without blocking."""
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1)
            return None

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to raw mode
            tty.setcbreak(sys.stdin.fileno())

            while True:
                # Clear screen and display
                os.system('clear')
                print(self.render_dashboard())

                # Wait for input with timeout
                start_time = time.time()
                while time.time() - start_time < self.refresh_interval:
                    ch = getch_nonblocking()
                    if ch:
                        if ch == 'q' or ch == 'Q':
                            return
                        elif ch in '1234':
                            self.current_view = int(ch)
                            break
                        elif ch == 'r' or ch == 'R':
                            break
                    time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\nExiting...")

    def run_simple(self, refresh_interval: int = 0):
        """Run dashboard with simple auto-refresh (no keyboard control)."""
        try:
            while True:
                os.system('clear')
                print(self.render_dashboard())

                if refresh_interval == 0:
                    break

                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(description='Unified Cappuccino Dashboard')
    parser.add_argument('--view', type=int, default=1, choices=[1, 2, 3, 4], help='Starting view (1-4)')
    parser.add_argument('--refresh', type=int, default=5, help='Auto-refresh interval in seconds (0 for single display)')
    parser.add_argument('--simple', action='store_true', help='Simple mode without keyboard controls')
    parser.add_argument('--compact', action='store_true', help='Compact display mode')
    args = parser.parse_args()

    dashboard = UnifiedDashboard(compact=args.compact)
    dashboard.current_view = args.view
    dashboard.refresh_interval = args.refresh

    if args.simple or args.refresh == 0:
        dashboard.run_simple(args.refresh)
    else:
        dashboard.run_interactive()


if __name__ == '__main__':
    main()
