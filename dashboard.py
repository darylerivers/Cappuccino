#!/usr/bin/env python3
"""
Cappuccino Trading System Dashboard

Real-time monitoring of:
- Training (3 workers)
- Paper Trading (Alpaca)
- Autonomous AI Advisor
- System Health (GPU, Disk, Memory, CPU)
- Watchdog Status
- Recent Performance Metrics

Usage:
    python dashboard.py
    python dashboard.py --compact  # Smaller view
    python dashboard.py --once     # Single refresh (no loop)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import sqlite3
    import pandas as pd
    import psutil
except ImportError:
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "psutil"], check=True)
    import sqlite3
    import pandas as pd
    import psutil


class CappuccinoDashboard:
    """Real-time dashboard for all cappuccino systems."""

    def __init__(self, compact: bool = False):
        self.compact = compact
        self.db_path = "databases/optuna_cappuccino.db"

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal width and height."""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except:
            return 80, 24

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

    def box(self, title: str, content: List[str], width: int = 80) -> str:
        """Create a nice box around content."""
        lines = []

        # Top border
        lines.append("┌─ " + self.colorize(title, 'bold') + " " + "─" * (width - len(title) - 5) + "┐")

        # Content
        for line in content:
            # Remove ANSI codes for length calculation
            clean_line = re.sub(r'\033\[[0-9;]+m', '', line)
            padding = width - len(clean_line) - 4
            lines.append("│ " + line + " " * padding + " │")

        # Bottom border
        lines.append("└" + "─" * (width - 2) + "┘")

        return "\n".join(lines)

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
                    # Get process info
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

            # Get trial counts from database
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

            # Get study info
            query = """
            SELECT s.study_name,
                   COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as completed,
                   COUNT(CASE WHEN t.state = 'RUNNING' THEN 1 END) as running,
                   COUNT(CASE WHEN t.state = 'FAIL' THEN 1 END) as failed,
                   MAX(tv.value) as best_value
            FROM studies s
            JOIN trials t ON s.study_id = t.study_id
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE s.study_name = 'cappuccino_3workers_20251102_2325'
            GROUP BY s.study_name
            """
            df = pd.read_sql_query(query, conn)

            # Get trials completed in last hour
            one_hour_ago = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            recent_query = """
            SELECT COUNT(*) as recent
            FROM trials
            WHERE state = 'COMPLETE' AND datetime_complete > ?
            """
            cursor = conn.cursor()
            cursor.execute(recent_query, (one_hour_ago,))
            recent_count = cursor.fetchone()[0]

            # Get top 10% threshold
            top10_query = """
            SELECT value
            FROM trial_values tv
            JOIN trials t ON tv.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY value DESC
            LIMIT 1 OFFSET (SELECT COUNT(*) / 10 FROM trials WHERE state = 'COMPLETE')
            """
            cursor.execute(top10_query)
            top10_result = cursor.fetchone()
            top10_threshold = top10_result[0] if top10_result else None

            conn.close()

            if not df.empty:
                return {
                    'completed': int(df.iloc[0]['completed']),
                    'running': int(df.iloc[0]['running']),
                    'failed': int(df.iloc[0]['failed']),
                    'best_value': float(df.iloc[0]['best_value']) if df.iloc[0]['best_value'] else 0.0,
                    'recent_completed': recent_count,
                    'top_10_percent_threshold': top10_threshold,
                }
        except Exception as e:
            pass

        return {'completed': 0, 'running': 0, 'failed': 0, 'best_value': 0.0, 'recent_completed': 0}

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

            # Get process info
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

            # Parse log file for more info
            log_file = Path("logs/paper_trading_live.log")
            if log_file.exists():
                try:
                    with log_file.open() as f:
                        lines = f.readlines()

                    # Check beginning of log for tickers
                    for line in lines[:20]:
                        if 'Tickers:' in line and 'tickers' not in info:
                            import ast
                            start = line.index('[')
                            end = line.index(']') + 1
                            info['tickers'] = ast.literal_eval(line[start:end])
                            break

                    # Check end of log for recent activity
                    for line in reversed(lines[-50:]):
                        if 'Polling for new bars' in line and 'last_poll' not in info:
                            if line.startswith('['):
                                timestamp = line[1:line.index(']')]
                                info['last_poll'] = timestamp

                        if 'Fetched' in line and 'raw bars' in line and 'bars_fetched' not in info:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'Fetched':
                                    info['bars_fetched'] = int(parts[i+1])
                                    break
                except Exception:
                    pass

            # Get latest CSV info - check both current session and timestamped files
            latest_csv = None

            # First check for current session file
            current_session = Path("paper_trades/alpaca_session.csv")
            if current_session.exists():
                latest_csv = current_session
            else:
                # Fall back to timestamped files
                csv_files = sorted(Path("paper_trades").glob("alpaca_session_*.csv"))
                if csv_files:
                    latest_csv = csv_files[-1]

            if latest_csv:
                info['csv_file'] = latest_csv.name

                # Count trades and get positions
                try:
                    with latest_csv.open() as f:
                        line_count = sum(1 for _ in f) - 1  # Subtract header
                    info['trades'] = line_count

                    # Get latest trade
                    if line_count > 0:
                        df = pd.read_csv(latest_csv)
                        if not df.empty:
                            latest = df.iloc[-1]
                            info['latest_timestamp'] = latest['timestamp']
                            info['total_asset'] = float(latest['total_asset'])
                            info['cash'] = float(latest['cash'])

                            # Extract positions (unrealized P&L)
                            positions = {}
                            for col in df.columns:
                                if col.startswith('holding_'):
                                    ticker = col.replace('holding_', '')
                                    holding = float(latest[col])
                                    if holding > 0.0001:  # Non-zero position
                                        price_col = f'price_{ticker}'
                                        if price_col in df.columns:
                                            price = float(latest[price_col])
                                            positions[ticker] = {
                                                'shares': holding,
                                                'price': price,
                                                'value': holding * price
                                            }

                            if positions:
                                info['positions'] = positions
                                info['positions_value'] = sum(p['value'] for p in positions.values())
                except Exception:
                    pass

            # Check for failsafe wrapper
            failsafe_result = subprocess.run(
                ["pgrep", "-f", "paper_trading_failsafe.sh"],
                capture_output=True,
                text=True
            )
            if failsafe_result.stdout.strip():
                info['failsafe_running'] = True

                # Get failsafe state
                failsafe_state_file = Path("deployments/paper_trading_state.json")
                if failsafe_state_file.exists():
                    with failsafe_state_file.open() as f:
                        state = json.load(f)
                        info['total_restarts'] = state.get('total_restarts', 0)
                        info['consecutive_failures'] = state.get('consecutive_failures', 0)

            # Check for arbitrage scanner
            arb_result = subprocess.run(
                ["pgrep", "-f", "arbitrage_scanner.py"],
                capture_output=True,
                text=True
            )
            if arb_result.stdout.strip():
                info['arbitrage_running'] = True

                # Parse arbitrage log for opportunities
                arb_log = Path("logs/arbitrage_scanner.log")
                if arb_log.exists():
                    try:
                        with arb_log.open() as f:
                            lines = f.readlines()

                        # Count opportunities from recent logs
                        opp_count = 0
                        for line in reversed(lines[-100:]):
                            if '[OPPORTUNITY]' in line:
                                opp_count += 1

                        if opp_count > 0:
                            info['arbitrage_opportunities'] = opp_count
                    except Exception:
                        pass

            return info
        except Exception as e:
            return {'running': False, 'error': str(e)}

    def get_advisor_status(self) -> Dict:
        """Get autonomous advisor status."""
        try:
            # Check PID file
            pid_file = Path("logs/autonomous_advisor.pid")
            if pid_file.exists():
                pid = pid_file.read_text().strip()

                # Verify process is running
                result = subprocess.run(
                    ["ps", "-p", pid, "-o", "pid,%cpu,%mem,etime"],
                    capture_output=True,
                    text=True
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split()
                        info = {
                            'running': True,
                            'pid': parts[0],
                            'cpu': parts[1],
                            'mem': parts[2],
                            'time': parts[3],
                        }

                        # Get state info
                        state_file = Path("analysis_reports/advisor_state.json")
                        if state_file.exists():
                            with state_file.open() as f:
                                state = json.load(f)
                                info['analysis_count'] = state.get('analysis_count', 0)
                                info['tested_configs'] = len(state.get('tested_configs', []))
                                info['best_discovered'] = state.get('best_discovered_value', 0.0)
                                info['last_trial_count'] = state.get('last_trial_count', 0)

                        # Get recent log activity
                        log_file = Path("logs/autonomous_advisor.log")
                        if log_file.exists():
                            with log_file.open() as f:
                                lines = f.readlines()
                                if lines:
                                    info['last_log'] = lines[-1].strip()

                        return info

            return {'running': False}
        except Exception as e:
            return {'running': False, 'error': str(e)}

    def get_system_health(self) -> Dict:
        """Get system health metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()

            # Disk usage for /home
            disk = psutil.disk_usage('/home')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': disk.percent,
            }
        except Exception as e:
            return {'error': str(e)}

    def get_watchdog_status(self) -> Dict:
        """Get watchdog state."""
        try:
            state_file = Path("deployments/watchdog_state.json")
            if state_file.exists():
                with state_file.open() as f:
                    state = json.load(f)
                    return {
                        'enabled': True,
                        'restart_counts': state.get('restart_counts', {}),
                        'last_restart_times': state.get('last_restart_times', {}),
                    }
        except Exception:
            pass

        return {'enabled': False}

    def get_gpu_status(self) -> Dict:
        """Get GPU status."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'available': True,
                    'utilization': int(parts[0].strip()),
                    'mem_util': int(parts[1].strip()),
                    'mem_used': int(parts[2].strip()),
                    'mem_total': int(parts[3].strip()),
                    'temp': int(parts[4].strip()),
                    'power': float(parts[5].strip()),
                }
        except Exception:
            pass

        return {'available': False}

    def format_training_box(self, status: Dict, width: int) -> str:
        """Format training status box."""
        content = []

        if status.get('running'):
            workers = status.get('workers', [])
            content.append(self.colorize(f"Status: RUNNING ({len(workers)} workers)", 'green'))

            if not self.compact and workers:
                for i, w in enumerate(workers, 1):
                    content.append(f"  Worker {i}: PID {w['pid']} | CPU {w['cpu']}% | Mem {w['mem']}% | Time {w['time']}")

            trials = status.get('trials', {})
            content.append("")
            total = trials.get('completed', 0) + trials.get('running', 0) + trials.get('failed', 0)
            completed_str = self.colorize(str(trials.get('completed', 0)), 'cyan')
            running_str = self.colorize(str(trials.get('running', 0)), 'yellow')
            content.append(f"Total: {total} | Completed: {completed_str} | Running: {running_str} | Failed: {trials.get('failed', 0)}")

            # Best value
            best_val = f"{trials.get('best_value', 0):.6f}"
            content.append(f"Best Value: {self.colorize(best_val, 'green')}")

            # Top 10% threshold
            if trials.get('top_10_percent_threshold'):
                top10 = f"{trials['top_10_percent_threshold']:.6f}"
                content.append(f"Top 10% Cutoff: {self.colorize(top10, 'cyan')}")

            # Recent completions
            recent = trials.get('recent_completed', 0)
            if recent > 0:
                content.append(f"Completed (last hour): {self.colorize(str(recent), 'green')}")
        else:
            content.append(self.colorize("Status: NOT RUNNING", 'red'))

        return self.box("TRAINING", content, width)

    def format_paper_trading_box(self, status: Dict, width: int) -> str:
        """Format paper trading status box."""
        content = []

        if status.get('running'):
            content.append(self.colorize(f"Status: RUNNING (PID {status['pid']})", 'green'))

            if not self.compact and 'cpu' in status:
                content.append(f"  CPU {status['cpu']}% | Mem {status['mem']}% | Uptime {status['time']}")

            # Show tickers summary if available
            if 'tickers' in status:
                tickers = status['tickers']
                content.append("")
                content.append(f"Trading {len(tickers)} assets")

            # Show last poll time
            if 'last_poll' in status:
                content.append(f"Last Poll: {status['last_poll']}")

            if 'bars_fetched' in status:
                content.append(f"Bars Fetched: {status['bars_fetched']}")

            # Failsafe and Arbitrage status
            if status.get('failsafe_running') or status.get('arbitrage_running'):
                content.append("")

                if status.get('failsafe_running'):
                    content.append(self.colorize("Fail-safe: ACTIVE", 'green'))
                    if 'total_restarts' in status:
                        restarts = status['total_restarts']
                        consecutive = status.get('consecutive_failures', 0)
                        content.append(f"  Restarts: {restarts} total, {consecutive} consecutive")

                if status.get('arbitrage_running'):
                    arb_str = self.colorize("Arbitrage Scanner: ACTIVE", 'green')
                    content.append(arb_str)

                    if 'arbitrage_opportunities' in status:
                        opp_count = status['arbitrage_opportunities']
                        opp_str = self.colorize(f"{opp_count} opportunities", 'yellow')
                        content.append(f"  Recent: {opp_str} (last 100 scans)")
                    else:
                        content.append(f"  Status: Monitoring (no opportunities yet)")

            if 'csv_file' in status:
                content.append("")
                content.append(f"Log: {status['csv_file']}")
                trades_str = self.colorize(str(status.get('trades', 0)), 'cyan')
                content.append(f"Trades: {trades_str}")

                if 'total_asset' in status:
                    cash = status.get('cash', 0)
                    total = status.get('total_asset', 0)
                    pnl = total - 1000  # Starting with $1K in paper trading
                    pnl_color = 'green' if pnl >= 0 else 'red'
                    pnl_pct = (pnl / 1000) * 100

                    content.append(f"Cash: ${cash:,.2f}")

                    # Show positions if any
                    if 'positions' in status:
                        positions = status['positions']
                        positions_value = status.get('positions_value', 0)
                        content.append(f"Positions Value: ${positions_value:,.2f}")

                        # Show top 3 positions
                        sorted_positions = sorted(positions.items(), key=lambda x: x[1]['value'], reverse=True)
                        for ticker, pos in sorted_positions[:3]:
                            content.append(f"  {ticker}: {pos['shares']:.4f} @ ${pos['price']:.2f} = ${pos['value']:.2f}")

                    pnl_str = self.colorize(f'${pnl:+,.2f}', pnl_color)
                    pnl_pct_str = self.colorize(f'{pnl_pct:+.2f}%', pnl_color)
                    content.append(f"Total: ${total:,.2f} | P&L: {pnl_str} ({pnl_pct_str})")

            # Show all tradeable assets
            if 'tickers' in status:
                content.append("")
                content.append(self.colorize("Tradeable Assets:", 'bold'))
                tickers = status['tickers']

                # Display in rows of 3 or 4 depending on ticker name length
                tickers_per_row = 3
                for i in range(0, len(tickers), tickers_per_row):
                    row_tickers = tickers[i:i+tickers_per_row]
                    # Check which ones have positions
                    ticker_strs = []
                    for ticker in row_tickers:
                        if 'positions' in status and ticker in status['positions']:
                            # Highlight tickers with positions
                            ticker_strs.append(self.colorize(f"● {ticker}", 'green'))
                        else:
                            ticker_strs.append(f"  {ticker}")
                    content.append("  " + "  ".join(ticker_strs))
        else:
            content.append(self.colorize("Status: NOT RUNNING", 'red'))
            content.append("")
            content.append("Start with: ./paper_trading_failsafe.sh")

        return self.box("PAPER TRADING", content, width)

    def format_advisor_box(self, status: Dict, width: int) -> str:
        """Format AI advisor status box."""
        content = []

        if status.get('running'):
            content.append(self.colorize(f"Status: RUNNING (PID {status['pid']})", 'green'))

            if not self.compact and 'cpu' in status:
                content.append(f"  CPU {status['cpu']}% | Mem {status['mem']}% | Uptime {status['time']}")

            content.append("")
            analyses_str = self.colorize(str(status.get('analysis_count', 0)), 'cyan')
            content.append(f"Analyses: {analyses_str}")
            content.append(f"Tested Configs: {status.get('tested_configs', 0)}")

            best = status.get('best_discovered', 0.0)
            if best > -1e10:  # Not the default -inf value
                best_str = self.colorize(f'{best:.6f}', 'green')
                content.append(f"Best Discovered: {best_str}")

            if not self.compact and 'last_log' in status:
                log = status['last_log']
                if len(log) > width - 10:
                    log = log[:width - 13] + "..."
                content.append("")
                content.append(f"Latest: {log[-60:]}")
        else:
            content.append(self.colorize("Status: NOT RUNNING", 'red'))
            content.append("")
            content.append("Start with: ./start_autonomous_advisor.sh")

        return self.box("AUTONOMOUS AI ADVISOR", content, width)

    def format_gpu_box(self, status: Dict, width: int) -> str:
        """Format GPU status box."""
        content = []

        if status.get('available'):
            util = status['utilization']
            util_color = 'green' if util > 80 else 'yellow' if util > 20 else 'white'

            util_str = self.colorize(f"{util}%", util_color)
            content.append(f"Utilization: {util_str}")

            mem_used_gb = status['mem_used'] / 1024
            mem_total_gb = status['mem_total'] / 1024
            content.append(f"Memory: {mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB ({status['mem_util']}%)")

            temp = status['temp']
            temp_color = 'red' if temp > 80 else 'yellow' if temp > 70 else 'green'
            temp_str = self.colorize(f"{temp}°C", temp_color)
            content.append(f"Temperature: {temp_str}")
            content.append(f"Power: {status['power']:.1f}W")
        else:
            content.append(self.colorize("GPU not available", 'red'))

        return self.box("GPU STATUS", content, width)

    def format_system_health_box(self, status: Dict, width: int) -> str:
        """Format system health box."""
        content = []

        if 'error' not in status:
            # CPU
            cpu = status['cpu_percent']
            cpu_color = 'red' if cpu > 90 else 'yellow' if cpu > 70 else 'green'
            cpu_str = self.colorize(f"{cpu:.1f}%", cpu_color)
            content.append(f"CPU Usage: {cpu_str}")

            # Memory
            mem_pct = status['memory_percent']
            mem_color = 'red' if mem_pct > 90 else 'yellow' if mem_pct > 70 else 'green'
            mem_str = self.colorize(f"{mem_pct:.1f}%", mem_color)
            content.append(f"Memory: {status['memory_used_gb']:.1f}GB / {status['memory_total_gb']:.1f}GB ({mem_str})")

            # Disk
            disk_pct = status['disk_percent']
            disk_color = 'red' if disk_pct > 80 else 'yellow' if disk_pct > 60 else 'green'
            disk_str = self.colorize(f"{disk_pct:.1f}%", disk_color)
            content.append(f"Disk (/home): {status['disk_used_gb']:.1f}GB used, {status['disk_free_gb']:.1f}GB free ({disk_str})")
        else:
            content.append(self.colorize(f"Error: {status['error']}", 'red'))

        return self.box("SYSTEM HEALTH", content, width)

    def render(self):
        """Render the dashboard."""
        self.clear_screen()

        width, height = self.get_terminal_size()
        box_width = min(width - 2, 100)

        # Header
        header = "═" * box_width
        title = "CAPPUCCINO TRADING SYSTEM DASHBOARD"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(self.colorize(header, 'cyan'))
        print(self.colorize(title.center(box_width), 'bold'))
        print(self.colorize(timestamp.center(box_width), 'white'))
        print(self.colorize(header, 'cyan'))
        print()

        # Get all statuses
        training = self.get_training_status()
        paper = self.get_paper_trading_status()
        advisor = self.get_advisor_status()
        gpu = self.get_gpu_status()
        system = self.get_system_health()
        watchdog = self.get_watchdog_status()

        # Render boxes
        if self.compact:
            # Compact mode: essential info only
            print(self.format_training_box(training, box_width))
            print()
            print(self.format_paper_trading_box(paper, box_width))
            print()
            print(self.format_gpu_box(gpu, box_width))
        else:
            # Full mode: all boxes
            print(self.format_training_box(training, box_width))
            print()
            print(self.format_paper_trading_box(paper, box_width))
            print()
            print(self.format_advisor_box(advisor, box_width))
            print()
            print(self.format_gpu_box(gpu, box_width))
            print()
            print(self.format_system_health_box(system, box_width))

        # Alerts
        self._print_alerts(training, paper, advisor, gpu, system, watchdog, box_width)

        # Footer
        print()
        print(self.colorize("─" * box_width, 'cyan'))
        print(self.colorize("Press Ctrl+C to exit | Refreshing every 3 seconds", 'white').center(box_width + 20))

    def _print_alerts(self, training, paper, advisor, gpu, system, watchdog, width):
        """Print system alerts."""
        alerts = []

        # Training alerts
        if not training.get('running'):
            alerts.append(("⚠️  TRAINING NOT RUNNING", 'red'))
        elif len(training.get('workers', [])) < 3:
            alerts.append((f"⚠️  Only {len(training['workers'])} training worker(s) running (expected 3)", 'yellow'))

        # Paper trading alerts
        if not paper.get('running'):
            alerts.append(("⚠️  PAPER TRADING NOT RUNNING", 'yellow'))

        # AI Advisor alerts
        if not advisor.get('running'):
            alerts.append(("⚠️  AI ADVISOR NOT RUNNING", 'yellow'))

        # GPU alerts
        if gpu.get('available'):
            if gpu['temp'] > 85:
                alerts.append((f"⚠️  HIGH GPU TEMPERATURE: {gpu['temp']}°C", 'red'))
            elif gpu['utilization'] < 50 and training.get('running'):
                alerts.append((f"⚠️  LOW GPU UTILIZATION: {gpu['utilization']}% (training is running)", 'yellow'))

        # System health alerts
        if 'disk_free_gb' in system:
            if system['disk_free_gb'] < 50:
                alerts.append((f"⚠️  LOW DISK SPACE: {system['disk_free_gb']:.1f}GB remaining", 'red'))
            if system['memory_percent'] > 90:
                alerts.append((f"⚠️  HIGH MEMORY USAGE: {system['memory_percent']:.1f}%", 'yellow'))

        # Print alerts if any
        if alerts:
            print()
            content = []
            for alert_text, color in alerts:
                content.append(self.colorize(alert_text, color))
            print(self.box("⚠️  ALERTS", content, width))

    def run(self, refresh_interval: int = 3):
        """Run dashboard with auto-refresh."""
        try:
            while True:
                self.render()
                time.sleep(refresh_interval)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Cappuccino Trading System Dashboard")
    parser.add_argument("--compact", action="store_true", help="Compact view")
    parser.add_argument("--interval", type=int, default=3, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Single refresh (no loop)")
    args = parser.parse_args()

    dashboard = CappuccinoDashboard(compact=args.compact)

    if args.once:
        # Single refresh mode
        dashboard.render()
    else:
        # Continuous refresh mode
        dashboard.run(refresh_interval=args.interval)


if __name__ == "__main__":
    main()
