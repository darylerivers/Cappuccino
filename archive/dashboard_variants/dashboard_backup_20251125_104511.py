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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


class CappuccinoDashboard:
    """Real-time dashboard for all cappuccino systems."""

    def __init__(self, compact: bool = False):
        self.compact = compact
        self.db_path = "databases/optuna_cappuccino.db"

        # Initialize Alpaca API if available
        self.alpaca_api = None
        if ALPACA_AVAILABLE:
            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_API_SECRET", "")
            if api_key and api_secret:
                try:
                    self.alpaca_api = tradeapi.REST(
                        api_key, api_secret,
                        "https://paper-api.alpaca.markets",
                        api_version='v2'
                    )
                except Exception:
                    pass

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
            # Check for regular training
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

            # Check for ensemble training
            ensemble_info = self.get_ensemble_status()

            return {
                'running': len(workers) > 0,
                'workers': workers,
                'trials': trial_info,
                'ensemble': ensemble_info,
            }
        except Exception as e:
            return {'running': False, 'error': str(e)}

    def get_ensemble_status(self) -> Dict:
        """Check if ensemble model exists and get info."""
        try:
            # Try JSON manifest first
            json_path = Path("train_results/ensemble/ensemble_manifest.json")
            if json_path.exists():
                import json
                with json_path.open("r") as f:
                    info = json.load(f)
                return {
                    'exists': True,
                    'model_count': info.get('model_count', 0),
                    'mean_value': info.get('mean_value', 0),
                    'best_value': info.get('best_value', 0),
                    'trial_numbers': info.get('trial_numbers', []),
                }

            # Fallback to pickle
            pickle_path = Path("train_results/ensemble/ensemble_info.pkl")
            if pickle_path.exists():
                import pickle
                with pickle_path.open("rb") as f:
                    info = pickle.load(f)
                return {
                    'exists': True,
                    'model_count': info.get('model_count', 0),
                    'mean_value': info.get('mean_value', 0),
                    'trial_numbers': info.get('trial_numbers', []),
                }
        except Exception:
            pass

        return {'exists': False}

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
            WHERE s.study_name = 'cappuccino_1year_20251121'
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

            # Get top 10% threshold for current study only
            top10_query = """
            SELECT value
            FROM trial_values tv
            JOIN trials t ON tv.trial_id = t.trial_id
            JOIN studies s ON t.study_id = s.study_id
            WHERE t.state = 'COMPLETE' AND s.study_name = 'cappuccino_1year_20251121'
            ORDER BY value DESC
            LIMIT 1 OFFSET (
                SELECT COUNT(*) / 10 FROM trials t2
                JOIN studies s2 ON t2.study_id = s2.study_id
                WHERE t2.state = 'COMPLETE' AND s2.study_name = 'cappuccino_1year_20251121'
            )
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

    def get_top_trials(self, limit: int = 5) -> List[Dict]:
        """Get top N trials from the current study."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
            SELECT t.number, tv.value, t.datetime_complete
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE s.study_name = 'cappuccino_1year_20251121'
            AND t.state = 'COMPLETE'
            ORDER BY tv.value DESC
            LIMIT ?
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            conn.close()

            return [{'number': r[0], 'value': r[1], 'completed': r[2]} for r in results]
        except Exception:
            return []

    def get_deployed_trial(self) -> Optional[int]:
        """Get the trial number currently deployed in paper trading."""
        try:
            import subprocess
            result = subprocess.run(
                ["pgrep", "-af", "paper_trader_alpaca"],
                capture_output=True, text=True
            )
            if result.stdout:
                # Parse command line for model-dir
                for line in result.stdout.strip().split('\n'):
                    if 'trial_' in line:
                        import re
                        match = re.search(r'trial_(\d+)', line)
                        if match:
                            return int(match.group(1))
        except Exception:
            pass
        return None

    def get_deployed_model_dir(self) -> Optional[str]:
        """Get the model directory currently used in paper trading."""
        try:
            import subprocess
            result = subprocess.run(
                ["pgrep", "-af", "paper_trader_alpaca"],
                capture_output=True, text=True
            )
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if '--model-dir' in line:
                        import re
                        match = re.search(r'--model-dir\s+(\S+)', line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return None

    def get_adaptive_ensemble_status(self, model_dir: str) -> Dict:
        """Get status of adaptive ensemble from model directory."""
        try:
            manifest_path = Path(model_dir) / "ensemble_manifest.json"
            if manifest_path.exists():
                import json
                with manifest_path.open("r") as f:
                    info = json.load(f)

                # Also check for model scores
                scores_path = Path(model_dir) / "model_scores.json"
                scores = {}
                if scores_path.exists():
                    with scores_path.open("r") as f:
                        scores = json.load(f)

                return {
                    'exists': True,
                    'adaptive': info.get('type') == 'adaptive',
                    'model_count': info.get('model_count', 0),
                    'mean_value': info.get('mean_value', 0),
                    'best_value': info.get('best_value', 0),
                    'trial_numbers': info.get('trial_numbers', []),
                    'study_name': info.get('study_name', 'unknown'),
                    'scores': scores,
                }
        except Exception:
            pass
        return {'exists': False}

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
            # Check for live log or most recent watchdog log
            log_file = Path("logs/paper_trading_live.log")
            if not log_file.exists():
                watchdog_logs = sorted(Path("logs").glob("paper_trading_watchdog_*.log"))
                if watchdog_logs:
                    log_file = watchdog_logs[-1]

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
                # Fall back to timestamped files (both alpaca_session and watchdog_session)
                csv_files = sorted(Path("paper_trades").glob("alpaca_session_*.csv"))
                watchdog_files = sorted(Path("paper_trades").glob("watchdog_session_*.csv"))
                all_files = csv_files + watchdog_files
                if all_files:
                    # Get most recent file by modification time
                    latest_csv = max(all_files, key=lambda p: p.stat().st_mtime)

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

                            # Calculate 24hr performance for alpha
                            # Find the row closest to 24 hours ago
                            from datetime import datetime, timedelta, timezone
                            try:
                                df['timestamp'] = pd.to_datetime(df['timestamp'])
                                now = datetime.now(timezone.utc)
                                target_time = now - timedelta(hours=24)

                                # Find row closest to 24 hours ago
                                df['time_diff'] = abs((df['timestamp'] - target_time).dt.total_seconds())
                                closest_idx = df['time_diff'].idxmin()

                                if closest_idx is not None and not pd.isna(df.loc[closest_idx, 'total_asset']):
                                    total_asset_24h_ago = float(df.loc[closest_idx, 'total_asset'])
                                    info['total_asset_24h_ago'] = total_asset_24h_ago
                            except Exception:
                                pass

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

    def _load_position_state(self) -> Dict:
        """Load position state from JSON file containing stop-loss levels."""
        try:
            positions_file = Path("paper_trades/positions_state.json")
            if positions_file.exists():
                with positions_file.open() as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _get_stop_loss_info(self, ticker: str, position_state: Dict) -> Optional[Dict]:
        """Extract stop-loss info for a specific ticker from position state.

        Returns the EFFECTIVE stop-loss (whichever is closer to current price):
        - Initial stop-loss (10% below entry)
        - Trailing stop-loss (1.5% below peak)
        """
        try:
            positions = position_state.get("positions", [])
            for pos in positions:
                if pos.get("ticker") == ticker:
                    initial_stop = pos.get('stop_loss_price', 0)
                    trailing_stop = pos.get('trailing_stop_price', 0)
                    current_price = pos.get('current_price', 0)

                    # Use whichever stop-loss is HIGHER (closer to current price)
                    # This is the one that would trigger first
                    effective_stop = max(initial_stop, trailing_stop)

                    # Calculate distance to effective stop
                    if effective_stop > 0 and current_price > 0:
                        distance_pct = ((current_price - effective_stop) / current_price) * 100
                    else:
                        distance_pct = 0

                    # Determine which stop is active
                    stop_type = "trail" if trailing_stop > initial_stop else "fixed"

                    return {
                        'stop_loss_price': effective_stop,
                        'distance_pct': distance_pct,
                        'entry_price': pos.get('entry_price', 0),
                        'stop_type': stop_type,
                        'high_water_mark': pos.get('high_water_mark', 0),
                    }
        except Exception:
            pass
        return None

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

    def get_market_data(self) -> Dict:
        """Get market data for all tradeable assets."""
        if not self.alpaca_api:
            return {'available': False}

        try:
            from config_main import TICKER_LIST
            from datetime import timedelta, timezone as dt_timezone

            end = datetime.now(dt_timezone.utc)
            start_24h = end - timedelta(hours=24)
            start_25h = end - timedelta(hours=25)

            assets = []
            for symbol in TICKER_LIST:
                try:
                    bars = self.alpaca_api.get_crypto_bars(
                        symbol,
                        TimeFrame.Hour,
                        start_25h.isoformat(),
                        end.isoformat(),
                    ).df

                    if bars.empty:
                        continue

                    latest_price = float(bars['close'].iloc[-1])

                    if len(bars) >= 24:
                        price_24h_ago = float(bars['close'].iloc[-24])
                    elif len(bars) >= 2:
                        price_24h_ago = float(bars['close'].iloc[0])
                    else:
                        price_24h_ago = latest_price

                    change_24h = latest_price - price_24h_ago
                    change_pct_24h = (change_24h / price_24h_ago * 100) if price_24h_ago > 0 else 0

                    assets.append({
                        'symbol': symbol,
                        'price': latest_price,
                        'change_24h': change_24h,
                        'change_pct_24h': change_pct_24h,
                    })
                except Exception:
                    continue

            # Calculate market average change for alpha calculation
            avg_change = sum(a['change_pct_24h'] for a in assets) / len(assets) if assets else 0

            return {
                'available': True,
                'assets': assets,
                'avg_change_24h': avg_change,
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

    def format_training_box(self, status: Dict, width: int) -> str:
        """Format training status box."""
        content = []

        # Show current study name
        content.append(self.colorize("Study: cappuccino_1year_20251121", 'cyan'))
        content.append("")

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

            # Show top trials
            top_trials = self.get_top_trials(5)
            deployed = self.get_deployed_trial()
            if top_trials:
                content.append("")
                content.append(self.colorize("Top 5 Trials:", 'bold'))
                for t in top_trials:
                    marker = " *" if t['number'] == deployed else ""
                    value_color = 'green' if t['value'] > 0 else 'red'
                    value_str = self.colorize(f"{t['value']:+.6f}", value_color)
                    content.append(f"  #{t['number']:>3}: {value_str}{marker}")
                if deployed:
                    content.append(f"  (* = deployed in paper trading)")
        else:
            content.append(self.colorize("Status: NOT RUNNING", 'red'))

            trials = status.get('trials', {})
            if trials:
                content.append("")
                total = trials.get('completed', 0) + trials.get('failed', 0)
                completed_str = self.colorize(str(trials.get('completed', 0)), 'cyan')
                content.append(f"Total Trials: {total} | Completed: {completed_str} | Failed: {trials.get('failed', 0)}")

                # Best value
                best_val = f"{trials.get('best_value', 0):.6f}"
                content.append(f"Best Value: {self.colorize(best_val, 'green')}")

        return self.box("TRAINING", content, width)

    def format_paper_trading_box(self, status: Dict, width: int, market_data: Optional[Dict] = None) -> str:
        """Format paper trading status box."""
        content = []

        # Show deployed model info
        deployed_trial = self.get_deployed_trial()
        if deployed_trial:
            # Get trial value from top trials
            top_trials = self.get_top_trials(10)
            trial_value = None
            trial_rank = None
            for i, t in enumerate(top_trials):
                if t['number'] == deployed_trial:
                    trial_value = t['value']
                    trial_rank = i + 1
                    break

            if trial_value is not None:
                value_color = 'green' if trial_value > 0 else 'red'
                content.append(f"Model: Trial #{deployed_trial} (rank #{trial_rank}, value: {self.colorize(f'{trial_value:+.6f}', value_color)})")
            else:
                content.append(f"Model: Trial #{deployed_trial}")
            content.append("")
        else:
            # Check for ensemble - first get the actual model directory being used
            model_dir = self.get_deployed_model_dir()
            if model_dir and 'ensemble' in model_dir:
                ensemble = self.get_adaptive_ensemble_status(model_dir)
                if ensemble.get('exists'):
                    model_count = ensemble.get('model_count', 0)
                    trial_nums = ensemble.get('trial_numbers', [])[:5]
                    study_name = ensemble.get('study_name', 'unknown')

                    if ensemble.get('adaptive'):
                        content.append(f"Model: {self.colorize('Adaptive Ensemble', 'cyan')} ({model_count} models, game theory)")
                        content.append(f"  Study: {study_name}")
                        content.append(f"  Trials: {', '.join(f'#{n}' for n in trial_nums)}")

                        # Show model scores if available
                        scores = ensemble.get('scores', {})
                        if scores:
                            active_scores = [(int(t), s.get('score', 0.5)) for t, s in scores.items()]
                            active_scores.sort(key=lambda x: x[1], reverse=True)
                            if active_scores:
                                best = active_scores[0]
                                worst = active_scores[-1]
                                content.append(f"  Best: #{best[0]} ({best[1]:.1%}) | Worst: #{worst[0]} ({worst[1]:.1%})")
                    else:
                        content.append(f"Model: {self.colorize('Ensemble', 'cyan')} ({model_count} models)")
                        content.append(f"  Trials: {', '.join(f'#{n}' for n in trial_nums)}...")
                    content.append("")
            else:
                # Fallback to old ensemble check
                ensemble = self.get_ensemble_status()
                if ensemble.get('exists'):
                    model_count = ensemble.get('model_count', 0)
                    trial_nums = ensemble.get('trial_numbers', [])[:3]
                    content.append(f"Model: {self.colorize('Ensemble', 'cyan')} ({model_count} models)")
                    content.append(f"  Trials: {', '.join(f'#{n}' for n in trial_nums)}...")
                    content.append("")

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
                    positions = status.get('positions', {})

                    # Calculate current positions value using live market prices
                    positions_value_current = 0
                    current_prices = {}

                    if market_data and market_data.get('available'):
                        # Build price lookup from live market data
                        current_prices = {asset['symbol']: asset['price']
                                        for asset in market_data.get('assets', [])}

                    # Revalue positions at current market prices
                    if positions:
                        for ticker, pos in positions.items():
                            shares = pos['shares']
                            current_price = current_prices.get(ticker, pos['price'])  # Use live price or fallback
                            positions_value_current += shares * current_price
                    else:
                        positions_value_current = status.get('positions_value', 0)

                    # Calculate current total with live prices
                    total = cash + positions_value_current
                    pnl = total - 1000  # Starting with $1K in paper trading
                    pnl_color = 'green' if pnl >= 0 else 'red'
                    pnl_pct = (pnl / 1000) * 100

                    content.append(f"Cash: ${cash:,.2f}")

                    # Show positions if any
                    if positions:
                        content.append(f"Positions Value: ${positions_value_current:,.2f} (live)")

                        # Load position state for stop-loss levels
                        position_state = self._load_position_state()

                        # Show top 3 positions
                        sorted_positions = sorted(positions.items(), key=lambda x: x[1]['value'], reverse=True)
                        for ticker, pos in sorted_positions[:3]:
                            shares = pos['shares']
                            logged_price = pos['price']  # Price from CSV (last poll)
                            logged_value = pos['value']

                            # Get current market price
                            current_price = current_prices.get(ticker, logged_price)
                            current_value = shares * current_price

                            # Get stop-loss info from position state
                            stop_info = self._get_stop_loss_info(ticker, position_state)

                            # Show both logged and current if different
                            if abs(current_price - logged_price) > 0.01:
                                pos_pnl_pct = ((current_price - logged_price) / logged_price * 100)
                                pos_pnl_color = 'green' if pos_pnl_pct >= 0 else 'red'
                                pos_pnl_str = self.colorize(f"{pos_pnl_pct:+.2f}%", pos_pnl_color)

                                # Build position line with stop-loss info
                                pos_line = f"  {ticker}: {shares:.4f} × ${logged_price:.2f} → ${current_price:.2f} ({pos_pnl_str}) = ${current_value:.2f}"

                                if stop_info:
                                    stop_price = stop_info['stop_loss_price']
                                    distance = stop_info['distance_pct']
                                    stop_type = stop_info.get('stop_type', 'fixed')
                                    distance_color = 'red' if distance < 2 else ('yellow' if distance < 5 else 'green')
                                    distance_str = self.colorize(f"{distance:.1f}%", distance_color)
                                    stop_label = "Trail" if stop_type == "trail" else "Stop"
                                    pos_line += f" | {stop_label}: ${stop_price:.2f} ({distance_str})"

                                content.append(pos_line)
                            else:
                                pos_line = f"  {ticker}: {shares:.4f} @ ${logged_price:.2f} = ${current_value:.2f}"

                                if stop_info:
                                    stop_price = stop_info['stop_loss_price']
                                    distance = stop_info['distance_pct']
                                    stop_type = stop_info.get('stop_type', 'fixed')
                                    distance_color = 'red' if distance < 2 else ('yellow' if distance < 5 else 'green')
                                    distance_str = self.colorize(f"{distance:.1f}%", distance_color)
                                    stop_label = "Trail" if stop_type == "trail" else "Stop"
                                    pos_line += f" | {stop_label}: ${stop_price:.2f} ({distance_str})"

                                content.append(pos_line)

                    pnl_str = self.colorize(f'${pnl:+,.2f}', pnl_color)
                    pnl_pct_str = self.colorize(f'{pnl_pct:+.2f}%', pnl_color)
                    content.append(f"Total: ${total:,.2f} | P&L (Session): {pnl_str} ({pnl_pct_str})")

                    # Calculate alpha using 24hr performance if available
                    if hasattr(self, '_market_avg_change') and 'total_asset_24h_ago' in status:
                        market_avg = self._market_avg_change
                        total_24h_ago = status['total_asset_24h_ago']

                        # Calculate model's 24hr return
                        model_24h_change = total - total_24h_ago
                        model_24h_pct = (model_24h_change / total_24h_ago * 100) if total_24h_ago > 0 else 0

                        # Alpha = model's 24hr return - market's 24hr return
                        alpha = model_24h_pct - market_avg
                        alpha_color = 'green' if alpha >= 0 else 'red'
                        alpha_str = self.colorize(f'{alpha:+.2f}%', alpha_color)
                        model_str = self.colorize(f'{model_24h_pct:+.2f}%', 'green' if model_24h_pct >= 0 else 'red')
                        content.append(f"24hr Performance: {model_str} | Market: {market_avg:+.2f}% | Alpha: {alpha_str}")
                    elif hasattr(self, '_market_avg_change'):
                        # Fallback: show session-based alpha with warning
                        market_avg = self._market_avg_change
                        alpha = pnl_pct - market_avg
                        alpha_color = 'green' if alpha >= 0 else 'red'
                        alpha_str = self.colorize(f'{alpha:+.2f}%', alpha_color)
                        content.append(f"Alpha (session vs 24hr market): {alpha_str} [different periods]")

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

    def format_market_grid_box(self, market_data: Dict, width: int) -> str:
        """Format market data grid box."""
        content = []

        if not market_data.get('available'):
            content.append(self.colorize("Market data unavailable", 'yellow'))
            if 'error' in market_data:
                content.append(f"Error: {market_data['error']}")
            return self.box("MARKET OVERVIEW", content, width)

        assets = market_data.get('assets', [])
        if not assets:
            content.append("No asset data")
            return self.box("MARKET OVERVIEW", content, width)

        # Sort by 24hr change (best performers first)
        assets_sorted = sorted(assets, key=lambda x: x['change_pct_24h'], reverse=True)

        # Display in a compact grid format
        content.append(self.colorize(f"{'Asset':<12} {'Price':>14} {'24hr Change':>20}", 'bold'))
        content.append("─" * (width - 4))

        for asset in assets_sorted:
            symbol = asset['symbol'].replace('/USD', '')  # Shorten display
            price = asset['price']
            change_pct = asset['change_pct_24h']

            # Format price
            if price >= 1000:
                price_str = f"${price:>12,.2f}"
            elif price >= 1:
                price_str = f"${price:>12,.4f}"
            else:
                price_str = f"${price:>12,.6f}"

            # Format change with color
            change_color = 'green' if change_pct >= 0 else 'red'
            change_str = self.colorize(f"{change_pct:+6.2f}%", change_color)

            content.append(f"{symbol:<12} {price_str:>14} {change_str:>20}")

        # Market summary
        avg_change = market_data.get('avg_change_24h', 0)
        positive_count = sum(1 for a in assets if a['change_24h'] > 0)
        negative_count = sum(1 for a in assets if a['change_24h'] < 0)

        content.append("─" * (width - 4))
        avg_color = 'green' if avg_change >= 0 else 'red'
        avg_str = self.colorize(f"{avg_change:+.2f}%", avg_color)
        content.append(f"Market Avg: {avg_str} | Up: {self.colorize(str(positive_count), 'green')} | Down: {self.colorize(str(negative_count), 'red')}")

        return self.box("MARKET OVERVIEW", content, width)

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
        market = self.get_market_data()

        # Store market average for alpha calculation
        if market.get('available'):
            self._market_avg_change = market.get('avg_change_24h', 0)

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
            print(self.format_paper_trading_box(paper, box_width, market_data=market))
            print()
            print(self.format_market_grid_box(market, box_width))
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
