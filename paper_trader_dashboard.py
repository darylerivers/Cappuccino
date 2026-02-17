#!/usr/bin/env python3
"""
Paper Trader Live Dashboard

Real-time monitoring of paper traders with countdown to next update.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️  Rich library not installed. Install with: pip install rich")
    sys.exit(1)


class PaperTraderDashboard:
    def __init__(self, poll_interval: int = 3600, show_training: bool = True,
                 training_study_name: Optional[str] = None):
        self.console = Console()
        self.poll_interval = poll_interval  # Seconds between trader polls
        self.show_training = show_training
        self.training_db_path = "databases/optuna_cappuccino.db"
        self.training_study_name = training_study_name  # None = auto-detect
        self._last_study_check = 0

    def _discover_active_study(self, storage: str) -> Optional[str]:
        """Auto-discover the most recently active Optuna study."""
        import optuna
        from datetime import datetime
        import sqlite3

        all_studies = optuna.study.get_all_study_names(storage)
        valid_studies = [s for s in all_studies if s and s.strip()]
        if not valid_studies:
            return None

        # Use SQL directly to avoid deepcopy issues with large trials
        db_path = storage.replace('sqlite:///', '')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        best_study = None
        best_score = (-1, 0, datetime.min)  # (running_count, is_aggressive, recent_time)

        for study_name in valid_studies:
            try:
                # Get study_id
                cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
                row = cursor.fetchone()
                if not row:
                    continue
                study_id = row[0]

                # Count running trials (state is stored as string 'RUNNING')
                cursor.execute(
                    "SELECT COUNT(*) FROM trials WHERE study_id = ? AND state = 'RUNNING'",
                    (study_id,)
                )
                running_count = cursor.fetchone()[0]

                # Get most recent activity
                cursor.execute(
                    "SELECT MAX(COALESCE(datetime_complete, datetime_start)) FROM trials WHERE study_id = ?",
                    (study_id,)
                )
                recent_time_str = cursor.fetchone()[0]
                if recent_time_str:
                    recent_time = datetime.fromisoformat(recent_time_str.replace(' ', 'T'))
                else:
                    recent_time = datetime.min

                # Prefer "aggressive" studies over others
                is_aggressive = 1 if 'aggressive' in study_name else 0

                # Score: prioritize running trials, then aggressive studies, then recent activity
                score = (running_count, is_aggressive, recent_time)
                if score > best_score:
                    best_score = score
                    best_study = study_name
            except Exception as e:
                continue

        conn.close()
        return best_study

    def _get_parallel_study_names(self, base_name: str, storage: str) -> List[str]:
        """Find all parallel studies matching a base name pattern (e.g. study_1, study_2, study_3)."""
        import optuna
        all_studies = optuna.study.get_all_study_names(storage)
        # Strip trailing _N to get prefix
        import re
        match = re.match(r'^(.+?)_(\d+)$', base_name)
        if match:
            prefix = match.group(1)
            return [s for s in all_studies if s.startswith(prefix + '_') and re.match(r'^' + re.escape(prefix) + r'_\d+$', s)]
        return [base_name]

    def get_training_progress(self) -> Optional[Dict]:
        """Get training progress from Optuna database using SQL directly to avoid distribution errors."""
        try:
            import sqlite3
            from datetime import datetime

            db_path = Path(self.training_db_path)
            if not db_path.exists():
                return None

            storage = f'sqlite:///{db_path}'

            # Auto-discover active study every 10 seconds (more responsive)
            now = time.time()
            if self.training_study_name is None or (now - self._last_study_check > 10):
                discovered = self._discover_active_study(storage)
                # ALWAYS update, even if None (clear dead studies)
                self.training_study_name = discovered
                self._last_study_check = now

            if not self.training_study_name:
                return None

            # Find all parallel studies with the same prefix
            study_names = self._get_parallel_study_names(self.training_study_name, storage)

            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Aggregate stats across all parallel studies
            total_complete = 0
            total_running = 0
            total_failed = 0
            total_trials = 0
            best_value = None
            best_trial_num = None
            best_study_name = None
            all_complete_times = []
            running_details = []

            for study_name in study_names:
                cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
                row = cursor.fetchone()
                if not row:
                    continue
                study_id = row[0]

                # Count by state
                cursor.execute("SELECT state, COUNT(*) FROM trials WHERE study_id = ? GROUP BY state", (study_id,))
                for state, count in cursor.fetchall():
                    if state == 'COMPLETE':
                        total_complete += count
                    elif state == 'RUNNING':
                        total_running += count
                    elif state == 'FAIL':
                        total_failed += count
                    total_trials += count

                # Best completed trial value
                cursor.execute("""
                    SELECT trial_id, number FROM trials
                    WHERE study_id = ? AND state = 'COMPLETE'
                    ORDER BY number
                """, (study_id,))
                for trial_id, trial_num in cursor.fetchall():
                    cursor.execute("SELECT value FROM trial_values WHERE trial_id = ? AND objective = 0", (trial_id,))
                    val_row = cursor.fetchone()
                    if val_row and val_row[0] is not None:
                        val = val_row[0]
                        if best_value is None or val > best_value:
                            best_value = val
                            best_trial_num = trial_num
                            best_study_name = study_name

                # Completed trial timestamps for ETA calc
                cursor.execute("""
                    SELECT datetime_complete FROM trials
                    WHERE study_id = ? AND state = 'COMPLETE' AND datetime_complete IS NOT NULL
                """, (study_id,))
                for (dt_str,) in cursor.fetchall():
                    try:
                        all_complete_times.append(datetime.fromisoformat(dt_str.replace(' ', 'T')))
                    except (ValueError, AttributeError):
                        pass

                # Running trial details
                cursor.execute("""
                    SELECT trial_id, number, datetime_start FROM trials
                    WHERE study_id = ? AND state = 'RUNNING'
                """, (study_id,))
                for trial_id, trial_num, dt_start_str in cursor.fetchall():
                    elapsed = 0
                    if dt_start_str:
                        try:
                            dt_start = datetime.fromisoformat(dt_start_str.replace(' ', 'T'))
                            elapsed = (datetime.now() - dt_start).total_seconds() / 3600
                        except (ValueError, AttributeError):
                            pass

                    # Get trial params
                    cursor.execute("""
                        SELECT param_name, param_value FROM trial_params WHERE trial_id = ?
                    """, (trial_id,))
                    params = {name: val for name, val in cursor.fetchall()}

                    running_details.append({
                        'number': trial_num,
                        'study': study_name,
                        'elapsed_hours': elapsed,
                        'best_sharpe': None,
                        'batch_size': params.get('batch_size'),
                        'net_dimension': params.get('net_dimension'),
                    })

            conn.close()

            # Calculate ETA
            eta_str = "Calculating..."
            trials_per_hour = 0
            target_trials = 501  # 3 studies × 167 trials

            if len(all_complete_times) >= 2:
                start_time = min(all_complete_times)
                end_time = max(all_complete_times)
                hours_elapsed = (end_time - start_time).total_seconds() / 3600
                if hours_elapsed > 0:
                    trials_per_hour = total_complete / hours_elapsed
                    remaining = target_trials - total_trials
                    if remaining > 0 and trials_per_hour > 0:
                        hours_remaining = remaining / trials_per_hour
                        days = int(hours_remaining // 24)
                        hours = int(hours_remaining % 24)
                        eta_str = f"{days}d {hours}h" if days > 0 else f"{hours}h {int((hours_remaining % 1) * 60)}m"

            elif total_running > 0 and running_details:
                avg_elapsed = sum(d['elapsed_hours'] for d in running_details) / len(running_details)
                if avg_elapsed > 0.5:
                    est_rate = total_running / avg_elapsed
                    remaining = target_trials - total_trials
                    if remaining > 0:
                        hours_remaining = remaining / est_rate
                        days = int(hours_remaining // 24)
                        hours = int(hours_remaining % 24)
                        eta_str = f"~{days}d {hours}h" if days > 0 else f"~{hours}h {int((hours_remaining % 1) * 60)}m"
                    trials_per_hour = est_rate
                else:
                    eta_str = "Waiting for data..."
            elif total_running > 0:
                eta_str = "Waiting for data..."

            display_name = ', '.join(study_names) if len(study_names) <= 3 else f"{study_names[0]} (+{len(study_names)-1})"

            return {
                'study_name': display_name,
                'total_trials': total_trials,
                'complete': total_complete,
                'running': total_running,
                'failed': total_failed,
                'best_trial': best_trial_num,
                'best_value': best_value,
                'current_trial': running_details[0]['number'] if running_details else None,
                'trials_per_hour': trials_per_hour,
                'eta': eta_str,
                'target_trials': target_trials,
                'running_details': running_details
            }

        except Exception as e:
            return None

    def get_active_traders(self) -> List[Tuple[str, str]]:
        """Find active paper trader CSV files."""
        traders = []

        # Find all session CSV files (trial or ensemble)
        csv_files = sorted(
            Path('paper_trades').glob('*_session.csv'),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        for csv in csv_files:
            # Check if file is recent (modified in last 2 hours for active monitoring)
            age_hours = (time.time() - csv.stat().st_mtime) / 3600
            if age_hours < 2:
                trial_name = csv.stem.replace('_session', '').replace('trial', 'Trial #')
                traders.append((trial_name, str(csv)))

        return traders

    def calculate_next_poll_time(self, csv_path: str) -> Optional[datetime]:
        """Calculate when the next poll should happen.

        Reads from heartbeat file for accurate countdown, falls back to CSV estimation.
        """
        import json

        # Try to read from heartbeat file (most accurate)
        try:
            heartbeat_name = Path(csv_path).stem.replace('_session', '_heartbeat.json')
            heartbeat_path = Path('paper_trades') / heartbeat_name

            if heartbeat_path.exists():
                with heartbeat_path.open('r') as f:
                    heartbeat = json.load(f)

                if 'next_poll_scheduled' in heartbeat and heartbeat['next_poll_scheduled']:
                    return pd.to_datetime(heartbeat['next_poll_scheduled'])
        except Exception as e:
            pass  # Fall back to CSV estimation

        # Fallback: estimate from CSV last timestamp + poll interval
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return None

            # Get last timestamp
            last_ts = pd.to_datetime(df['timestamp'].iloc[-1])

            # Next poll is last timestamp + poll interval
            next_poll = last_ts + timedelta(seconds=self.poll_interval)

            return next_poll

        except Exception as e:
            return None

    def calculate_metrics(self, csv_path: str) -> Optional[Dict]:
        """Calculate trader metrics from CSV."""
        try:
            df = pd.read_csv(csv_path)

            if len(df) <= 1:
                return None

            returns = df['total_asset'].pct_change().dropna()

            if len(returns) == 0 or returns.std() == 0:
                return None

            # Overall Sharpe
            sharpe_overall = (returns.mean() / returns.std()) * np.sqrt(365 * 24)

            # Expanding window Sharpe (average)
            expanding_sharpes = []
            for i in range(3, len(returns) + 1):
                window_returns = returns.iloc[:i]
                if window_returns.std() > 0:
                    window_sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(365 * 24)
                    expanding_sharpes.append(window_sharpe)
            sharpe_average = np.mean(expanding_sharpes) if expanding_sharpes else None

            # First and last periods
            window_size = min(10, len(returns) // 3)

            initial_returns = returns.iloc[:window_size]
            sharpe_initial = (initial_returns.mean() / initial_returns.std()) * np.sqrt(365 * 24) if initial_returns.std() > 0 else None

            recent_returns = returns.iloc[-window_size:]
            sharpe_recent = (recent_returns.mean() / recent_returns.std()) * np.sqrt(365 * 24) if recent_returns.std() > 0 else None

            # Last bar impact
            sharpe_previous = None
            if len(returns) >= 4:
                prev_returns = returns.iloc[:-1]
                if prev_returns.std() > 0:
                    sharpe_previous = (prev_returns.mean() / prev_returns.std()) * np.sqrt(365 * 24)

            # Portfolio
            initial_value = df['total_asset'].iloc[0]
            current_value = df['total_asset'].iloc[-1]
            cash = df['cash'].iloc[-1]
            total_return = ((current_value - initial_value) / initial_value) * 100

            # Positions and live prices
            latest = df.iloc[-1]
            tickers = ['AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD', 'ETH/USD', 'LTC/USD', 'UNI/USD']
            positions = {}
            live_prices = {}

            for ticker in tickers:
                holding_col = f'holding_{ticker}'
                price_col = f'price_{ticker}'

                # Store live price for all tickers
                if price_col in latest:
                    live_prices[ticker] = latest[price_col]

                # Store position details for held assets
                if holding_col in latest and latest[holding_col] > 0.0001:
                    positions[ticker] = {
                        'quantity': latest[holding_col],
                        'price': latest[price_col],
                        'value': latest[holding_col] * latest[price_col]
                    }

            # Calculate alpha vs equal-weight benchmark
            # Equal-weight benchmark: buy equal amounts of each asset at start, hold
            alpha = None
            benchmark_return = None
            try:
                first_row = df.iloc[0]
                initial_prices = {ticker: first_row[f'price_{ticker}'] for ticker in tickers if f'price_{ticker}' in first_row}
                current_prices = {ticker: latest[f'price_{ticker}'] for ticker in tickers if f'price_{ticker}' in latest}

                if initial_prices and current_prices:
                    # Calculate equal-weight return
                    price_returns = [(current_prices[t] / initial_prices[t] - 1) for t in tickers
                                    if t in initial_prices and t in current_prices]
                    if price_returns:
                        benchmark_return = np.mean(price_returns) * 100  # Average return across assets
                        alpha = total_return - benchmark_return  # Portfolio return - benchmark return
            except Exception as e:
                pass  # Alpha calculation optional

            return {
                'bars': len(df),
                'initial_value': initial_value,
                'current_value': current_value,
                'cash': cash,
                'total_return': total_return,
                'sharpe_overall': sharpe_overall,
                'sharpe_average': sharpe_average,
                'sharpe_initial': sharpe_initial,
                'sharpe_recent': sharpe_recent,
                'sharpe_previous': sharpe_previous,
                'sharpe_change': sharpe_overall - sharpe_previous if sharpe_previous else None,
                'positions': positions,
                'live_prices': live_prices,
                'benchmark_return': benchmark_return,
                'alpha': alpha,
                'window_size': window_size,
                'timestamp': latest['timestamp']
            }
        except Exception as e:
            return None

    def format_countdown(self, seconds: float) -> Tuple[str, str]:
        """Format countdown timer with color."""
        if seconds < 0:
            return "[red]OVERDUE![/red]", "red"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            time_str = f"{minutes}m {secs}s"
        else:
            time_str = f"{secs}s"

        # Color based on time remaining
        if seconds < 60:
            color = "yellow"
        elif seconds < 300:
            color = "cyan"
        else:
            color = "green"

        return f"[{color}]{time_str}[/{color}]", color

    def create_trader_table(self, name: str, metrics: Dict, next_poll: Optional[datetime]) -> Table:
        """Create a table for a single trader."""
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 0), collapse_padding=True)
        table.add_column("Label", style="bold cyan", no_wrap=True)
        table.add_column("Value", no_wrap=True)

        # Countdown
        if next_poll:
            now = datetime.now(timezone.utc)
            seconds_remaining = (next_poll - now).total_seconds()
            countdown_str, color = self.format_countdown(seconds_remaining)
            table.add_row("Poll", countdown_str)

        # Portfolio (compact)
        return_color = "green" if metrics['total_return'] >= 0 else "red"
        table.add_row("Value", f"${metrics['current_value']:.0f} [{return_color}]{metrics['total_return']:+.1f}%[/{return_color}]")

        # Sharpe (compact)
        sharpe_color = "green" if metrics['sharpe_overall'] > 0 else "red"
        table.add_row("Sharpe", f"[{sharpe_color}]{metrics['sharpe_overall']:.3f}[/{sharpe_color}]")

        # Last bar impact
        if metrics['sharpe_change']:
            change_color = "green" if metrics['sharpe_change'] > 0 else "red"
            symbol = "↑" if metrics['sharpe_change'] > 0 else "↓"
            table.add_row("⚡ Last Bar Impact", f"[{change_color}]{symbol} {metrics['sharpe_change']:+.4f}[/{change_color}]")

        # Improvement trend
        if metrics['sharpe_initial'] and metrics['sharpe_recent']:
            improvement = metrics['sharpe_recent'] - metrics['sharpe_initial']
            improvement_pct = (improvement / abs(metrics['sharpe_initial']) * 100) if metrics['sharpe_initial'] != 0 else 0
            trend_color = "green" if improvement > 0 else "red"
            trend_symbol = "↑" if improvement > 0 else "↓"
            table.add_row("📊 Trend", f"[{trend_color}]{trend_symbol} {improvement:+.2f} ({improvement_pct:+.1f}%)[/{trend_color}]")

        # Alpha (vs equal-weight benchmark)
        if metrics.get('alpha') is not None and metrics.get('benchmark_return') is not None:
            alpha_color = "green" if metrics['alpha'] >= 0 else "red"
            table.add_row("🎯 Alpha", f"[{alpha_color}]{metrics['alpha']:+.2f}%[/{alpha_color}]")
            bench_color = "green" if metrics['benchmark_return'] >= 0 else "red"
            table.add_row("📊 Benchmark", f"[{bench_color}]{metrics['benchmark_return']:+.2f}%[/{bench_color}]")

        # Positions
        if metrics['positions']:
            pos_text = ""
            for ticker, data in list(metrics['positions'].items())[:3]:  # Show top 3
                ticker_short = ticker.split('/')[0]
                pos_text += f"{ticker_short}: {data['quantity']:.4f} @ ${data['price']:.2f}\n"
            table.add_row("💼 Positions", pos_text.strip())
        else:
            table.add_row("💼 Positions", "[yellow]100% Cash[/yellow]")

        # Live Prices (show top 4 by price change or holdings)
        if metrics.get('live_prices'):
            price_text = ""
            # Prioritize assets we hold, then show others
            shown = 0
            for ticker in metrics['positions'].keys():
                if shown >= 4:
                    break
                ticker_short = ticker.split('/')[0]
                price = metrics['live_prices'].get(ticker, 0)
                price_text += f"{ticker_short}: ${price:.2f}  "
                shown += 1

            # Fill remaining with assets we don't hold
            if shown < 4:
                for ticker, price in list(metrics['live_prices'].items()):
                    if shown >= 4:
                        break
                    if ticker not in metrics['positions']:
                        ticker_short = ticker.split('/')[0]
                        price_text += f"{ticker_short}: ${price:.2f}  "
                        shown += 1

            if price_text:
                table.add_row("💲 Live Prices", price_text.strip())

        table.add_row("📊 Bars", str(metrics['bars']))

        return table

    def get_training_logs(self, lines: int = 25) -> List[str]:
        """Get recent useful training logs from active study."""
        import subprocess
        try:
            log_dir = Path('/opt/user-data/experiment/cappuccino/logs')

            # Find most recent aggressive worker log
            log_files = list(log_dir.glob('worker_aggressive_*.log'))
            if not log_files:
                log_files = list(log_dir.glob('worker_*.log'))

            if not log_files:
                return ["[yellow]No training logs found[/yellow]"]

            # Use most recently modified
            log_file = max(log_files, key=lambda f: f.stat().st_mtime)

            result = subprocess.run(
                ['tail', '-n', str(lines * 4), str(log_file)],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout:
                lines_list = result.stdout.strip().split('\n')

                # Filter to useful lines only
                filtered = []
                for line in lines_list:
                    # Skip errors and tracebacks
                    if any(skip in line for skip in ['Traceback', 'File "/', 'sqlalchemy', 'raise ', 'Error:', 'exception']):
                        continue
                    # Keep progress indicators
                    if any(keep in line for keep in ['Split', 'Trial', 'Sharpe', 'Return', 'GPU', 'Complete', 'BATCH', 'Using', 'Train Samples', 'Test Samples', '======']):
                        # Clean ANSI codes if present
                        clean = line.replace('\x1b[93m', '').replace('\x1b[0m', '').replace('\x1b[96m', '').replace('\x1b[92m', '')
                        filtered.append(clean)

                return filtered[-lines:] if filtered else ["[dim]Initializing training...[/dim]"]
            return ["[yellow]Error reading logs[/yellow]"]
        except Exception as e:
            return ["[yellow]No logs available[/yellow]"]

    def create_training_logs_panel(self, lines: int = 12) -> Panel:
        """Create a panel showing recent training logs."""
        logs = self.get_training_logs(lines)
        log_text = "\n".join(logs[-lines:])  # Show last N lines

        return Panel(
            Text(log_text, style="dim white", overflow="ellipsis"),
            title="Logs",
            border_style="blue",
            padding=(0, 0)
        )

    def create_training_panel(self, training_info: Dict) -> Panel:
        """Create a panel showing training progress with GPU info."""
        import subprocess
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 0), collapse_padding=True)
        table.add_column("Label", style="bold cyan", no_wrap=True)
        table.add_column("Value", no_wrap=True)

        # Study name (compact)
        study_parts = training_info['study_name'].split(', ')
        if len(study_parts) > 1:
            base = study_parts[0].rsplit('_', 1)[0]
            table.add_row("📚 Studies", f"[cyan]{base.split('_')[-2:]}[/cyan] ×{len(study_parts)}")

        # Progress
        progress_pct = (training_info['complete'] / training_info['target_trials']) * 100
        progress_color = "green" if progress_pct > 50 else "yellow" if progress_pct > 20 else "cyan"
        table.add_row("🎯 Progress", f"[{progress_color}]{training_info['complete']}/{training_info['target_trials']} ({progress_pct:.1f}%)[/{progress_color}]")

        # GPU Usage
        try:
            gpu_result = subprocess.run(['rocm-smi', '--showuse'], capture_output=True, text=True, timeout=1)
            if gpu_result.returncode == 0:
                for line in gpu_result.stdout.split('\n'):
                    if 'GPU[0]' in line and 'GPU use' in line:
                        gpu_pct = int(line.split(':')[-1].strip())
                        gpu_color = "green" if gpu_pct > 75 else "yellow" if gpu_pct > 50 else "red"
                        table.add_row("🎮 GPU", f"[{gpu_color}]{gpu_pct}%[/{gpu_color}]")
                        break
        except:
            pass

        # Speed and ETA
        if training_info['trials_per_hour'] > 0:
            table.add_row("⚡ Speed", f"[cyan]{training_info['trials_per_hour']:.1f}[/cyan] trials/h")
        eta_color = "green" if "h" in training_info['eta'] and "d" not in training_info['eta'] else "yellow"
        table.add_row("⏱️  ETA", f"[{eta_color}]{training_info['eta']}[/{eta_color}]")

        # Status (compact one line)
        status_parts = []
        if training_info['complete'] > 0:
            status_parts.append(f"[green]✅{training_info['complete']}[/green]")
        if training_info['running'] > 0:
            status_parts.append(f"[cyan]🔄{training_info['running']}[/cyan]")
        if training_info['failed'] > 0:
            status_parts.append(f"[red]❌{training_info['failed']}[/red]")
        if status_parts:
            table.add_row("📊 Status", " ".join(status_parts))

        # Running trials (compact)
        if training_info.get('running_details'):
            table.add_row("", "")  # Spacer
            for i, detail in enumerate(training_info['running_details'][:3]):
                config_parts = []
                if detail.get('batch_size') is not None:
                    batch_idx = int(float(detail['batch_size']))
                    batch_map = {0: '2k', 1: '4k', 2: '8k'}
                    config_parts.append(f"b:{batch_map.get(batch_idx, str(batch_idx))}")
                if detail.get('net_dimension') is not None:
                    config_parts.append(f"d:{int(float(detail['net_dimension']))}")
                config_str = ' '.join(config_parts) if config_parts else "init"

                study_num = detail.get('study', '').split('_')[-1]
                label = f"W{study_num}" if study_num else f"W{i+1}"

                table.add_row(
                    f"  {label}",
                    f"[cyan]T#{detail['number']}[/cyan] | {detail['elapsed_hours']:.1f}h | {config_str}"
                )

        return Panel(
            table,
            title="[bold]🧠 Training[/bold]",
            border_style="magenta",
            padding=(0, 1)
        )

    def generate_layout(self) -> Layout:
        """Generate the dashboard layout."""
        traders = self.get_active_traders()

        # Get training info if enabled
        training_panel = None
        if self.show_training:
            training_info = self.get_training_progress()
            if training_info:
                training_panel = self.create_training_panel(training_info)

        if not traders:
            # Show training panel if available, otherwise show no traders message
            if training_panel:
                training_logs_panel = self.create_training_logs_panel(lines=10)
                layout = Layout()
                layout.split_column(
                    Layout(Panel(
                        Text("Dashboard", style="bold cyan"),
                        border_style="cyan",
                        padding=(0, 0)
                    ), size=1),
                    Layout(Panel(
                        "[yellow]No traders (waiting)[/yellow]",
                        title="Paper",
                        border_style="yellow",
                        padding=(0, 0)
                    ), size=2),
                    Layout(training_panel, size=8),
                    Layout(training_logs_panel)
                )
                return layout
            else:
                return Layout(
                    Panel(
                        "[yellow]⚠️  No active paper traders found[/yellow]",
                        title="Paper Trader Dashboard",
                        border_style="cyan"
                    )
                )

        # Create header (compact)
        header_text = Text()
        header_text.append(f"Dashboard {datetime.now().strftime('%H:%M:%S')}", style="bold cyan")

        header = Panel(header_text, border_style="cyan", padding=(0, 0))

        # Create trader panels
        trader_panels = []
        for name, csv_path in traders:
            metrics = self.calculate_metrics(csv_path)

            if not metrics:
                # No data yet - calculate time until first poll
                csv = Path(csv_path)
                file_age_seconds = time.time() - csv.stat().st_mtime
                # Poll interval is 300 seconds (5 minutes)
                time_until_poll = max(0, 300 - file_age_seconds)
                countdown_str, color = self.format_countdown(time_until_poll)

                waiting_text = Text()
                waiting_text.append("⏳ Waiting for first poll...\n\n", style="yellow")
                waiting_text.append("The paper trader is running but hasn't made its first trade poll yet.\n", style="dim")
                waiting_text.append(f"Next poll in: ", style="dim")
                waiting_text.append(f"{countdown_str}\n", style=color)

                panel = Panel(
                    waiting_text,
                    title=f"[bold]{name}[/bold]",
                    border_style="cyan"
                )
                trader_panels.append(panel)
                continue

            next_poll = self.calculate_next_poll_time(csv_path)
            table = self.create_trader_table(name, metrics, next_poll)

            # Color based on performance
            if metrics['sharpe_recent'] and metrics['sharpe_recent'] > 0:
                border_color = "green"
            elif metrics['total_return'] >= 0:
                border_color = "yellow"
            else:
                border_color = "red"

            panel = Panel(
                table,
                title=f"[bold]{name}[/bold]",
                border_style=border_color
            )
            trader_panels.append(panel)

        # Get training info if enabled
        training_panel = None
        if self.show_training:
            training_info = self.get_training_progress()
            if training_info:
                training_panel = self.create_training_panel(training_info)

        # Create layout (compact)
        layout = Layout()
        layout.split_column(
            Layout(header, size=1),
            Layout(name="main_content")
        )

        # If we have both traders and training, split the main content
        if trader_panels and training_panel:
            # Create training section with progress + logs (compact)
            training_logs_panel = self.create_training_logs_panel(lines=8)
            training_section = Layout(name="training_section")
            training_section.split_column(
                Layout(training_panel, ratio=1),
                Layout(training_logs_panel, ratio=1)
            )

            layout["main_content"].split_row(
                Layout(name="traders", ratio=1),
                Layout(training_section, ratio=2)
            )

            # Split traders section based on number of traders
            if len(trader_panels) == 1:
                layout["main_content"]["traders"].update(trader_panels[0])
            elif len(trader_panels) == 2:
                layout["main_content"]["traders"].split_column(*trader_panels)
            else:
                layout["main_content"]["traders"].split_column(
                    *[Layout(panel) for panel in trader_panels]
                )

        # Only traders, no training
        elif trader_panels:
            if len(trader_panels) == 1:
                layout["main_content"].update(trader_panels[0])
            elif len(trader_panels) == 2:
                layout["main_content"].split_row(*trader_panels)
            else:
                layout["main_content"].split_column(
                    *[Layout(panel) for panel in trader_panels]
                )

        # Only training, no traders
        elif training_panel:
            layout["main_content"].update(training_panel)

        # Nothing to show
        else:
            layout["main_content"].update(
                Panel(
                    "[yellow]⚠️  No active paper traders or training found[/yellow]",
                    border_style="cyan"
                )
            )

        return layout

    def run(self, refresh_interval: int = 1):
        """Run the live dashboard."""
        with Live(self.generate_layout(), console=self.console, refresh_per_second=1) as live:
            try:
                while True:
                    time.sleep(refresh_interval)
                    live.update(self.generate_layout())
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Paper Trader Live Dashboard with Training Progress')
    parser.add_argument('--poll-interval', type=int, default=3600,
                       help='Trader poll interval in seconds (default: 3600)')
    parser.add_argument('--refresh', type=int, default=1,
                       help='Dashboard refresh interval in seconds (default: 1)')
    parser.add_argument('--no-training', action='store_true',
                       help='Disable training progress display (default: show training)')
    parser.add_argument('--training-db', type=str, default='databases/optuna_cappuccino.db',
                       help='Path to Optuna database (default: databases/optuna_cappuccino.db)')
    parser.add_argument('--training-study', type=str, default=None,
                       help='Training study name (default: auto-detect most active)')

    args = parser.parse_args()

    dashboard = PaperTraderDashboard(
        poll_interval=args.poll_interval,
        show_training=not args.no_training,
        training_study_name=args.training_study  # None = auto-detect
    )
    dashboard.training_db_path = args.training_db
    dashboard.run(refresh_interval=args.refresh)


if __name__ == '__main__':
    main()
