#!/usr/bin/env python3
"""
Cappuccino Trading System Dashboard - Multi-Page Interactive Monitor

Pages:
  0 - Main Dashboard: System overview
  1 - Training Monitor: Latest training session progress
  2 - Paper Trading: Live ensemble performance with BEST models
  3 - Model Arena: [INACTIVE] Old individual model evaluation archive
  4 - Trade History: Recent trade activity from ensemble
  5 - Performance: Returns and metrics over time
  6 - System Health: Services, resources, alerts
  7 - Configuration: Current system settings
  8 - Tiburtina AI: Market analysis, macro data, news, AI insights
  9 - Two-Phase Training: Weekly automated training progress

Navigation:
  0-9: Jump to page
  ←/→: Previous/Next page
  r: Refresh
  q: Quit

Active Trading:
  • Ensemble: train_results/ensemble_best (Top 20 models, Sharpe 0.14-0.15)
  • Models: Trials 686, 687, 521, 578, 520... from cappuccino_alpaca_v2
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    import psutil
except ImportError:
    print("Installing required packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "psutil"], check=True)
    import pandas as pd
    import psutil


class CappuccinoDashboard:
    """Multi-page interactive dashboard for Cappuccino trading system."""

    def __init__(self):
        self.current_page = 0
        self.total_pages = 10  # Added Two-Phase Training page
        self.last_refresh = time.time()
        self.refresh_interval = 5  # seconds

        # Paths
        self.db_path = "databases/optuna_cappuccino.db"
        self.arena_state = Path("arena_state/arena_state.json")
        self.arena_leaderboard = Path("arena_state/leaderboard.txt")
        self.deployment_state = Path("deployments/deployment_state.json")

        # Two-phase training paths
        self.phase1_db = Path("databases/optuna_phase1.db")
        self.phase2_ppo_db = Path("databases/optuna_phase2_ppo.db")
        self.phase2_ddqn_db = Path("databases/optuna_phase2_ddqn.db")
        self.phase1_winner = Path("phase1_winner.json")
        self.two_phase_checkpoint = Path("two_phase_checkpoint.json")

        # Tiburtina integration
        try:
            from tiburtina_integration import get_tiburtina_client
            self.tiburtina = get_tiburtina_client()
        except:
            self.tiburtina = None

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

    def get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal width and height."""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except:
            return 120, 40

    def colorize(self, text: str, color: str) -> str:
        """Add ANSI color to text."""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'end': '\033[0m',
        }
        return f"{colors.get(color, '')}{text}{colors['end']}"

    def get_warning_indicator(self, severity: str) -> str:
        """Get visual warning indicator based on severity."""
        if severity == 'critical':
            return self.colorize("🔴", "red")
        elif severity == 'warning':
            return self.colorize("🟡", "yellow")
        else:
            return self.colorize("🟢", "green")

    def render_header(self) -> str:
        """Render page header with navigation."""
        width, _ = self.get_terminal_size()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "CAPPUCCINO TRADING SYSTEM"

        header = []
        header.append("=" * width)
        header.append(f"{self.colorize(title, 'bold'):^{width}}")
        header.append(f"{now:^{width}}")
        header.append("=" * width)

        # Page indicator
        page_names = [
            "Main", "Training", "Paper Trade", "Arena",
            "Trades", "Performance", "Health", "Config", "Tiburtina AI", "Two-Phase"
        ]
        current_name = page_names[self.current_page]
        nav_text = f"Page {self.current_page}/{self.total_pages-1}: {current_name}"
        header.append(f"{nav_text:^{width}}")

        # Navigation help
        nav_help = "0-9: Jump | ←/→: Navigate | r: Refresh | q: Quit"
        header.append(f"{self.colorize(nav_help, 'cyan'):^{width+10}}")  # +10 for color codes
        header.append("-" * width)

        return "\n".join(header)

    def render_footer(self) -> str:
        """Render page footer."""
        width, _ = self.get_terminal_size()
        elapsed = time.time() - self.last_refresh
        footer = []
        footer.append("-" * width)
        footer.append(f"Last refresh: {elapsed:.1f}s ago | Auto-refresh: {self.refresh_interval}s")
        footer.append("=" * width)
        return "\n".join(footer)

    # ============================================================================
    # PAGE 0: Main Dashboard
    # ============================================================================

    def render_page_0(self) -> str:
        """Main dashboard - system overview."""
        lines = []
        lines.append(self.colorize("SYSTEM OVERVIEW", "bold"))
        lines.append("")

        # Get service status
        services = self.get_service_status()

        lines.append(self.colorize("Services Status:", "yellow"))
        for service_name, status in services.items():
            status_color = "green" if status["running"] else "red"
            status_text = "RUNNING" if status["running"] else "STOPPED"
            pid_text = f"(PID: {status['pid']})" if status['pid'] else ""
            lines.append(f"  {service_name:30s} {self.colorize(status_text, status_color)} {pid_text}")

        lines.append("")

        # Latest training stats
        training_stats = self.get_latest_training_stats()
        if training_stats:
            lines.append(self.colorize("Latest Training:", "yellow"))
            lines.append(f"  Study: {training_stats['study_name']}")
            lines.append(f"  Total Trials: {training_stats['total_trials']}")
            lines.append(f"  Best Value: {training_stats['best_value']:.6f}")
            lines.append(f"  Best Trial: {training_stats['best_trial']}")

        lines.append("")

        # Arena summary
        arena_stats = self.get_arena_stats()
        if arena_stats:
            lines.append(self.colorize("Model Arena:", "yellow"))
            lines.append(f"  Active Models: {arena_stats['total_models']}")
            lines.append(f"  Top Performer: {arena_stats['top_model']} ({arena_stats['top_return']:+.2f}%)")
            lines.append(f"  Total Trades: {arena_stats['total_trades']}")

        lines.append("")

        # Paper trader status
        paper_stats = self.get_paper_trader_stats()
        if paper_stats:
            lines.append(self.colorize("Paper Trading:", "yellow"))
            lines.append(f"  Current Model: {paper_stats['model']}")
            lines.append(f"  Deployed: {paper_stats['deployed_at']}")
            lines.append(f"  Status: {self.colorize('ACTIVE', 'green') if paper_stats['running'] else self.colorize('STOPPED', 'red')}")

        return "\n".join(lines)

    # ============================================================================
    # PAGE 1: Training Monitor
    # ============================================================================

    def render_page_1(self) -> str:
        """Training session monitor."""
        lines = []
        lines.append(self.colorize("TRAINING SESSION MONITOR", "bold"))
        lines.append("")

        # Get latest training session
        training = self.get_latest_training_stats()
        if not training:
            lines.append("No training data available")
            return "\n".join(lines)

        lines.append(f"Study: {self.colorize(training['study_name'], 'cyan')}")
        lines.append(f"Database: {self.db_path}")
        lines.append("")

        # Overall statistics
        lines.append(self.colorize("Overall Statistics:", "yellow"))
        lines.append(f"  Total Trials: {training['total_trials']}")
        lines.append(f"  Completed: {training['completed_trials']}")
        lines.append(f"  Failed: {training['failed_trials']}")
        lines.append(f"  Success Rate: {training['success_rate']:.1f}%")
        lines.append("")

        # Best trial
        lines.append(self.colorize("Best Performance:", "yellow"))
        lines.append(f"  Trial Number: {training['best_trial']}")
        best_val = training['best_value']
        best_val_str = f'{best_val:.6f}'
        lines.append(f"  Objective Value: {self.colorize(best_val_str, 'green')}")
        lines.append("")

        # Recent trials
        recent = self.get_recent_trials(limit=10)
        if recent:
            lines.append(self.colorize("Recent Trials (Last 10):", "yellow"))
            lines.append(f"  {'Trial':<8} {'Value':<12} {'State':<12} {'Params'}")
            lines.append(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*30}")
            for trial in recent:
                value_str = f"{trial['value']:.6f}" if trial['value'] else "N/A"
                lines.append(f"  {trial['number']:<8} {value_str:<12} {trial['state']:<12} ...")

        return "\n".join(lines)

    # ============================================================================
    # PAGE 2: Paper Trading
    # ============================================================================

    def render_page_2(self) -> str:
        """Paper trading status - LIVE ENSEMBLE."""
        lines = []
        lines.append(self.colorize("LIVE PAPER TRADING - ENSEMBLE", "bold"))
        lines.append("")

        # Check for ensemble_best
        ensemble_best = Path("train_results/ensemble_best")
        ensemble_manifest = ensemble_best / "ensemble_manifest.json"

        if ensemble_manifest.exists():
            try:
                with open(ensemble_manifest) as f:
                    manifest = json.load(f)

                lines.append(self.colorize("✓ ACTIVE ENSEMBLE (BEST 20 MODELS)", "green"))
                lines.append(f"  Location: {ensemble_best}")
                lines.append(f"  Models: {manifest['model_count']}")
                lines.append(f"  Mean Sharpe: {manifest['mean_value']:.4f}")
                lines.append(f"  Best: Trial {manifest['trial_numbers'][0]} (Sharpe {manifest['best_value']:.4f})")
                lines.append(f"  Worst: Trial {manifest['trial_numbers'][-1]} (Sharpe {manifest['worst_value']:.4f})")
                lines.append(f"  Updated: {manifest['updated']}")
                lines.append("")

                # Top 5 models
                lines.append(self.colorize("Top 5 Models in Ensemble:", "cyan"))
                for i in range(min(5, len(manifest['trial_numbers']))):
                    trial = manifest['trial_numbers'][i]
                    sharpe = manifest['trial_values'][i]
                    lines.append(f"  {i+1}. Trial {trial}: Sharpe {sharpe:.4f}")

                lines.append("")

            except Exception as e:
                lines.append(f"Error reading ensemble: {e}")
        else:
            lines.append(self.colorize("⚠️  ensemble_best not found", "yellow"))

        # Check paper trader process
        import psutil
        lines.append("")
        lines.append(self.colorize("Paper Trader Status:", "cyan"))

        paper_trader_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'paper_trader' in cmdline and 'ensemble_best' in cmdline:
                    lines.append(f"  Status: {self.colorize('✓ RUNNING', 'green')}")
                    lines.append(f"  PID: {proc.info['pid']}")
                    lines.append(f"  Log: logs/paper_trading_BEST.log")
                    paper_trader_running = True
                    break
            except:
                pass

        if not paper_trader_running:
            lines.append(f"  Status: {self.colorize('✗ NOT RUNNING', 'red')}")

        # Current trading data
        lines.append("")
        lines.append("-" * 80)
        lines.append(self.colorize("CURRENT TRADING DATA", "cyan"))

        # Load positions
        positions_file = Path("paper_trades/positions_state.json")
        if positions_file.exists():
            try:
                with open(positions_file) as f:
                    positions = json.load(f)

                lines.append(f"  Portfolio Value: ${positions['portfolio_value']:,.2f}")
                lines.append(f"  Cash: ${positions['cash']:,.2f}")
                lines.append(f"  Last Update: {positions['timestamp']}")
                lines.append("")

                if positions['positions']:
                    lines.append(self.colorize("  Active Positions:", "cyan"))
                    for pos in positions['positions']:
                        pnl_color = 'green' if pos['pnl_pct'] >= 0 else 'red'
                        pnl_str = f'{pos["pnl_pct"]:+.2f}%'
                        lines.append(f"    {pos['ticker']:<12s} "
                                   f"Value: ${pos['position_value']:>8,.2f}  "
                                   f"P&L: {self.colorize(pnl_str, pnl_color)}")
                else:
                    lines.append("  No active positions (100% cash)")

                # Overall P&L
                initial = positions['portfolio_protection']['initial_value']
                current = positions['portfolio_value']
                pnl = current - initial
                pnl_pct = (pnl / initial) * 100
                pnl_color = 'green' if pnl >= 0 else 'red'
                pnl_str = f'${pnl:+,.2f} ({pnl_pct:+.2f}%)'
                lines.append("")
                lines.append(f"  Total P&L: {self.colorize(pnl_str, pnl_color)}")

            except Exception as e:
                lines.append(f"  Error loading positions: {e}")
        else:
            lines.append("  No position data available yet")

        # Trading log data
        lines.append("")
        log_file = Path("logs/paper_trading_BEST.log")
        if log_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(log_file)
                if len(df) > 0:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    lines.append(f"  Trading Started: {df.iloc[0]['timestamp']}")
                    lines.append(f"  Data Points: {len(df)} hours")
                    lines.append(f"  Latest Bar: {df.iloc[-1]['timestamp']}")

                    # Market comparison
                    if len(df) > 1:
                        start_btc = df.iloc[0]['price_BTC/USD']
                        end_btc = df.iloc[-1]['price_BTC/USD']
                        market_return = ((end_btc - start_btc) / start_btc) * 100

                        start_value = df.iloc[0]['total_asset']
                        end_value = df.iloc[-1]['total_asset']
                        agent_return = ((end_value - start_value) / start_value) * 100
                        alpha = agent_return - market_return

                        alpha_color = 'green' if alpha >= 0 else 'red'
                        alpha_str = f'{alpha:+.2f}%'
                        lines.append("")
                        lines.append(f"  Market Return (BTC): {market_return:+.2f}%")
                        lines.append(f"  Agent Return: {agent_return:+.2f}%")
                        lines.append(f"  Alpha: {self.colorize(alpha_str, alpha_color)}")
                    else:
                        lines.append("")
                        lines.append("  (Insufficient data for performance comparison)")
            except Exception as e:
                lines.append(f"  Error loading trading log: {e}")
        else:
            lines.append("  No trading log available yet")

        # Old deployment info (archived)
        lines.append("")
        lines.append("-" * 80)
        lines.append(self.colorize("ARCHIVED: Old Single-Model Deployment", "yellow"))

        paper = self.get_paper_trader_stats()
        if paper:
            lines.append(f"  Last model: {paper['model']}")
            lines.append(f"  Deployed: {paper['deployed_at']}")
            lines.append(f"  (Replaced by ensemble on 2025-12-17)")

        return "\n".join(lines)

    # ============================================================================
    # PAGE 3: Model Arena
    # ============================================================================

    def render_page_3(self) -> str:
        """Model Arena - Individual Traders."""
        lines = []

        # Check if Arena is currently active
        arena_config_path = Path("arena_state/arena_config.json")
        arena_active = arena_config_path.exists()

        # Check if arena process is running
        try:
            result = subprocess.run(
                ["pgrep", "-f", "arena_runner.py"],
                capture_output=True,
                text=True,
                timeout=1
            )
            arena_running = bool(result.stdout.strip())
        except:
            arena_running = False

        if arena_active and arena_running:
            # ACTIVE ARENA
            lines.append(self.colorize("MODEL ARENA - INDIVIDUAL TRADERS [ACTIVE]", "green"))
            lines.append("")

            # Load arena configuration
            try:
                with open(arena_config_path) as f:
                    config = json.load(f)

                lines.append(self.colorize("ARENA CONFIGURATION", "cyan"))
                lines.append(f"  Study: {config.get('study_name', 'Unknown')}")
                lines.append(f"  Models: {config.get('model_count', 0)}")
                lines.append(f"  Arena Name: {config.get('arena_name', 'Unknown')}")
                lines.append("")

                # Show model rankings with validation status
                lines.append(self.colorize("MODEL RANKINGS", "cyan"))
                lines.append(f"  {'#':<4} {'Trial':<8} {'ID':<8} {'DB Sharpe':<12} {'Live P&L':<12} {'Status':<10}")
                lines.append(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*10}")

                models = config.get('models', [])
                for i, model in enumerate(models[:15], 1):  # Show top 15
                    trial_num = model.get('trial_number', '?')
                    trial_id = model.get('trial_id', '?')
                    sharpe = model.get('sharpe_value', 0.0)

                    # Check validation
                    val_status = model.get('validation_status', {})
                    if val_status.get('all_valid', False):
                        status = self.colorize("✓ Valid", "green")
                    else:
                        status = self.colorize("⚠ Issues", "yellow")

                    # Try to load live performance from arena state
                    live_pnl = "N/A"
                    arena_manifest = Path("arena_state/arena_manifest.json")
                    if arena_manifest.exists():
                        try:
                            with open(arena_manifest) as f:
                                manifest = json.load(f)
                            performance = manifest.get('performance', {})
                            model_perf = performance.get(f"model_{i-1}", {})
                            pnl_pct = model_perf.get('pnl_pct', None)
                            if pnl_pct is not None:
                                pnl_color = 'green' if pnl_pct >= 0 else 'red'
                                live_pnl = self.colorize(f"{pnl_pct:+.2f}%", pnl_color)
                        except:
                            pass

                    lines.append(f"  {i:<4} {trial_num:<8} {trial_id:<8} {sharpe:<12.4f} {live_pnl:<20} {status}")

                if len(models) > 15:
                    lines.append(f"  ... and {len(models) - 15} more models")

                lines.append("")

                # Show current positions if available
                deployments_dir = Path("deployments")
                if deployments_dir.exists():
                    lines.append(self.colorize("MODEL PORTFOLIOS", "cyan"))
                    lines.append(f"  {'Model':<15} {'Value':<12} {'Cash':<12} {'Positions':<40}")
                    lines.append(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*40}")

                    for i in range(min(10, len(models))):  # Show top 10 portfolios
                        model_dir = deployments_dir / f"model_{i}"
                        positions_file = model_dir / "positions.json"

                        if positions_file.exists():
                            try:
                                with open(positions_file) as f:
                                    positions = json.load(f)

                                portfolio_value = positions.get('portfolio_value', 0)
                                cash = positions.get('cash', 0)
                                assets = positions.get('positions', [])

                                # Format positions
                                pos_str = ""
                                if assets:
                                    pos_parts = []
                                    for asset in assets[:3]:  # Top 3 positions
                                        ticker = asset.get('ticker', '?')
                                        value = asset.get('position_value', 0)
                                        pos_parts.append(f"{ticker}=${value:.0f}")
                                    pos_str = ", ".join(pos_parts)
                                    if len(assets) > 3:
                                        pos_str += f" +{len(assets)-3} more"
                                else:
                                    pos_str = "All cash"

                                lines.append(f"  model_{i:<10} ${portfolio_value:<11,.0f} ${cash:<11,.0f} {pos_str:<40}")
                            except:
                                lines.append(f"  model_{i:<10} {'N/A':<12} {'N/A':<12} {'N/A':<40}")
                        else:
                            lines.append(f"  model_{i:<10} {'N/A':<12} {'N/A':<12} {'No data':<40}")

                    lines.append("")

                # Show recent activity
                arena_log = Path("logs/arena.log")
                if arena_log.exists():
                    lines.append(self.colorize("RECENT ACTIVITY (Last 10 lines)", "cyan"))
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["tail", "-10", str(arena_log)],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        for line in result.stdout.strip().split('\n'):
                            lines.append(f"  {line}")
                    except:
                        lines.append("  Unable to read arena log")
                    lines.append("")

                # Control instructions
                lines.append(self.colorize("ARENA CONTROLS", "yellow"))
                lines.append("  • Stop Arena: ./stop_arena.sh")
                lines.append("  • View logs: tail -f logs/arena.log")
                lines.append("  • Status: ./status_arena.sh")

            except Exception as e:
                lines.append(self.colorize(f"ERROR loading Arena config: {str(e)}", "red"))

        elif arena_active and not arena_running:
            # CONFIG EXISTS BUT NOT RUNNING
            lines.append(self.colorize("MODEL ARENA - INDIVIDUAL TRADERS [CONFIGURED, NOT RUNNING]", "yellow"))
            lines.append("")
            lines.append(self.colorize("⚠️  Arena is configured but not currently running", "yellow"))
            lines.append("")

            try:
                with open(arena_config_path) as f:
                    config = json.load(f)

                lines.append(self.colorize("CONFIGURED ARENA:", "cyan"))
                lines.append(f"  Study: {config.get('study_name', 'Unknown')}")
                lines.append(f"  Models: {config.get('model_count', 0)}")
                lines.append("")

                lines.append(self.colorize("TO START ARENA:", "green"))
                lines.append("  python arena_runner.py --config arena_state/arena_config.json")
                lines.append("  OR: ./start_arena.sh")

            except Exception as e:
                lines.append(self.colorize(f"ERROR loading config: {str(e)}", "red"))

        else:
            # ARENA NOT CONFIGURED - SHOW SETUP INSTRUCTIONS
            lines.append(self.colorize("MODEL ARENA - INDIVIDUAL TRADERS [NOT CONFIGURED]", "yellow"))
            lines.append("")
            lines.append(self.colorize("Arena is not currently set up.", "yellow"))
            lines.append("")

            lines.append(self.colorize("CURRENT ACTIVE TRADING:", "green"))
            lines.append("  → Page 2: Ensemble of top 20 models (Sharpe 0.14-0.15)")
            lines.append("  → Models: 686, 687, 521, 578, 520... from study 'cappuccino_alpaca_v2'")
            lines.append("  → Location: train_results/ensemble_best/")
            lines.append("")

            lines.append(self.colorize("TO SET UP ARENA:", "cyan"))
            lines.append("  1. Validate models:")
            lines.append("     python validate_models.py --study cappuccino_alpaca_v2 --top-n 10")
            lines.append("")
            lines.append("  2. Setup and deploy Arena:")
            lines.append("     python setup_arena_clean.py --top-n 10")
            lines.append("")
            lines.append("  This will:")
            lines.append("    • Find best study")
            lines.append("    • Validate top 10 models")
            lines.append("    • Deploy to deployments/")
            lines.append("    • Create arena_config.json")
            lines.append("    • Start Arena")
            lines.append("")

            lines.append("-" * 80)
            lines.append(self.colorize("ARCHIVED ARENA DATA (Dec 12-16)", "yellow"))
            lines.append("")

            # Show archived data
            if self.arena_leaderboard.exists():
                leaderboard = self.arena_leaderboard.read_text()
                lines.append(leaderboard)
            else:
                arena_data = self.get_arena_stats()
                if arena_data and arena_data['total_models'] > 0:
                    lines.append(f"Total models (archived): {arena_data['total_models']}")
                else:
                    lines.append("No archived arena data")

        return "\n".join(lines)

    # ============================================================================
    # PAGE 4: Trade History
    # ============================================================================

    def render_page_4(self) -> str:
        """Recent trade activity."""
        lines = []
        lines.append(self.colorize("TRADE HISTORY", "bold"))
        lines.append("")

        # Get arena trade history
        arena_data = self.load_arena_state()
        if arena_data and 'portfolios' in arena_data:
            lines.append(self.colorize("Arena Trades (Recent):", "yellow"))

            all_trades = []
            for model_id, portfolio in arena_data['portfolios'].items():
                if 'trade_history' in portfolio:
                    for trade in portfolio['trade_history'][-5:]:
                        trade['model'] = model_id
                        all_trades.append(trade)

            # Sort by timestamp
            all_trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            if all_trades:
                lines.append(f"  {'Model':<12} {'Side':<6} {'Ticker':<10} {'Qty':<10} {'Price':<10} {'Time'}")
                lines.append(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*19}")

                for trade in all_trades[:20]:
                    model = trade.get('model', 'N/A')
                    side = trade.get('side', 'N/A')
                    ticker = trade.get('ticker', 'N/A')
                    qty = trade.get('qty', 0)
                    price = trade.get('price', 0)
                    timestamp = trade.get('timestamp', '')[:19]

                    side_color = 'green' if side == 'buy' else 'red'
                    lines.append(f"  {model:<12} {self.colorize(side.upper(), side_color):<6} {ticker:<10} {qty:<10.4f} {price:<10.2f} {timestamp}")
            else:
                lines.append("  No trades recorded yet")
        else:
            lines.append("No trade data available")

        return "\n".join(lines)

    # ============================================================================
    # PAGE 5: Performance
    # ============================================================================

    def render_page_5(self) -> str:
        """Performance metrics over time."""
        lines = []
        lines.append(self.colorize("PERFORMANCE METRICS", "bold"))
        lines.append("")

        arena_data = self.load_arena_state()
        if not arena_data or 'portfolios' not in arena_data:
            lines.append("No performance data available")
            return "\n".join(lines)

        lines.append(self.colorize("Arena Models Performance:", "yellow"))
        lines.append(f"  {'Model':<12} {'Return %':<10} {'Sharpe':<8} {'MaxDD %':<10} {'Win Rate':<10} {'Trades'}")
        lines.append(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*7}")

        for model_id, portfolio in arena_data['portfolios'].items():
            # Calculate metrics
            cash = portfolio.get('cash', 1000)
            initial = portfolio.get('initial_value', 1000)

            # Calculate total portfolio value (cash + holdings)
            # Use latest value from history if available, otherwise estimate from cash
            value_history = portfolio.get('value_history', [])
            if len(value_history) > 0:
                # Use the most recent portfolio value
                current_value = value_history[-1][1]
            else:
                # Fallback: just cash (shouldn't happen normally)
                current_value = cash

            return_pct = ((current_value - initial) / initial) * 100

            trades = portfolio.get('total_trades', 0)
            winning = portfolio.get('winning_trades', 0)
            win_rate = (winning / trades * 100) if trades > 0 else 0

            # Calculate max drawdown from value history
            if len(value_history) > 1:
                values = [v for _, v in value_history]
                peak = max(values)
                max_dd = 0
                for v in values:
                    if v < peak:
                        dd = (peak - v) / peak * 100
                        max_dd = max(max_dd, dd)
            else:
                max_dd = 0

            # Calculate Sharpe ratio from value history
            if len(value_history) >= 10:
                values = [v for _, v in value_history]
                returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
                if len(returns) > 0 and np.std(returns) > 0:
                    # Annualized Sharpe (assuming hourly data)
                    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(24 * 365)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            return_color = 'green' if return_pct > 0 else 'red' if return_pct < 0 else 'white'
            return_str = f'{return_pct:>+8.2f}'

            lines.append(f"  {model_id:<12} {self.colorize(return_str, return_color):<10} "
                        f"{sharpe:>6.2f}  {max_dd:>8.2f}  {win_rate:>8.1f}%  {trades:>7}")

        return "\n".join(lines)

    # ============================================================================
    # PAGE 6: System Health
    # ============================================================================

    def render_page_6(self) -> str:
        """System health and resources."""
        lines = []
        lines.append(self.colorize("SYSTEM HEALTH", "bold"))
        lines.append("")

        # Services
        services = self.get_service_status()
        lines.append(self.colorize("Services:", "yellow"))
        for name, status in services.items():
            status_icon = self.colorize("●", "green") if status['running'] else self.colorize("○", "red")
            lines.append(f"  {status_icon} {name}")

        lines.append("")

        # Resources
        lines.append(self.colorize("System Resources:", "yellow"))

        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_color = 'green' if cpu_percent < 50 else 'yellow' if cpu_percent < 80 else 'red'
        cpu_str = f'{cpu_percent:.1f}%'
        lines.append(f"  CPU: {self.colorize(cpu_str, cpu_color)}")

        # Memory
        mem = psutil.virtual_memory()
        mem_color = 'green' if mem.percent < 70 else 'yellow' if mem.percent < 85 else 'red'
        mem_str = f'{mem.percent:.1f}%'
        lines.append(f"  Memory: {self.colorize(mem_str, mem_color)} ({mem.used // (1024**3)}GB / {mem.total // (1024**3)}GB)")

        # Disk
        disk = psutil.disk_usage('/')
        disk_color = 'green' if disk.percent < 70 else 'yellow' if disk.percent < 85 else 'red'
        disk_str = f'{disk.percent:.1f}%'
        lines.append(f"  Disk: {self.colorize(disk_str, disk_color)} ({disk.used // (1024**3)}GB / {disk.total // (1024**3)}GB)")

        # GPU (if nvidia-smi available)
        try:
            gpu_output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], text=True)
            lines.append("")
            lines.append(self.colorize("GPU:", "yellow"))
            for i, line in enumerate(gpu_output.strip().split('\n')):
                parts = line.split(', ')
                if len(parts) >= 4:
                    util, mem_used, mem_total, temp = parts
                    lines.append(f"  GPU {i}: {util}% util, {mem_used}MB / {mem_total}MB, {temp}°C")
        except:
            pass

        return "\n".join(lines)

    # ============================================================================
    # PAGE 7: Configuration
    # ============================================================================

    def render_page_7(self) -> str:
        """System configuration."""
        lines = []
        lines.append(self.colorize("SYSTEM CONFIGURATION", "bold"))
        lines.append("")

        lines.append(self.colorize("Database:", "yellow"))
        lines.append(f"  Path: {self.db_path}")
        lines.append(f"  Exists: {os.path.exists(self.db_path)}")

        lines.append("")
        lines.append(self.colorize("Arena Settings:", "yellow"))
        lines.append(f"  State File: {self.arena_state}")
        lines.append(f"  Leaderboard: {self.arena_leaderboard}")
        lines.append(f"  Max Models: 10")
        lines.append(f"  Min Evaluation: 168 hours (7 days)")
        lines.append(f"  Promotion Threshold: 2% return")

        lines.append("")
        lines.append(self.colorize("Deployment:", "yellow"))
        lines.append(f"  State File: {self.deployment_state}")
        lines.append(f"  Mode: Hybrid (Training + Arena)")

        lines.append("")
        lines.append(self.colorize("Hybrid Trading System:", "yellow"))
        lines.append("  • Best training model → Immediate deployment to paper trading")
        lines.append("  • Top 9 models → Arena for evaluation")
        lines.append("  • Arena promotion after 7 days if performance proven")

        return "\n".join(lines)

    # ============================================================================
    # PAGE 8: Tiburtina AI Analysis
    # ============================================================================

    def render_page_8(self) -> str:
        """Tiburtina AI-powered market analysis."""
        lines = []
        lines.append(self.colorize("TIBURTINA AI ANALYSIS", "bold"))
        lines.append("")

        # Check if Tiburtina is available
        if not self.tiburtina or not self.tiburtina.is_available():
            error = self.tiburtina.get_error() if self.tiburtina else "Tiburtina client not initialized"
            lines.append(self.colorize(f"Tiburtina not available: {error}", "red"))
            lines.append("")
            lines.append("To enable Tiburtina integration:")
            lines.append("  1. Ensure Tiburtina is installed at /home/mrc/experiment/tiburtina")
            lines.append("  2. Configure API keys in Tiburtina's .env file")
            lines.append("  3. Install dependencies: cd /home/mrc/experiment/tiburtina && pip install -r requirements.txt")
            lines.append("  4. Restart the dashboard")
            lines.append("")
            lines.append("For now, you can use the Tiburtina terminal directly:")
            lines.append("  cd /home/mrc/experiment/tiburtina && python terminal/cli.py")
            return "\n".join(lines)

        lines.append(self.colorize("Note: Data is cached for fast loading", "yellow"))
        lines.append("First view may be slow, subsequent views are instant.")
        lines.append("Press 'r' to refresh. Macro data fetches in background after first load.")
        lines.append("")

        # Macro Economic Snapshot
        lines.append(self.colorize("Macro Economic Snapshot:", "yellow"))

        # Check cache first - if no cached data, skip to avoid slow API
        try:
            cache_status = self.tiburtina.get_cache_status()
            if not cache_status['macro']['has_data']:
                lines.append(f"  {self.colorize('Loading...', 'yellow')} (will be cached for next view)")
                lines.append("  Macro data can take 30-60s on first load")
            else:
                # Have cached data - display it
                macro = self.tiburtina.get_macro_snapshot(use_cache=True)
                if "error" in macro:
                    lines.append(f"  {self.colorize('Error fetching macro data', 'red')}")
                    lines.append(f"  {str(macro['error'])[:80]}")
                elif not macro:
                    lines.append("  No macro data available")
                else:
                    for key, data in macro.items():
                        if isinstance(data, dict):
                            value = data.get("value", "N/A")
                            date = data.get("date", "")[:10] if isinstance(data.get("date", ""), str) else ""
                            display_name = key.replace("_", " ").title()
                            if isinstance(value, (int, float)):
                                lines.append(f"  {display_name:<20} {value:>8.2f}  ({date})")
                            else:
                                lines.append(f"  {display_name:<20} {value}")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            lines.append(f"  {self.colorize('Error:', 'red')} {str(e)[:60]}")

        lines.append("")

        # Crypto Market Overview
        lines.append(self.colorize("Top Crypto Markets:", "yellow"))
        try:
            # Always use cache for crypto (fast API, should work)
            cryptos = self.tiburtina.get_crypto_overview(use_cache=True)
            if cryptos and isinstance(cryptos, list) and len(cryptos) > 0:
                if "error" in cryptos[0]:
                    lines.append(f"  {self.colorize('Error fetching crypto data', 'red')}")
                    lines.append(f"  {str(cryptos[0].get('error', ''))[:80]}")
                else:
                    lines.append(f"  {'Symbol':<8} {'Price':<15} {'24h Change':<12} {'Market Cap'}")
                    lines.append(f"  {'-'*8} {'-'*15} {'-'*12} {'-'*15}")
                    for crypto in cryptos[:5]:
                        symbol = crypto.get("symbol", "N/A")
                        price = crypto.get("price", 0)
                        change = crypto.get("change_24h", 0)
                        mcap = crypto.get("market_cap", 0)

                        price_str = f"${price:,.2f}" if price and price > 1 else f"${price:.6f}" if price else "N/A"
                        change_str = f"{change:+.1f}%" if change else "N/A"
                        mcap_str = f"${mcap/1e9:.1f}B" if mcap else "N/A"

                        lines.append(f"  {symbol:<8} {price_str:<15} {change_str:<12} {mcap_str}")
            else:
                lines.append("  No crypto data available")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            lines.append(f"  {self.colorize('Error:', 'red')} {str(e)[:60]}")

        lines.append("")

        # Latest News Headlines
        lines.append(self.colorize("Latest Financial News:", "yellow"))
        try:
            # Always use cache for news
            news = self.tiburtina.get_news_summary(use_cache=True)
            if news and isinstance(news, list) and len(news) > 0:
                if "error" in news[0]:
                    lines.append(f"  {self.colorize('Error fetching news', 'red')}")
                    lines.append(f"  {str(news[0].get('error', ''))[:80]}")
                else:
                    for i, article in enumerate(news[:5], 1):
                        if isinstance(article, dict):
                            title = article.get("title", "No title")
                            source = article.get("source", "")
                            lines.append(f"  {i}. {title[:70]}...")
                            if source:
                                lines.append(f"     Source: {source}")
            else:
                lines.append("  No news available")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            lines.append(f"  {self.colorize('Error:', 'red')} {str(e)[:60]}")

        lines.append("")

        # Asset Performance Comparison
        lines.append(self.colorize("Asset Class Performance:", "yellow"))
        try:
            performance = self.tiburtina.get_asset_performance()
            if "error" in performance:
                lines.append(f"  {self.colorize('Error:', 'red')} {performance['error'][:60]}")
            else:
                if "crypto_24h" in performance:
                    change = performance["crypto_24h"]
                    color = "green" if change > 0 else "red"
                    change_str = f'{change:+.2f}%'
                    lines.append(f"  Crypto (Top 5 avg):  {self.colorize(change_str, color)} (24h)")

                if "stocks_daily" in performance:
                    change = performance["stocks_daily"]
                    color = "green" if change > 0 else "red"
                    change_str = f'{change:+.2f}%'
                    lines.append(f"  Stocks (S&P 500):    {self.colorize(change_str, color)} (today)")

                if not ("crypto_24h" in performance or "stocks_daily" in performance):
                    lines.append("  Limited performance data available")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            lines.append(f"  {self.colorize('Error:', 'red')} {str(e)[:60]}")

        lines.append("")

        # AI Market Analysis (if cache allows)
        try:
            cache_status = self.tiburtina.get_cache_status()
            # Only show AI analysis if we have some cached data
            if cache_status['crypto']['has_data'] or cache_status['news']['has_data']:
                lines.append(self.colorize("AI Market Analysis:", "yellow"))
                lines.append("  Generating comprehensive analysis...")
                lines.append("  (This uses local LLM, may take 10-30 seconds)")
                lines.append("")
                lines.append("  For instant AI analysis, use Tiburtina terminal:")
                lines.append("  $ cd /home/mrc/experiment/tiburtina && python terminal/cli.py")
                lines.append("  $ /brief  # Get AI market brief")
        except:
            pass

        lines.append("")

        # Cache status
        try:
            cache_status = self.tiburtina.get_cache_status()
            lines.append(self.colorize("Cache Status:", "yellow"))
            for source, info in cache_status.items():
                if info['has_data']:
                    age = info.get('age_seconds', 0)
                    if age:
                        age_str = f"{int(age)}s ago"
                        if age > 60:
                            age_str = f"{int(age/60)}m ago"
                    else:
                        age_str = "just now"
                    status = self.colorize("FRESH", "green") if not info['is_stale'] else self.colorize("stale", "yellow")
                    lines.append(f"  {source.capitalize():<10} {status} ({age_str})")
                else:
                    lines.append(f"  {source.capitalize():<10} {self.colorize('no data', 'red')}")
        except:
            pass

        lines.append("")
        lines.append(self.colorize("About Tiburtina:", "yellow"))
        lines.append("  Tiburtina is an AI-powered financial research terminal")
        lines.append("  Data is cached (Macro: 30m, Crypto: 5m, News: 10m)")
        lines.append("")
        lines.append("  For detailed historical analysis, use Tiburtina terminal:")
        lines.append("  $ cd /home/mrc/experiment/tiburtina")
        lines.append("  $ python terminal/cli.py")
        lines.append("")
        lines.append("  Terminal features:")
        lines.append("    /quote AAPL      - Get stock quote")
        lines.append("    /fund MSFT       - Company fundamentals")
        lines.append("    /compare BTC ETH - AI comparison analysis")
        lines.append("    /brief           - Comprehensive AI market brief")
        lines.append("    /macro           - Full macro snapshot")

        return "\n".join(lines)

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def get_service_status(self) -> Dict:
        """Get status of all services."""
        services = {
            "Training Workers": {"running": False, "pid": None},
            "Arena Runner": {"running": False, "pid": None},
            "Auto Deployer": {"running": False, "pid": None},
            "Paper Trader": {"running": False, "pid": None},
        }

        try:
            ps_output = subprocess.check_output(['ps', 'aux'], text=True)

            for line in ps_output.split('\n'):
                if '1_optimize_unified.py' in line and 'grep' not in line:
                    services["Training Workers"]["running"] = True
                    services["Training Workers"]["pid"] = line.split()[1]

                elif 'arena_runner.py' in line and 'grep' not in line:
                    services["Arena Runner"]["running"] = True
                    services["Arena Runner"]["pid"] = line.split()[1]

                elif 'auto_model_deployer.py' in line and 'grep' not in line:
                    services["Auto Deployer"]["running"] = True
                    services["Auto Deployer"]["pid"] = line.split()[1]

                elif 'paper_trader_alpaca_polling.py' in line and 'grep' not in line:
                    services["Paper Trader"]["running"] = True
                    services["Paper Trader"]["pid"] = line.split()[1]
        except:
            pass

        return services

    def get_latest_training_stats(self) -> Optional[Dict]:
        """Get statistics from latest training session (most active study)."""
        if not os.path.exists(self.db_path):
            return None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # First, try to find the study that matches running training workers
            study_name = None
            try:
                ps_output = subprocess.check_output(['ps', 'aux'], text=True)
                for line in ps_output.split('\n'):
                    if '1_optimize_unified.py' in line and '--study-name' in line:
                        # Extract study name from command line
                        parts = line.split('--study-name')
                        if len(parts) > 1:
                            study_name = parts[1].split()[0]
                            break
            except:
                pass

            # If no running study found, use the study with most trials
            if not study_name:
                cursor.execute("""
                    SELECT s.study_name, COUNT(t.trial_id) as trial_count
                    FROM studies s
                    LEFT JOIN trials t ON s.study_id = t.study_id
                    GROUP BY s.study_id
                    ORDER BY trial_count DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                study_name = row[0]

            # Get the study_id for this study
            cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
            study_id = row[0]

            # Get trial stats for this specific study
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN t.state = 'COMPLETE' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN t.state = 'FAIL' THEN 1 ELSE 0 END) as failed,
                    MAX(CASE WHEN t.state = 'COMPLETE' THEN tv.value ELSE NULL END) as best_value,
                    (SELECT t2.number FROM trials t2
                     LEFT JOIN trial_values tv2 ON t2.trial_id = tv2.trial_id
                     WHERE t2.study_id = ? AND t2.state = 'COMPLETE'
                     ORDER BY tv2.value DESC LIMIT 1) as best_trial
                FROM trials t
                LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE t.study_id = ?
            """, (study_id, study_id))

            row = cursor.fetchone()
            total, completed, failed, best_value, best_trial = row

            conn.close()

            success_rate = (completed / total * 100) if total > 0 else 0

            return {
                'study_name': study_name,
                'total_trials': total or 0,
                'completed_trials': completed or 0,
                'failed_trials': failed or 0,
                'success_rate': success_rate,
                'best_value': best_value or 0,
                'best_trial': best_trial or 0,
            }
        except Exception as e:
            return None

    def get_recent_trials(self, limit: int = 10) -> List[Dict]:
        """Get recent trials from the active study."""
        if not os.path.exists(self.db_path):
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get the active study (same logic as get_latest_training_stats)
            study_name = None
            try:
                ps_output = subprocess.check_output(['ps', 'aux'], text=True)
                for line in ps_output.split('\n'):
                    if '1_optimize_unified.py' in line and '--study-name' in line:
                        parts = line.split('--study-name')
                        if len(parts) > 1:
                            study_name = parts[1].split()[0]
                            break
            except:
                pass

            if not study_name:
                cursor.execute("""
                    SELECT s.study_name
                    FROM studies s
                    LEFT JOIN trials t ON s.study_id = t.study_id
                    GROUP BY s.study_id
                    ORDER BY COUNT(t.trial_id) DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                if row:
                    study_name = row[0]

            if not study_name:
                conn.close()
                return []

            # Get study_id
            cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return []
            study_id = row[0]

            # Get recent trials from this study
            cursor.execute(f"""
                SELECT t.number, tv.value, t.state
                FROM trials t
                LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
                WHERE t.study_id = ?
                ORDER BY t.trial_id DESC
                LIMIT {limit}
            """, (study_id,))

            trials = []
            for row in cursor.fetchall():
                trials.append({
                    'number': row[0],
                    'value': row[1],
                    'state': row[2],
                })

            conn.close()
            return trials
        except:
            return []

    def get_paper_trader_stats(self) -> Optional[Dict]:
        """Get paper trader status."""
        if not self.deployment_state.exists():
            return None

        try:
            data = json.loads(self.deployment_state.read_text())

            # Check if process is running
            services = self.get_service_status()
            is_running = services["Paper Trader"]["running"]
            pid = services["Paper Trader"]["pid"]

            return {
                'model': f"trial_{data.get('last_deployed_trial', 'N/A')}",
                'training_value': data.get('last_deployed_value', 0),
                'deployed_at': data.get('last_deployment_time', '')[:19],
                'running': is_running,
                'pid': pid,
                'history': data.get('deployment_history', []),
            }
        except:
            return None

    def load_arena_state(self) -> Optional[Dict]:
        """Load arena state."""
        if not self.arena_state.exists():
            return None

        try:
            return json.loads(self.arena_state.read_text())
        except:
            return None

    def get_arena_stats(self) -> Optional[Dict]:
        """Get arena statistics."""
        data = self.load_arena_state()
        if not data or 'portfolios' not in data:
            return None

        portfolios = data['portfolios']
        if not portfolios:
            return {'total_models': 0, 'top_model': 'N/A', 'top_return': 0, 'total_trades': 0}

        # Find top performer
        top_model = None
        top_return = -999999
        total_trades = 0

        for model_id, portfolio in portfolios.items():
            cash = portfolio.get('cash', 1000)
            initial = portfolio.get('initial_value', 1000)
            return_pct = ((cash - initial) / initial) * 100

            if return_pct > top_return:
                top_return = return_pct
                top_model = model_id

            total_trades += portfolio.get('total_trades', 0)

        return {
            'total_models': len(portfolios),
            'top_model': top_model or 'N/A',
            'top_return': top_return,
            'total_trades': total_trades,
        }

    # ============================================================================
    # PAGE 9: Two-Phase Training
    # ============================================================================

    def render_page_9(self) -> str:
        """Two-Phase training progress monitor."""
        lines = []
        lines.append(self.colorize("TWO-PHASE TRAINING PROGRESS", "bold"))
        lines.append("")

        # Check checkpoint status
        checkpoint_data = None
        if self.two_phase_checkpoint.exists():
            try:
                with open(self.two_phase_checkpoint) as f:
                    checkpoint_data = json.load(f)
            except:
                pass

        # Phase 1 Status
        lines.append(self.colorize("PHASE 1: TIME-FRAME OPTIMIZATION", "cyan"))
        lines.append("-" * 60)

        phase1_complete = checkpoint_data and checkpoint_data.get('phase1_complete', False)

        if phase1_complete:
            p1_results = checkpoint_data.get('phase1_results', {})
            lines.append(f"Status:      {self.colorize('✓ COMPLETE', 'green')}")
            lines.append(f"Winner:      {p1_results.get('timeframe', 'N/A')} @ {p1_results.get('interval', 'N/A')}")
            lines.append(f"Best Sharpe: {p1_results.get('best_sharpe_bot', 0):.4f}")
            lines.append(f"Trials:      {p1_results.get('n_trials', 0)}")
            duration_hrs = p1_results.get('phase1_duration', 0) / 3600
            lines.append(f"Duration:    {duration_hrs:.1f} hours")
        elif self.phase1_db.exists():
            # Phase 1 in progress
            try:
                conn = sqlite3.connect(str(self.phase1_db))
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT study_name, COUNT(*) as total,
                           SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) as complete,
                           SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) as running
                    FROM trials t
                    JOIN studies s ON t.study_id = s.study_id
                    GROUP BY study_name
                """)
                studies = cursor.fetchall()
                conn.close()

                lines.append(f"Status:      {self.colorize('⏳ IN PROGRESS', 'yellow')}")
                lines.append("")
                for study_name, total, complete, running in studies:
                    lines.append(f"  {study_name}:")
                    lines.append(f"    Complete: {complete}/{total}")
                    if running > 0:
                        lines.append(f"    Running:  {running}")
            except:
                lines.append(f"Status:      {self.colorize('⏳ STARTING', 'yellow')}")
        else:
            lines.append(f"Status:      {self.colorize('○ NOT STARTED', 'white')}")

        lines.append("")

        # Phase 2 Status
        lines.append(self.colorize("PHASE 2: FEATURE-ENHANCED TRAINING", "cyan"))
        lines.append("-" * 60)

        phase2_complete = checkpoint_data and checkpoint_data.get('phase2_complete', False)

        if phase2_complete:
            p2_results = checkpoint_data.get('phase2_results', {})
            lines.append(f"Status:      {self.colorize('✓ COMPLETE', 'green')}")
            lines.append(f"Winner:      {p2_results.get('winner_algorithm', 'N/A')}")
            lines.append(f"Best Sharpe: {p2_results.get('best_sharpe', 0):.4f}")
        else:
            # Check PPO
            if self.phase2_ppo_db.exists():
                try:
                    conn = sqlite3.connect(str(self.phase2_ppo_db))
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) as total,
                               SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) as complete,
                               SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) as running
                        FROM trials
                    """)
                    ppo_total, ppo_complete, ppo_running = cursor.fetchone()
                    conn.close()

                    lines.append(f"Status:      {self.colorize('⏳ IN PROGRESS', 'yellow')}")
                    lines.append("")
                    lines.append(f"PPO Trials:")
                    lines.append(f"  Complete: {ppo_complete}/200")
                    lines.append(f"  Running:  {ppo_running}")
                    lines.append(f"  Progress: {ppo_complete/200*100:.1f}%")

                    # Estimated time remaining
                    if ppo_complete > 0:
                        # Rough estimate: 30 min per trial
                        remaining = (200 - ppo_complete) * 30 / 60
                        lines.append(f"  Est. remaining: {remaining:.1f} hours")
                except:
                    lines.append(f"Status:      {self.colorize('⏳ STARTING', 'yellow')}")
            else:
                if phase1_complete:
                    lines.append(f"Status:      {self.colorize('○ WAITING FOR PHASE 1', 'white')}")
                else:
                    lines.append(f"Status:      {self.colorize('○ NOT STARTED', 'white')}")

            # Check DDQN
            if self.phase2_ddqn_db.exists():
                try:
                    conn = sqlite3.connect(str(self.phase2_ddqn_db))
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT COUNT(*) as total,
                               SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) as complete
                        FROM trials
                    """)
                    ddqn_total, ddqn_complete = cursor.fetchone()
                    conn.close()

                    lines.append("")
                    lines.append(f"DDQN Trials:")
                    lines.append(f"  Complete: {ddqn_complete}/200")
                    lines.append(f"  Progress: {ddqn_complete/200*100:.1f}%")
                except:
                    pass

        lines.append("")
        lines.append("-" * 60)

        # Overall progress
        if checkpoint_data:
            timestamp = checkpoint_data.get('timestamp', 'N/A')
            lines.append(f"Last updated: {timestamp}")

        return "\n".join(lines)

    # ============================================================================
    # Main Render Loop
    # ============================================================================

    def render(self) -> str:
        """Render current page."""
        output = []
        output.append(self.render_header())
        output.append("")

        # Render current page
        if self.current_page == 0:
            output.append(self.render_page_0())
        elif self.current_page == 1:
            output.append(self.render_page_1())
        elif self.current_page == 2:
            output.append(self.render_page_2())
        elif self.current_page == 3:
            output.append(self.render_page_3())
        elif self.current_page == 4:
            output.append(self.render_page_4())
        elif self.current_page == 5:
            output.append(self.render_page_5())
        elif self.current_page == 6:
            output.append(self.render_page_6())
        elif self.current_page == 7:
            output.append(self.render_page_7())
        elif self.current_page == 8:
            output.append(self.render_page_8())
        elif self.current_page == 9:
            output.append(self.render_page_9())

        output.append("")
        output.append(self.render_footer())

        return "\n".join(output)

    def next_page(self):
        """Go to next page."""
        self.current_page = (self.current_page + 1) % self.total_pages

    def prev_page(self):
        """Go to previous page."""
        self.current_page = (self.current_page - 1) % self.total_pages

    def jump_to_page(self, page_num: int):
        """Jump to specific page."""
        if 0 <= page_num < self.total_pages:
            self.current_page = page_num

    def run(self, once: bool = False):
        """Run dashboard loop."""
        import select
        import sys
        import tty
        import termios

        if once:
            self.clear_screen()
            print(self.render())
            return

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while True:
                self.clear_screen()
                print(self.render())
                self.last_refresh = time.time()

                # Check for input with timeout
                timeout = self.refresh_interval
                while timeout > 0:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.5)

                    if rlist:
                        ch = sys.stdin.read(1)

                        # Handle input
                        if ch == 'q':
                            return
                        elif ch == 'r':
                            break  # Force refresh
                        elif ch in '01234567':
                            self.jump_to_page(int(ch))
                            break
                        elif ch == '\x1b':  # Escape sequence
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                if ch3 == 'C':  # Right arrow
                                    self.next_page()
                                    break
                                elif ch3 == 'D':  # Left arrow
                                    self.prev_page()
                                    break

                    timeout -= 0.5

        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    parser = argparse.ArgumentParser(description="Cappuccino Trading Dashboard")
    parser.add_argument("--once", action="store_true", help="Render once and exit")
    args = parser.parse_args()

    dashboard = CappuccinoDashboard()
    dashboard.run(once=args.once)


if __name__ == "__main__":
    main()
