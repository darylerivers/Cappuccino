#!/usr/bin/env python3
"""
Enhanced Training Dashboard for Cappuccino
Shows detailed training metrics, convergence, and hyperparameter analysis
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    import psutil


class TrainingDashboard:
    """Detailed training metrics dashboard."""

    def __init__(self, db_path: str = "databases/optuna_cappuccino.db"):
        self.db_path = db_path

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')

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

    def get_study_overview(self) -> Dict:
        """Get overall study statistics."""
        conn = sqlite3.connect(self.db_path)

        # Get all studies
        studies_df = pd.read_sql_query("SELECT * FROM studies", conn)

        # Get trial counts by study
        query = """
        SELECT
            s.study_name,
            s.study_id,
            COUNT(CASE WHEN t.state = 'COMPLETE' THEN 1 END) as completed,
            COUNT(CASE WHEN t.state = 'RUNNING' THEN 1 END) as running,
            COUNT(CASE WHEN t.state = 'FAIL' THEN 1 END) as failed,
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

    def get_recent_trials(self, limit: int = 10) -> List[Dict]:
        """Get most recent completed trials."""
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            t.trial_id,
            t.number,
            t.datetime_start,
            t.datetime_complete,
            tv.value,
            t.state,
            CAST((julianday(t.datetime_complete) - julianday(t.datetime_start)) * 24 AS INT) as duration_hours
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.datetime_complete DESC
        LIMIT ?
        """

        recent = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        return recent.to_dict('records')

    def get_top_trials(self, limit: int = 10) -> List[Dict]:
        """Get top performing trials."""
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

        # Clean up JSON strings
        for trial in top_trials.to_dict('records'):
            if trial.get('timeframe'):
                trial['timeframe'] = trial['timeframe'].strip('"')
            if trial.get('folder'):
                trial['folder'] = trial['folder'].strip('"')

        return top_trials.to_dict('records')

    def get_convergence_data(self, window: int = 100) -> Dict:
        """Get convergence statistics."""
        conn = sqlite3.connect(self.db_path)

        # Get all completed trials ordered by completion time
        query = """
        SELECT
            t.number,
            tv.value,
            t.datetime_complete
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.datetime_complete
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {}

        # Calculate rolling statistics
        df['rolling_max'] = df['value'].expanding().max()
        df['rolling_avg'] = df['value'].rolling(window=window, min_periods=1).mean()
        df['rolling_std'] = df['value'].rolling(window=window, min_periods=1).std()

        # Recent performance
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

    def get_hyperparameter_stats(self) -> Dict:
        """Get hyperparameter distribution statistics."""
        conn = sqlite3.connect(self.db_path)

        # Get top 10% trials
        top10_query = """
        SELECT trial_id
        FROM trial_values
        WHERE trial_id IN (SELECT trial_id FROM trials WHERE state = 'COMPLETE')
        ORDER BY value DESC
        LIMIT (SELECT COUNT(*) / 10 FROM trials WHERE state = 'COMPLETE')
        """

        top10_ids = pd.read_sql_query(top10_query, conn)['trial_id'].tolist()

        if not top10_ids:
            conn.close()
            return {}

        # Get params for top 10%
        placeholders = ','.join('?' * len(top10_ids))
        params_query = f"""
        SELECT tp.param_name, tp.param_value
        FROM trial_params tp
        WHERE tp.trial_id IN ({placeholders})
        """

        params_df = pd.read_sql_query(params_query, conn, params=top10_ids)
        conn.close()

        # Calculate statistics for numeric params
        stats = {}
        for param in params_df['param_name'].unique():
            values = params_df[params_df['param_name'] == param]['param_value']

            # Try to convert to numeric
            try:
                numeric_values = pd.to_numeric(values)
                stats[param] = {
                    'mean': numeric_values.mean(),
                    'std': numeric_values.std(),
                    'min': numeric_values.min(),
                    'max': numeric_values.max(),
                }
            except:
                # Categorical param
                stats[param] = {
                    'mode': values.mode()[0] if len(values.mode()) > 0 else None,
                    'unique_count': values.nunique(),
                }

        return stats

    def get_training_velocity(self) -> Dict:
        """Calculate training velocity metrics."""
        conn = sqlite3.connect(self.db_path)

        time_windows = {
            'last_hour': 1,
            'last_6_hours': 6,
            'last_24_hours': 24,
        }

        velocity = {}
        for name, hours in time_windows.items():
            cutoff = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            query = """
            SELECT COUNT(*) as count
            FROM trials
            WHERE state = 'COMPLETE' AND datetime_complete > ?
            """
            cursor = conn.cursor()
            cursor.execute(query, (cutoff,))
            count = cursor.fetchone()[0]
            velocity[name] = count
            velocity[f'{name}_per_hour'] = count / hours

        conn.close()
        return velocity

    def display_dashboard(self):
        """Display the enhanced training dashboard."""
        self.clear_screen()

        width = 100
        print("=" * width)
        print(self.colorize("CAPPUCCINO TRAINING DASHBOARD - DETAILED VIEW", 'bold').center(width + 10))
        print("=" * width)
        print()

        # Study Overview
        print(self.colorize("┌─ STUDY OVERVIEW " + "─" * (width - 19) + "┐", 'cyan'))
        studies = self.get_study_overview()
        if studies:
            for study in studies[:5]:  # Show top 5 studies
                print(f"│ {study['study_name']:<40} │ Trials: {study['completed']:>4} │ Best: {study['best_value']:>8.6f} │ Avg: {study['avg_value']:>8.6f} │")
        else:
            print("│ No studies found" + " " * (width - 19) + "│")
        print("└" + "─" * (width - 2) + "┘")
        print()

        # Convergence Stats
        print(self.colorize("┌─ CONVERGENCE ANALYSIS " + "─" * (width - 25) + "┐", 'cyan'))
        conv = self.get_convergence_data()
        if conv:
            best_str = self.colorize(f"{conv['current_best']:.6f}", 'green')
            improvement_color = 'green' if conv['improvement_rate'] > 0 else 'red'
            improvement_str = self.colorize(f"{conv['improvement_rate']:+.2f}%", improvement_color)
            print(f"│ Total Trials: {conv['total_trials']:>5} │ Current Best: {best_str} │ Avg (Last 100): {conv['avg_last_100']:.6f} │ Std: {conv['std_last_100']:.6f}")
            print(f"│ Improvement Rate: {improvement_str} │ Avg (Last 500): {conv['avg_last_500']:.6f}")
        print("└" + "─" * (width - 2) + "┘")
        print()

        # Training Velocity
        print(self.colorize("┌─ TRAINING VELOCITY " + "─" * (width - 22) + "┐", 'cyan'))
        velocity = self.get_training_velocity()
        print(f"│ Last Hour: {velocity['last_hour']:>3} trials ({velocity['last_hour_per_hour']:.1f}/hr) │ Last 6h: {velocity['last_6_hours']:>3} ({velocity['last_6_hours_per_hour']:.1f}/hr) │ Last 24h: {velocity['last_24_hours']:>3} ({velocity['last_24_hours_per_hour']:.1f}/hr)")
        print("└" + "─" * (width - 2) + "┘")
        print()

        # Top Trials
        print(self.colorize("┌─ TOP 10 TRIALS " + "─" * (width - 18) + "┐", 'cyan'))
        top_trials = self.get_top_trials(10)
        print(f"│ {'Rank':<6} {'Trial#':<8} {'Sharpe':<12} {'Timeframe':<12} {'Folder':<40} │")
        print("│" + "─" * (width - 2) + "│")
        for i, trial in enumerate(top_trials, 1):
            color = 'green' if i <= 3 else 'cyan' if i <= 5 else 'white'
            sharpe_str = self.colorize(f"{trial['value']:.6f}", color)
            timeframe = trial.get('timeframe', 'N/A')
            folder = trial.get('folder', 'N/A')[-38:] if trial.get('folder') else 'N/A'
            print(f"│ {i:<6} {trial['number']:<8} {sharpe_str:<12} {timeframe:<12} {folder:<40} │")
        print("└" + "─" * (width - 2) + "┘")
        print()

        # Recent Trials
        print(self.colorize("┌─ RECENT COMPLETIONS " + "─" * (width - 23) + "┐", 'cyan'))
        recent = self.get_recent_trials(8)
        print(f"│ {'Trial#':<8} {'Sharpe':<12} {'Duration':<12} {'Completed':<25} │")
        print("│" + "─" * (width - 2) + "│")
        for trial in recent:
            completed = trial['datetime_complete'][:19] if trial['datetime_complete'] else 'N/A'
            duration = f"{trial.get('duration_hours', 0)}h" if trial.get('duration_hours') else 'N/A'
            value_str = f"{trial['value']:.6f}" if trial['value'] else 'N/A'
            print(f"│ {trial['number']:<8} {value_str:<12} {duration:<12} {completed:<25} │")
        print("└" + "─" * (width - 2) + "┘")
        print()

        # Hyperparameter Stats (Top 10%)
        print(self.colorize("┌─ TOP 10% HYPERPARAMETER RANGES " + "─" * (width - 34) + "┐", 'cyan'))
        hp_stats = self.get_hyperparameter_stats()
        important_params = ['learning_rate', 'gamma', 'net_dimension', 'batch_size', 'lookback']
        for param in important_params:
            if param in hp_stats:
                stats = hp_stats[param]
                if 'mean' in stats:
                    print(f"│ {param:<20}: Mean={stats['mean']:.6f} ± {stats['std']:.6f} │ Range=[{stats['min']:.6f}, {stats['max']:.6f}]")
                else:
                    print(f"│ {param:<20}: Mode={stats.get('mode', 'N/A')} │ Unique={stats.get('unique_count', 0)}")
        print("└" + "─" * (width - 2) + "┘")
        print()

        print(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to exit")


def main():
    parser = argparse.ArgumentParser(description='Enhanced Training Dashboard')
    parser.add_argument('--db', default='databases/optuna_cappuccino.db', help='Path to Optuna database')
    parser.add_argument('--refresh', type=int, default=10, help='Refresh interval in seconds (0 for single display)')
    args = parser.parse_args()

    dashboard = TrainingDashboard(args.db)

    if args.refresh == 0:
        dashboard.display_dashboard()
    else:
        import time
        try:
            while True:
                dashboard.display_dashboard()
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\n\nExiting...")


if __name__ == '__main__':
    main()
