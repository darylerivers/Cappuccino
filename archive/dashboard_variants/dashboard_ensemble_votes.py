#!/usr/bin/env python3
"""
Enhanced Ensemble Voting Dashboard

Shows what each model voted on plus their performance metrics in a live dashboard.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class EnsembleVotesDashboard:
    """Live dashboard for ensemble voting breakdown with model metrics."""

    def __init__(self):
        self.votes_file = Path("paper_trades/ensemble_votes.json")
        self.manifest_file = Path("train_results/ensemble/ensemble_manifest.json")

    def clear_screen(self):
        """Clear terminal screen."""
        print("\033[H\033[2J", end="")

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
            'reset': '\033[0m',
        }
        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def load_data(self):
        """Load voting data and model metrics."""
        if not self.votes_file.exists():
            return None, None

        with self.votes_file.open("r") as f:
            votes = json.load(f)

        manifest = None
        if self.manifest_file.exists():
            with self.manifest_file.open("r") as f:
                manifest = json.load(f)

        return votes, manifest

    def render(self):
        """Render the dashboard."""
        votes, manifest = self.load_data()

        if votes is None:
            self.clear_screen()
            print(self.colorize("=" * 120, 'cyan'))
            print(self.colorize("ENSEMBLE VOTING DASHBOARD", 'bold'))
            print(self.colorize("=" * 120, 'cyan'))
            print()
            print(self.colorize("⏳ Waiting for first trading decision...", 'yellow'))
            print()
            print("The ensemble needs to make at least one decision first.")
            print(f"Votes file: {self.votes_file}")
            print()
            print(self.colorize("=" * 120, 'cyan'))
            return

        self.clear_screen()
        print(self.colorize("=" * 120, 'cyan'))
        print(self.colorize("ENSEMBLE VOTING DASHBOARD", 'bold'))
        print(self.colorize("=" * 120, 'cyan'))
        print()

        # Header info
        timestamp = votes.get('timestamp', 'Unknown')
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        time_ago = (datetime.now(dt.tzinfo) - dt).total_seconds()
        if time_ago < 60:
            time_str = f"{int(time_ago)}s ago"
        elif time_ago < 3600:
            time_str = f"{int(time_ago/60)}m ago"
        else:
            time_str = f"{int(time_ago/3600)}h ago"

        print(f"Last Decision: {self.colorize(dt.strftime('%Y-%m-%d %H:%M:%S UTC'), 'cyan')} ({time_str})")
        print()

        individual = votes['individual_predictions']
        ensemble = votes['ensemble_average']
        tickers = votes.get('ticker_names', [])
        num_models = votes['num_models']

        # Model Performance Metrics
        if manifest:
            print(self.colorize("┌─ MODEL PERFORMANCE METRICS " + "─" * 91 + "┐", 'cyan'))
            print(self.colorize("│", 'cyan') + " " * 118 + self.colorize("│", 'cyan'))

            trial_nums = manifest['trial_numbers']
            trial_vals = manifest['trial_values']

            header = f"│ {'Model':<8} │ {'Trial':<8} │ {'Value':<12} │ {'Rank':<6} │ {'Percentile':<11} │"
            print(self.colorize(header, 'cyan'))
            print(self.colorize("│" + "─" * 118 + "│", 'cyan'))

            for i, (trial_num, trial_val) in enumerate(zip(trial_nums, trial_vals), 1):
                # Calculate percentile (all models are top 10%)
                percentile = 100 - (i * 10 / num_models)

                # Color code by performance
                if trial_val >= manifest['mean_value']:
                    val_str = self.colorize(f"+{trial_val:.6f}", 'green')
                else:
                    val_str = self.colorize(f"+{trial_val:.6f}", 'yellow')

                row = f"│ Model {i:<2} │ #{trial_num:<7} │ {val_str:<21} │ {i:<6} │ Top {percentile:>4.0f}%   │"
                print(row)

            print(self.colorize("│" + "─" * 118 + "│", 'cyan'))
            mean_str = self.colorize(f"+{manifest['mean_value']:.6f}", 'cyan')
            best_str = self.colorize(f"+{manifest['best_value']:.6f}", 'green')
            print(f"│ {'Ensemble Mean:':<20} {mean_str:<21} │ {'Best:':<8} {best_str:<21} │")
            print(self.colorize("│", 'cyan') + " " * 118 + self.colorize("│", 'cyan'))
            print(self.colorize("└" + "─" * 118 + "┘", 'cyan'))
            print()

        # Individual Model Predictions
        print(self.colorize("┌─ INDIVIDUAL MODEL PREDICTIONS " + "─" * 87 + "┐", 'cyan'))
        print(self.colorize("│", 'cyan') + " " * 118 + self.colorize("│", 'cyan'))

        # Build header
        header = f"│ {'Model':<10} │"
        for ticker in tickers:
            ticker_short = ticker.replace('/USD', '')
            header += f" {ticker_short:>8} │"
        print(self.colorize(header, 'cyan'))
        print(self.colorize("│" + "─" * 118 + "│", 'cyan'))

        # Show each model's prediction
        for i, pred in enumerate(individual, 1):
            row = f"│ Model {i:<4} │"
            for j, action in enumerate(pred[0]):
                # Color code actions
                if action > 0.05:
                    action_str = self.colorize(f"{action:+.4f}", 'green')
                elif action < -0.05:
                    action_str = self.colorize(f"{action:+.4f}", 'red')
                else:
                    action_str = f"{action:+.4f}"
                row += f" {action_str:>17} │"
            print(row)

        print(self.colorize("└" + "─" * 118 + "┘", 'cyan'))
        print()

        # Ensemble Average
        print(self.colorize("┌─ ENSEMBLE AVERAGE (FINAL DECISION) " + "─" * 81 + "┐", 'cyan'))
        print(self.colorize("│", 'cyan') + " " * 118 + self.colorize("│", 'cyan'))

        header = f"│ {'Average':<10} │"
        for ticker in tickers:
            ticker_short = ticker.replace('/USD', '')
            header += f" {ticker_short:>8} │"
        print(self.colorize(header, 'cyan'))
        print(self.colorize("│" + "─" * 118 + "│", 'cyan'))

        row = f"│ Ensemble   │"
        for action in ensemble[0]:
            if action > 0.05:
                action_str = self.colorize(f"{action:+.4f}", 'green')
            elif action < -0.05:
                action_str = self.colorize(f"{action:+.4f}", 'red')
            else:
                action_str = f"{action:+.4f}"
            row += f" {action_str:>17} │"
        print(row)

        print(self.colorize("│", 'cyan') + " " * 118 + self.colorize("│", 'cyan'))
        print(self.colorize("└" + "─" * 118 + "┘", 'cyan'))
        print()

        # Agreement Analysis
        print(self.colorize("┌─ CONSENSUS ANALYSIS " + "─" * 96 + "┐", 'cyan'))
        print(self.colorize("│", 'cyan') + " " * 118 + self.colorize("│", 'cyan'))

        header = f"│ {'Asset':<8} │ {'Mean':<10} │ {'Std Dev':<10} │ {'Range':<20} │ {'Agreement':<10} │ {'Votes':<30} │"
        print(self.colorize(header, 'cyan'))
        print(self.colorize("│" + "─" * 118 + "│", 'cyan'))

        for j, ticker in enumerate(tickers):
            ticker_short = ticker.replace('/USD', '')
            actions = [pred[0][j] for pred in individual]

            mean_action = np.mean(actions)
            std_action = np.std(actions)
            min_action = np.min(actions)
            max_action = np.max(actions)

            # Agreement score: lower std = more agreement
            agreement = 1.0 / (1.0 + std_action)

            # Count votes
            positive = sum(1 for a in actions if a > 0.01)
            negative = sum(1 for a in actions if a < -0.01)
            neutral = sum(1 for a in actions if -0.01 <= a <= 0.01)

            # Color code agreement
            if agreement > 0.995:
                agree_str = self.colorize(f"{agreement:.3f}", 'green')
            elif agreement > 0.990:
                agree_str = self.colorize(f"{agreement:.3f}", 'cyan')
            else:
                agree_str = self.colorize(f"{agreement:.3f}", 'yellow')

            # Format votes
            vote_parts = []
            if positive > 0:
                vote_parts.append(self.colorize(f"{positive} BUY", 'green'))
            if neutral > 0:
                vote_parts.append(f"{neutral} HOLD")
            if negative > 0:
                vote_parts.append(self.colorize(f"{negative} SELL", 'red'))
            votes_str = " | ".join(vote_parts)

            range_str = f"[{min_action:+.4f}, {max_action:+.4f}]"

            # Calculate actual displayed width
            row = f"│ {ticker_short:<8} │ {mean_action:+.4f}    │ {std_action:.4f}    │ {range_str:<20} │ {agree_str:>19} │ {votes_str}"
            # Pad to ensure proper alignment
            displayed_len = len(row) - (len(agree_str) - len(f"{agreement:.3f}")) - (len(votes_str) - len(vote_parts[0] if vote_parts else ""))
            padding = 118 - displayed_len + 1
            row += " " * padding + "│"
            print(row)

        print(self.colorize("│", 'cyan') + " " * 118 + self.colorize("│", 'cyan'))
        print(self.colorize("└" + "─" * 118 + "┘", 'cyan'))
        print()

        # Footer
        print(self.colorize("─" * 120, 'cyan'))
        print(self.colorize("Press Ctrl+C to exit | Refreshing every 3 seconds", 'white'))
        print(self.colorize("=" * 120, 'cyan'))

    def run(self):
        """Run the dashboard in a loop."""
        try:
            while True:
                self.render()
                time.sleep(3)
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)


def main():
    """Entry point."""
    dashboard = EnsembleVotesDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
