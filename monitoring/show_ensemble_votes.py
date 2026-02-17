#!/usr/bin/env python3
"""
Show what the ensemble voted on for the last decision.

This reads the paper trading state and shows individual model predictions.
"""

import json
from pathlib import Path

import numpy as np

from config_main import TICKER_LIST


def show_voting_breakdown():
    """Display ensemble voting breakdown."""

    # Check if there's a voting log file
    vote_log = Path("paper_trades/ensemble_votes.json")

    if not vote_log.exists():
        print("No ensemble voting log found yet.")
        print("The ensemble needs to make at least one decision first.")
        return

    with vote_log.open("r") as f:
        votes = json.load(f)

    print("=" * 100)
    print("ENSEMBLE VOTING BREAKDOWN - LAST DECISION")
    print("=" * 100)
    print()

    timestamp = votes.get('timestamp', 'Unknown')
    print(f"Timestamp: {timestamp}")
    print()

    individual = votes['individual_predictions']
    ensemble = votes['ensemble_average']
    tickers = votes.get('ticker_names', TICKER_LIST)

    print("─" * 100)
    print("INDIVIDUAL MODEL PREDICTIONS")
    print("─" * 100)
    print()

    # Show as table
    header = f"{'Model':<10}"
    for ticker in tickers:
        ticker_short = ticker.replace('/USD', '')
        header += f"{ticker_short:>10}"
    print(header)
    print("-" * 100)

    for i, pred in enumerate(individual, 1):
        row = f"Model {i:<4}"
        for j, action in enumerate(pred[0]):
            row += f"{action:>10.4f}"
        print(row)

    print("-" * 100)
    print()

    # Show ensemble average
    print("─" * 100)
    print("ENSEMBLE AVERAGE (FINAL DECISION)")
    print("─" * 100)
    print()

    header = f"{'Ensemble':<10}"
    for ticker in tickers:
        ticker_short = ticker.replace('/USD', '')
        header += f"{ticker_short:>10}"
    print(header)
    print("-" * 100)

    row = f"Average   "
    for action in ensemble[0]:
        row += f"{action:>10.4f}"
    print(row)
    print()

    # Calculate agreement metrics
    print("─" * 100)
    print("AGREEMENT ANALYSIS")
    print("─" * 100)
    print()

    for j, ticker in enumerate(tickers):
        ticker_short = ticker.replace('/USD', '')
        actions = [pred[0][j] for pred in individual]

        mean_action = np.mean(actions)
        std_action = np.std(actions)
        min_action = np.min(actions)
        max_action = np.max(actions)

        # Agreement score: lower std = more agreement
        agreement = 1.0 / (1.0 + std_action)

        print(f"{ticker_short:>8}: Mean={mean_action:+.4f} | Std={std_action:.4f} | Range=[{min_action:+.4f}, {max_action:+.4f}] | Agreement={agreement:.3f}")

        # Show how many models agree on direction
        positive = sum(1 for a in actions if a > 0.01)
        negative = sum(1 for a in actions if a < -0.01)
        neutral = sum(1 for a in actions if -0.01 <= a <= 0.01)

        print(f"         Votes: {positive} BUY | {neutral} HOLD | {negative} SELL")
        print()

    print("=" * 100)


if __name__ == "__main__":
    show_voting_breakdown()
