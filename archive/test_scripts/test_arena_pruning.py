#!/usr/bin/env python3
"""
Quick script to manually trigger arena pruning.

This is useful for testing the pruning mechanism without waiting 24 hours.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_arena import ModelArena


def main():
    """Load arena and force pruning."""
    print("Loading arena...")
    arena = ModelArena(
        max_models=10,
        min_evaluation_hours=48,  # Lower threshold for testing (2 days instead of 7)
        prune_interval_hours=24,
        below_average_threshold=0.25,  # Remove bottom 25%
    )

    print(f"\nCurrent models in arena: {len(arena.portfolios)}")

    if len(arena.portfolios) == 0:
        print("No models in arena!")
        return

    # Show current rankings
    rankings = arena.get_rankings()
    print("\nCurrent rankings:")
    print(f"{'Rank':<6} {'Model':<15} {'Return %':<10} {'Sortino':<10} {'Hours':<8} {'Eligible'}")
    print("-" * 70)

    for i, model in enumerate(rankings, 1):
        print(
            f"{i:<6} {model['model_id']:<15} {model['return_pct']:>8.2f}% "
            f"{model['sortino_ratio']:>8.2f}  {model['hours_active']:>6.0f}h  "
            f"{'YES' if model['eligible_for_promotion'] else 'NO'}"
        )

    # Calculate stats
    evaluated = [m for m in rankings if m['eligible_for_promotion']]
    if evaluated:
        avg_return = sum(m['return_pct'] for m in evaluated) / len(evaluated)
        avg_sortino = sum(m['sortino_ratio'] for m in evaluated if m['sortino_ratio'] > -50) / len(evaluated)
        print(f"\nAverage return: {avg_return:.2f}%")
        print(f"Average Sortino: {avg_sortino:.2f}")
        print(f"Evaluated models: {len(evaluated)}/{len(rankings)}")

    # Ask user to confirm pruning
    print("\n" + "=" * 70)
    response = input("Force prune underperformers now? (y/n): ")

    if response.lower() == 'y':
        print("\nPruning underperformers...")
        arena.force_prune_underperformers()

        print(f"\nModels remaining: {len(arena.portfolios)}")

        # Show updated rankings
        rankings = arena.get_rankings()
        if rankings:
            print("\nUpdated rankings:")
            print(f"{'Rank':<6} {'Model':<15} {'Return %':<10} {'Sortino':<10} {'Hours'}")
            print("-" * 60)
            for i, model in enumerate(rankings, 1):
                print(
                    f"{i:<6} {model['model_id']:<15} {model['return_pct']:>8.2f}% "
                    f"{model['sortino_ratio']:>8.2f}  {model['hours_active']:>6.0f}h"
                )
        else:
            print("No models remaining in arena!")
    else:
        print("Pruning cancelled.")


if __name__ == "__main__":
    main()
