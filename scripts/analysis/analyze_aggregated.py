#!/usr/bin/env python3
"""
Analyze Aggregated Paper Trading Data
Quick analysis queries on consolidated data.
"""

import argparse
import sqlite3
import sys

import pandas as pd


def query_trial_performance(db_path: str, trial_num: int = None):
    """Show performance by trial."""
    conn = sqlite3.connect(db_path)

    if trial_num:
        query = """
            SELECT
                trial_number,
                trial_sharpe,
                MIN(timestamp) as first_trade,
                MAX(timestamp) as last_trade,
                COUNT(*) as num_trades,
                MIN(total_asset) as min_portfolio,
                MAX(total_asset) as max_portfolio,
                (MAX(total_asset) - MIN(total_asset)) / MIN(total_asset) * 100 as return_pct
            FROM paper_trading_sessions
            WHERE trial_number = ?
            GROUP BY trial_number
        """
        df = pd.read_sql(query, conn, params=(trial_num,))
    else:
        query = """
            SELECT
                trial_number,
                trial_sharpe,
                MIN(timestamp) as first_trade,
                MAX(timestamp) as last_trade,
                COUNT(*) as num_trades,
                (MAX(total_asset) - MIN(total_asset)) / MIN(total_asset) * 100 as return_pct
            FROM paper_trading_sessions
            GROUP BY trial_number
            ORDER BY trial_sharpe DESC
        """
        df = pd.read_sql(query, conn)

    conn.close()
    return df


def query_best_performers(db_path: str, top_n: int = 10):
    """Show top performing trials."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT
            trial_number,
            trial_sharpe,
            COUNT(*) as num_trades,
            (MAX(total_asset) - MIN(total_asset)) / MIN(total_asset) * 100 as return_pct,
            AVG(reward) as avg_reward
        FROM paper_trading_sessions
        GROUP BY trial_number
        HAVING num_trades > 10
        ORDER BY return_pct DESC
        LIMIT ?
    """

    df = pd.read_sql(query, conn, params=(top_n,))
    conn.close()
    return df


def query_timeline(db_path: str, trial_num: int):
    """Get full timeline for a trial."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT timestamp, total_asset, reward, cash
        FROM paper_trading_sessions
        WHERE trial_number = ?
        ORDER BY timestamp
    """

    df = pd.read_sql(query, conn, params=(trial_num,))
    conn.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="paper_trades/aggregated.db", help="Database path")
    parser.add_argument("--trial", type=int, help="Show specific trial")
    parser.add_argument("--top", type=int, default=10, help="Show top N trials")
    parser.add_argument("--timeline", type=int, help="Show timeline for trial")
    args = parser.parse_args()

    if not args.db or not pd.io.sql.sqlite3.connect(args.db):
        print(f"Database not found: {args.db}")
        print("Run: python scripts/analysis/aggregate_paper_trading.py")
        sys.exit(1)

    if args.timeline:
        print(f"Timeline for Trial #{args.timeline}:")
        df = query_timeline(args.db, args.timeline)
        print(df.to_string(index=False))

    elif args.trial:
        print(f"Performance for Trial #{args.trial}:")
        df = query_trial_performance(args.db, args.trial)
        print(df.to_string(index=False))

    else:
        print(f"Top {args.top} Performers:")
        df = query_best_performers(args.db, args.top)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
