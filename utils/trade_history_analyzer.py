#!/usr/bin/env python3
"""
Trade History Analyzer

Extracts and analyzes all trades from paper trading sessions:
- Entry and exit prices
- Profit/Loss per trade
- Win/loss statistics
- Best and worst trades
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class Trade:
    """Represents a single trade (entry or exit)."""
    timestamp: datetime
    ticker: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    value: float
    portfolio_value: float


@dataclass
class CompletedTrade:
    """Represents a completed round-trip trade with P&L."""
    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    entry_value: float
    exit_value: float
    pnl: float
    pnl_pct: float
    holding_period_hours: float
    entry_fee: float = 0.0
    exit_fee: float = 0.0
    pnl_gross: float = 0.0  # P&L before fees


class TradeHistoryAnalyzer:
    """Analyze paper trading sessions and extract trade history."""

    def __init__(self, session_files: List[Path], buy_fee_pct: float = 0.0025, sell_fee_pct: float = 0.0025):
        self.session_files = session_files
        self.buy_fee_pct = buy_fee_pct  # 0.25% default
        self.sell_fee_pct = sell_fee_pct  # 0.25% default
        self.df = self._load_sessions()
        self.tickers = self._extract_tickers()

    def _load_sessions(self) -> pd.DataFrame:
        """Load and combine all session files."""
        dfs = []
        for file in self.session_files:
            if file.exists():
                try:
                    df = pd.read_csv(file)
                    if len(df) > 0:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
        return combined

    def _extract_tickers(self) -> List[str]:
        """Extract ticker symbols from column names."""
        if self.df.empty:
            return []

        action_cols = [col for col in self.df.columns if col.startswith('action_')]
        return [col.replace('action_', '') for col in action_cols]

    def extract_all_trades(self) -> List[Trade]:
        """Extract all individual trade actions."""
        if self.df.empty:
            return []

        trades = []

        for idx, row in self.df.iterrows():
            timestamp = row['timestamp']
            portfolio_value = row['total_asset']

            for ticker in self.tickers:
                action_col = f'action_{ticker}'
                price_col = f'price_{ticker}'

                if action_col not in row or price_col not in row:
                    continue

                action_qty = row[action_col]
                price = row[price_col]

                # Significant action (threshold to avoid noise)
                if abs(action_qty) > 0.0001:
                    action_type = 'BUY' if action_qty > 0 else 'SELL'
                    value = abs(action_qty) * price

                    trade = Trade(
                        timestamp=timestamp,
                        ticker=ticker,
                        action=action_type,
                        quantity=abs(action_qty),
                        price=price,
                        value=value,
                        portfolio_value=portfolio_value
                    )
                    trades.append(trade)

        return trades

    def extract_completed_trades(self) -> List[CompletedTrade]:
        """Extract completed round-trip trades (buy then sell) with P&L."""
        if self.df.empty:
            return []

        completed_trades = []

        for ticker in self.tickers:
            holding_col = f'holding_{ticker}'
            price_col = f'price_{ticker}'
            action_col = f'action_{ticker}'

            if holding_col not in self.df.columns:
                continue

            # Track position state
            position_qty = 0
            position_entry_price = 0
            position_entry_time = None
            position_entries = []  # Track multiple entries for averaging

            for idx, row in self.df.iterrows():
                timestamp = row['timestamp']
                action_qty = row[action_col]
                price = row[price_col]
                holdings = row[holding_col]

                if abs(action_qty) < 0.0001:
                    continue

                # Buy action
                if action_qty > 0:
                    if position_qty == 0:
                        # New position
                        position_entry_price = price
                        position_entry_time = timestamp
                        position_entries = [(timestamp, price, action_qty)]
                    else:
                        # Adding to position - calculate new average entry
                        total_cost = position_qty * position_entry_price + action_qty * price
                        position_qty_new = position_qty + action_qty
                        position_entry_price = total_cost / position_qty_new if position_qty_new > 0 else price
                        position_entries.append((timestamp, price, action_qty))

                    position_qty += action_qty

                # Sell action
                elif action_qty < 0:
                    sell_qty = abs(action_qty)

                    if position_qty > 0:
                        # Calculate realized P&L for the portion being sold
                        sell_qty_actual = min(sell_qty, position_qty)

                        # Calculate entry and exit values
                        entry_value = sell_qty_actual * position_entry_price
                        exit_value = sell_qty_actual * price

                        # Calculate fees
                        entry_fee = entry_value * self.buy_fee_pct
                        exit_fee = exit_value * self.sell_fee_pct

                        # Gross P&L (before fees)
                        pnl_gross = exit_value - entry_value

                        # Net P&L (after fees) - this is the actual profit/loss
                        pnl_net = (exit_value - exit_fee) - (entry_value + entry_fee)

                        # P&L percentage based on net return on invested capital (including entry fee)
                        cost_basis = entry_value + entry_fee
                        pnl_pct = (pnl_net / cost_basis * 100) if cost_basis > 0 else 0

                        holding_period = (timestamp - position_entry_time).total_seconds() / 3600

                        completed_trade = CompletedTrade(
                            ticker=ticker,
                            entry_time=position_entry_time,
                            exit_time=timestamp,
                            entry_price=position_entry_price,
                            exit_price=price,
                            quantity=sell_qty_actual,
                            entry_value=entry_value,
                            exit_value=exit_value,
                            pnl=pnl_net,  # Net P&L after fees
                            pnl_pct=pnl_pct,
                            holding_period_hours=holding_period,
                            entry_fee=entry_fee,
                            exit_fee=exit_fee,
                            pnl_gross=pnl_gross,  # P&L before fees
                        )
                        completed_trades.append(completed_trade)

                        # Update position
                        position_qty -= sell_qty_actual

                        # If position fully closed, reset
                        if position_qty < 0.0001:
                            position_qty = 0
                            position_entry_price = 0
                            position_entry_time = None
                            position_entries = []

        return completed_trades

    def generate_trade_report(self) -> str:
        """Generate detailed trade history report."""
        all_trades = self.extract_all_trades()
        completed_trades = self.extract_completed_trades()

        report = []
        report.append("=" * 100)
        report.append("TRADE HISTORY REPORT")
        report.append("=" * 100)
        report.append("")

        # Summary stats
        report.append("SUMMARY")
        report.append("-" * 100)
        report.append(f"  Total Actions:        {len(all_trades)}")
        report.append(f"  Completed Trades:     {len(completed_trades)}")

        if completed_trades:
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            total_pnl_net = sum(t.pnl for t in completed_trades)
            total_pnl_gross = sum(t.pnl_gross for t in completed_trades)
            total_fees = sum(t.entry_fee + t.exit_fee for t in completed_trades)
            win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0

            report.append(f"  Winning Trades:       {len(winning_trades)} ({win_rate:.1f}%)")
            report.append(f"  Losing Trades:        {len(losing_trades)}")
            report.append(f"  Total P&L (Net):      ${total_pnl_net:,.2f}")
            report.append(f"  Total P&L (Gross):    ${total_pnl_gross:,.2f}")
            report.append(f"  Total Fees:           ${total_fees:,.2f}")
            report.append("")

            # Best/worst trades
            if winning_trades:
                best_trade = max(winning_trades, key=lambda t: t.pnl)
                report.append(f"  Best Trade:           {best_trade.ticker} +${best_trade.pnl:.2f} ({best_trade.pnl_pct:+.1f}%)")

            if losing_trades:
                worst_trade = min(losing_trades, key=lambda t: t.pnl)
                report.append(f"  Worst Trade:          {worst_trade.ticker} -${abs(worst_trade.pnl):.2f} ({worst_trade.pnl_pct:.1f}%)")

        report.append("")

        # All trades chronologically
        report.append("ALL TRADE ACTIONS")
        report.append("-" * 100)
        report.append(f"{'Time':<20s} {'Ticker':<12s} {'Action':<6s} {'Quantity':>12s} {'Price':>12s} {'Value':>12s} {'Portfolio':>12s}")
        report.append("-" * 100)

        for trade in all_trades[:50]:  # Limit to first 50 for readability
            report.append(
                f"{trade.timestamp.strftime('%Y-%m-%d %H:%M'):<20s} "
                f"{trade.ticker:<12s} "
                f"{trade.action:<6s} "
                f"{trade.quantity:>12.6f} "
                f"${trade.price:>11.2f} "
                f"${trade.value:>11.2f} "
                f"${trade.portfolio_value:>11.2f}"
            )

        if len(all_trades) > 50:
            report.append(f"... ({len(all_trades) - 50} more trades)")

        report.append("")

        # Completed trades with P&L
        if completed_trades:
            report.append("COMPLETED TRADES (with P&L)")
            report.append("-" * 120)
            report.append(f"{'Ticker':<12s} {'Entry Time':<17s} {'Exit Time':<17s} "
                         f"{'Entry $':>10s} {'Exit $':>10s} {'Qty':>10s} {'P&L Net':>12s} {'P&L %':>10s} "
                         f"{'Fees':>8s} {'Hours':>8s}")
            report.append("-" * 120)

            for trade in completed_trades:
                total_fees = trade.entry_fee + trade.exit_fee
                report.append(
                    f"{trade.ticker:<12s} "
                    f"{trade.entry_time.strftime('%m-%d %H:%M'):<17s} "
                    f"{trade.exit_time.strftime('%m-%d %H:%M'):<17s} "
                    f"${trade.entry_price:>9.2f} "
                    f"${trade.exit_price:>9.2f} "
                    f"{trade.quantity:>10.4f} "
                    f"${trade.pnl:>11.2f} "
                    f"{trade.pnl_pct:>+9.1f}% "
                    f"${total_fees:>7.2f} "
                    f"{trade.holding_period_hours:>7.1f}h"
                )

        report.append("")
        report.append("=" * 100)
        return "\n".join(report)

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Return completed trades as DataFrame for dashboard display."""
        completed_trades = self.extract_completed_trades()

        if not completed_trades:
            return pd.DataFrame()

        data = []
        for trade in completed_trades:
            data.append({
                'ticker': trade.ticker,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'holding_hours': trade.holding_period_hours,
            })

        return pd.DataFrame(data)


def main():
    """CLI for trade history analysis."""
    # Find session files
    session_files = [
        Path("paper_trades/ensemble_session.csv"),
        Path("paper_trades/alpaca_session.csv"),
    ]

    # Add dated files
    paper_trades_dir = Path("paper_trades")
    if paper_trades_dir.exists():
        for f in paper_trades_dir.glob("*session*.csv"):
            if f not in session_files:
                session_files.append(f)

    analyzer = TradeHistoryAnalyzer(session_files)
    report = analyzer.generate_trade_report()
    print(report)


if __name__ == "__main__":
    main()
