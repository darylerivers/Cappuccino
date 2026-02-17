#!/usr/bin/env python3
"""
Performance Grading System for Paper Trading
Evaluates paper trader performance and determines promotion to live trading.

Grading Criteria:
- Minimum 1 week of trading history
- 60% success rate threshold
- Positive alpha vs market
- Risk-adjusted returns (Sharpe ratio)
- Maximum drawdown limits

Usage:
    python performance_grader.py --check
    python performance_grader.py --status
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PerformanceGrader:
    def __init__(
        self,
        paper_trades_dir: str = "paper_trades",
        state_file: str = "deployments/grading_state.json",
        min_days: int = 7,
        success_threshold: float = 0.60,  # 60% success rate
    ):
        self.paper_trades_dir = Path(paper_trades_dir)
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(exist_ok=True)

        self.min_days = min_days
        self.success_threshold = success_threshold

        # Grading thresholds
        self.thresholds = {
            "min_win_rate": 0.60,  # 60% of trades must be profitable
            "min_alpha": 0.0,  # Must beat market (positive alpha)
            "min_sharpe": 0.5,  # Decent risk-adjusted returns
            "max_drawdown": 0.15,  # Max 15% drawdown
            "min_total_return": 0.0,  # Must be profitable overall
            "min_trades": 20,  # Minimum number of trades for statistical significance
        }

        self.state = self._load_state()
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/performance_grader.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_state(self) -> Dict:
        """Load grading state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "promoted_to_live": False,
            "promotion_date": None,
            "last_grade": None,
            "last_grade_date": None,
            "grade_history": [],
        }

    def _save_state(self):
        """Save grading state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def load_trading_history(self, days: int = 7) -> Optional[pd.DataFrame]:
        """Load trading history from CSV files."""
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Find all session CSV files
        csv_files = list(self.paper_trades_dir.glob("watchdog_session_*.csv"))
        csv_files.extend(list(self.paper_trades_dir.glob("alpaca_session*.csv")))
        csv_files.extend(list(self.paper_trades_dir.glob("ensemble_session*.csv")))

        if not csv_files:
            self.logger.warning("No trading session files found")
            return None

        # Load and combine data from recent sessions
        dfs = []
        for csv_file in csv_files:
            # Check file modification time
            from datetime import timezone
            mtime = datetime.fromtimestamp(csv_file.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff_date:
                continue

            try:
                df = pd.read_csv(csv_file)
                if len(df) > 0:
                    df['source_file'] = csv_file.name
                    dfs.append(df)
            except Exception as e:
                self.logger.warning(f"Error loading {csv_file}: {e}")
                continue

        if not dfs:
            self.logger.warning(f"No trading data from last {days} days")
            return None

        # Combine all data
        combined = pd.concat(dfs, ignore_index=True)

        # Parse timestamps
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])

        # Filter to last N days
        combined = combined[combined['timestamp'] >= cutoff_date]

        # Sort by timestamp
        combined = combined.sort_values('timestamp')

        return combined

    def calculate_trades(self, df: pd.DataFrame) -> List[Dict]:
        """Extract completed round-trip trades from trading history."""
        trades = []

        # Get list of tickers (columns with holding_ prefix)
        tickers = [col.replace('holding_', '') for col in df.columns if col.startswith('holding_')]

        for ticker in tickers:
            holding_col = f'holding_{ticker}'
            price_col = f'price_{ticker}'

            if holding_col not in df.columns or price_col not in df.columns:
                continue

            # Track positions
            in_position = False
            entry_idx = None

            for idx, row in df.iterrows():
                position = row[holding_col]

                if position > 0 and not in_position:
                    # Entry
                    in_position = True
                    entry_idx = idx

                elif position == 0 and in_position:
                    # Exit - calculate P&L
                    entry_row = df.loc[entry_idx]
                    exit_row = row

                    entry_price = entry_row[price_col]
                    exit_price = exit_row[price_col]
                    quantity = entry_row[holding_col]

                    # Calculate P&L including fees (0.25% buy + 0.25% sell = 0.5% round-trip)
                    entry_value = quantity * entry_price
                    exit_value = quantity * exit_price
                    entry_fee = entry_value * 0.0025  # 0.25% buy fee
                    exit_fee = exit_value * 0.0025    # 0.25% sell fee

                    # Net P&L after fees
                    pnl_net = (exit_value - exit_fee) - (entry_value + entry_fee)
                    pnl_pct = (pnl_net / (entry_value + entry_fee) * 100) if (entry_value + entry_fee) > 0 else 0

                    trade = {
                        'ticker': ticker,
                        'entry_time': entry_row['timestamp'],
                        'exit_time': exit_row['timestamp'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'pnl': pnl_net,  # Net P&L after fees
                        'pnl_pct': pnl_pct,
                        'profitable': pnl_net > 0,  # Only count as win if profitable after fees
                    }
                    trades.append(trade)

                    in_position = False
                    entry_idx = None

        return trades

    def calculate_metrics(self, df: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        if df.empty or not trades:
            return None

        # Portfolio metrics
        initial_value = df['total_asset'].iloc[0]
        final_value = df['total_asset'].iloc[-1]
        total_return = (final_value / initial_value - 1)

        # Returns series for Sharpe calculation
        returns = df['total_asset'].pct_change().dropna()

        # Sharpe ratio (annualized, assuming hourly data)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(24 * 365)

        # Maximum drawdown
        cumulative = (1 + df['total_asset'].pct_change()).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # Trade metrics
        trade_count = len(trades)
        winning_trades = sum(1 for t in trades if t['profitable'])
        win_rate = winning_trades / trade_count if trade_count > 0 else 0

        avg_win = np.mean([t['pnl'] for t in trades if t['profitable']]) if winning_trades > 0 else 0
        losing_trades = [t for t in trades if not t['profitable']]
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0

        # Profit factor
        total_wins = sum(t['pnl'] for t in trades if t['profitable'])
        total_losses = abs(sum(t['pnl'] for t in trades if not t['profitable']))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Time range
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        days_of_data = (end_date - start_date).total_seconds() / 86400

        return {
            'total_return': total_return,
            'final_value': final_value,
            'initial_value': initial_value,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': len(losing_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'days_of_data': days_of_data,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
        }

    def get_market_return(self, start_date: datetime, end_date: datetime) -> float:
        """Get market return for alpha calculation."""
        try:
            from dotenv import load_dotenv
            load_dotenv()

            import alpaca_trade_api as tradeapi
            from alpaca_trade_api.rest import TimeFrame

            api = tradeapi.REST(
                os.getenv("ALPACA_API_KEY"),
                os.getenv("ALPACA_SECRET_KEY"),
                os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            )

            # Use BTC as market proxy
            bars = api.get_crypto_bars(
                "BTC/USD",
                TimeFrame.Hour,
                start=start_date.isoformat(),
                end=end_date.isoformat()
            ).df

            if len(bars) > 0:
                return (bars['close'].iloc[-1] / bars['close'].iloc[0] - 1)
        except Exception as e:
            self.logger.warning(f"Could not get market return: {e}")

        return 0.0

    def calculate_grade(self, metrics: Dict, market_return: float) -> Dict:
        """Calculate grade based on metrics."""
        if not metrics:
            return {
                'passed': False,
                'grade': 'F',
                'score': 0,
                'reason': 'No trading data available',
            }

        # Calculate alpha
        alpha = metrics['total_return'] - market_return

        # Check each criterion
        checks = {}
        checks['min_days'] = metrics['days_of_data'] >= self.min_days
        checks['min_trades'] = metrics['trade_count'] >= self.thresholds['min_trades']
        checks['win_rate'] = metrics['win_rate'] >= self.thresholds['min_win_rate']
        checks['alpha'] = alpha >= self.thresholds['min_alpha']
        checks['sharpe'] = metrics['sharpe_ratio'] >= self.thresholds['min_sharpe']
        checks['drawdown'] = metrics['max_drawdown'] >= -self.thresholds['max_drawdown']
        checks['total_return'] = metrics['total_return'] >= self.thresholds['min_total_return']

        # Calculate score (percentage of checks passed)
        passed_count = sum(checks.values())
        total_count = len(checks)
        score = passed_count / total_count

        # Determine grade
        if score >= 0.95:
            grade = 'A'
        elif score >= 0.85:
            grade = 'B'
        elif score >= 0.70:
            grade = 'C'
        elif score >= 0.50:
            grade = 'D'
        else:
            grade = 'F'

        # Overall pass/fail
        passed = all([
            checks['min_days'],
            checks['min_trades'],
            checks['win_rate'],
            checks['alpha'],
            checks['total_return'],
        ])

        # Generate report
        failed_checks = [k for k, v in checks.items() if not v]
        reason = "All criteria met" if passed else f"Failed: {', '.join(failed_checks)}"

        # Convert numpy bool_ to native Python bool for JSON serialization
        checks_serializable = {k: bool(v) for k, v in checks.items()}

        return {
            'passed': bool(passed),
            'grade': grade,
            'score': float(score * 100),
            'alpha': float(alpha),
            'checks': checks_serializable,
            'reason': reason,
            'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()},
            'market_return': float(market_return),
        }

    def evaluate(self) -> Optional[Dict]:
        """Evaluate current paper trading performance."""
        self.logger.info("=" * 80)
        self.logger.info("PERFORMANCE GRADING EVALUATION")
        self.logger.info("=" * 80)

        # Load trading history
        df = self.load_trading_history(days=self.min_days)
        if df is None or df.empty:
            self.logger.warning("No trading data available for grading")
            return None

        self.logger.info(f"Loaded {len(df)} data points from last {self.min_days} days")

        # Extract trades
        trades = self.calculate_trades(df)
        self.logger.info(f"Found {len(trades)} completed trades")

        if not trades:
            self.logger.warning("No completed trades found")
            return None

        # Calculate metrics
        metrics = self.calculate_metrics(df, trades)
        if not metrics:
            return None

        # Get market return for alpha
        start_date = pd.to_datetime(metrics['start_date'])
        end_date = pd.to_datetime(metrics['end_date'])
        market_return = self.get_market_return(start_date, end_date)

        # Calculate grade
        grade_result = self.calculate_grade(metrics, market_return)

        # Log results
        self.logger.info("")
        self.logger.info("PERFORMANCE METRICS:")
        self.logger.info(f"  Trading Period: {metrics['days_of_data']:.1f} days")
        self.logger.info(f"  Total Trades: {metrics['trade_count']}")
        self.logger.info(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
        self.logger.info(f"  Total Return: {metrics['total_return']*100:+.2f}%")
        self.logger.info(f"  Market Return: {market_return*100:+.2f}%")
        self.logger.info(f"  Alpha: {grade_result['alpha']*100:+.2f}%")
        self.logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
        self.logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        self.logger.info("")
        self.logger.info("GRADING RESULT:")
        self.logger.info(f"  Grade: {grade_result['grade']}")
        self.logger.info(f"  Score: {grade_result['score']:.1f}%")
        self.logger.info(f"  Status: {'✅ PASSED' if grade_result['passed'] else '❌ FAILED'}")
        self.logger.info(f"  Reason: {grade_result['reason']}")
        self.logger.info("")

        # Update state
        self.state['last_grade'] = grade_result
        self.state['last_grade_date'] = datetime.now().isoformat()
        self.state['grade_history'].append({
            'date': datetime.now().isoformat(),
            'grade': grade_result['grade'],
            'score': grade_result['score'],
            'passed': grade_result['passed'],
        })

        # Keep only last 30 grades
        if len(self.state['grade_history']) > 30:
            self.state['grade_history'] = self.state['grade_history'][-30:]

        # Check for promotion
        if grade_result['passed'] and not self.state['promoted_to_live']:
            self.logger.info("🎉 PROMOTION CRITERIA MET!")
            self.logger.info("   Paper trader has passed all requirements")
            self.logger.info("   Ready for promotion to live trading")
            self.state['ready_for_promotion'] = True

        self._save_state()
        self.logger.info("=" * 80)

        return grade_result

    def promote_to_live(self):
        """Promote paper trader to live trading."""
        if self.state.get('promoted_to_live', False):
            self.logger.warning("Already promoted to live trading")
            return False

        if not self.state.get('ready_for_promotion', False):
            self.logger.error("Not ready for promotion - criteria not met")
            return False

        self.logger.info("=" * 80)
        self.logger.info("PROMOTING TO LIVE TRADING")
        self.logger.info("=" * 80)

        self.state['promoted_to_live'] = True
        self.state['promotion_date'] = datetime.now().isoformat()
        self._save_state()

        self.logger.info("✅ Successfully promoted to live trading!")
        self.logger.info(f"   Promotion date: {self.state['promotion_date']}")
        self.logger.info("=" * 80)

        return True

    def get_status(self) -> Dict:
        """Get current grading status."""
        return {
            'promoted_to_live': self.state.get('promoted_to_live', False),
            'ready_for_promotion': self.state.get('ready_for_promotion', False),
            'promotion_date': self.state.get('promotion_date'),
            'last_grade': self.state.get('last_grade'),
            'last_grade_date': self.state.get('last_grade_date'),
            'grade_history_count': len(self.state.get('grade_history', [])),
        }

    def check_promotion_eligibility(self, trial_number: Optional[int] = None,
                                   required_grades: List[str] = ['A', 'B']) -> Tuple[bool, Optional[Dict]]:
        """
        Check if trial is eligible for promotion to live trading.

        Args:
            trial_number: Specific trial to check (optional)
            required_grades: List of acceptable grades for promotion

        Returns:
            (eligible, metrics) - True if eligible, along with performance metrics
        """
        # Evaluate current performance
        grade_result = self.evaluate()

        if not grade_result:
            self.logger.info("No performance data available for promotion check")
            return False, None

        # Check if grade meets requirements
        grade = grade_result.get('grade', 'F')
        passed = grade_result.get('passed', False)

        if not passed:
            self.logger.info(f"Trial not eligible: Failed grading ({grade})")
            return False, grade_result

        if grade not in required_grades:
            self.logger.info(f"Trial not eligible: Grade {grade} not in required grades {required_grades}")
            return False, grade_result

        # Check minimum days requirement
        metrics = grade_result.get('metrics', {})
        days_evaluated = metrics.get('days_of_data', 0)

        if days_evaluated < self.min_days:
            self.logger.info(f"Trial not eligible: Only {days_evaluated:.1f} days evaluated (need {self.min_days})")
            return False, grade_result

        # All criteria met
        self.logger.info(f"Trial ELIGIBLE for live trading promotion (Grade: {grade}, Days: {days_evaluated:.1f})")
        return True, grade_result


def main():
    parser = argparse.ArgumentParser(description="Paper trading performance grader")
    parser.add_argument("--check", action="store_true", help="Run grading evaluation")
    parser.add_argument("--promote", action="store_true", help="Promote to live trading")
    parser.add_argument("--status", action="store_true", help="Show grading status")
    parser.add_argument("--min-days", type=int, default=7, help="Minimum days of data")
    parser.add_argument("--threshold", type=float, default=0.60, help="Success threshold")

    args = parser.parse_args()

    grader = PerformanceGrader(
        min_days=args.min_days,
        success_threshold=args.threshold,
    )

    if args.check:
        grader.evaluate()
    elif args.promote:
        grader.promote_to_live()
    elif args.status:
        status = grader.get_status()
        print(json.dumps(status, indent=2))
    else:
        # Default: show status
        status = grader.get_status()
        print("\n" + "=" * 60)
        print("PERFORMANCE GRADING STATUS")
        print("=" * 60)
        print(f"Promoted to Live: {status['promoted_to_live']}")
        print(f"Ready for Promotion: {status['ready_for_promotion']}")
        if status['last_grade']:
            print(f"\nLast Grade: {status['last_grade']['grade']}")
            print(f"Score: {status['last_grade']['score']:.1f}%")
            print(f"Status: {'✅ PASSED' if status['last_grade']['passed'] else '❌ FAILED'}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
