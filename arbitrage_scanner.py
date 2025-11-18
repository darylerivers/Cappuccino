#!/usr/bin/env python3
"""
Arbitrage Scanner for Cross-Currency Opportunities

Monitors USD pairs for triangular arbitrage opportunities across crypto markets.
Executes when profit exceeds transaction costs.

Example:
    BTC/USD = 95000
    ETH/USD = 3200
    Implied BTC/ETH = 95000/3200 = 29.69

    If actual BTC/ETH market = 29.50:
    → Arbitrage: Buy BTC/ETH (29.50), Sell BTC/USD (95000), Buy ETH/USD (3200)
    → Profit = 0.19 ETH per BTC (~0.64% before fees)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, CryptoLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame


class ArbitrageScanner:
    def __init__(
        self,
        usd_pairs: List[str],
        transaction_fee: float = 0.0025,
        min_profit_pct: float = 0.01,  # 1% minimum profit after fees
        log_file: Optional[Path] = None
    ):
        """
        Initialize arbitrage scanner.

        Args:
            usd_pairs: List of USD pairs to monitor (e.g., ['BTC/USD', 'ETH/USD'])
            transaction_fee: Transaction fee per trade (default 0.25%)
            min_profit_pct: Minimum profit percentage to execute (default 1%)
            log_file: Optional log file path
        """
        self.usd_pairs = usd_pairs
        self.transaction_fee = transaction_fee
        self.min_profit_pct = min_profit_pct
        self.log_file = log_file or Path("logs/arbitrage_scanner.log")

        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize Alpaca client (read-only for market data)
        self.client = CryptoHistoricalDataClient()

        # Track opportunities
        self.opportunities_found = 0
        self.last_scan_time = None

        self._log(f"Arbitrage scanner initialized with {len(usd_pairs)} USD pairs")
        self._log(f"Min profit threshold: {min_profit_pct:.2%}")
        self._log(f"Transaction fee per trade: {transaction_fee:.2%}")

    def _log(self, message: str, level: str = "INFO"):
        """Log message to file and optionally stdout."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"

        with self.log_file.open('a') as f:
            f.write(log_line + '\n')

        print(log_line)

    def get_latest_prices(self) -> Dict[str, float]:
        """Fetch latest prices for all USD pairs."""
        try:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=self.usd_pairs)
            quotes = self.client.get_crypto_latest_quote(request)

            prices = {}
            for symbol, quote in quotes.items():
                # Use mid-price for fairness
                prices[symbol] = (quote.ask_price + quote.bid_price) / 2

            return prices

        except Exception as e:
            self._log(f"Error fetching prices: {e}", "ERROR")
            return {}

    def get_cross_pair_price(self, base: str, quote: str) -> Optional[float]:
        """
        Fetch actual cross-pair market price if available.

        Args:
            base: Base currency (e.g., 'BTC')
            quote: Quote currency (e.g., 'ETH')

        Returns:
            Current market price or None if pair doesn't exist
        """
        symbol = f"{base}/{quote}"

        try:
            request = CryptoLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.client.get_crypto_latest_quote(request)

            if symbol in quotes:
                quote_data = quotes[symbol]
                return (quote_data.ask_price + quote_data.bid_price) / 2

        except Exception:
            # Pair doesn't exist or error fetching
            pass

        return None

    def calculate_triangular_arbitrage(
        self,
        prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Calculate all triangular arbitrage opportunities.

        For each pair of assets A and B:
        1. Calculate implied A/B rate from USD pairs
        2. Check actual A/B market price
        3. If spread > transaction costs → arbitrage opportunity

        Returns:
            List of arbitrage opportunities with profit estimates
        """
        opportunities = []

        # Extract asset names (remove '/USD')
        assets = [pair.replace('/USD', '') for pair in self.usd_pairs]

        # Check all pairs
        for i, asset_a in enumerate(assets):
            for j, asset_b in enumerate(assets):
                if i >= j:  # Skip duplicates and self-pairs
                    continue

                usd_pair_a = f"{asset_a}/USD"
                usd_pair_b = f"{asset_b}/USD"

                if usd_pair_a not in prices or usd_pair_b not in prices:
                    continue

                price_a_usd = prices[usd_pair_a]
                price_b_usd = prices[usd_pair_b]

                # Calculate implied cross rate
                implied_a_per_b = price_a_usd / price_b_usd  # How many A per B

                # Check actual market price
                actual_a_per_b = self.get_cross_pair_price(asset_a, asset_b)
                actual_b_per_a = self.get_cross_pair_price(asset_b, asset_a)

                # Strategy 1: A/B market cheaper than implied
                if actual_a_per_b is not None:
                    spread_pct = (implied_a_per_b - actual_a_per_b) / actual_a_per_b

                    # Account for 3 trades (3 * fee)
                    net_profit_pct = spread_pct - (3 * self.transaction_fee)

                    if net_profit_pct >= self.min_profit_pct:
                        opportunities.append({
                            'type': 'triangular',
                            'direction': f'{asset_b} → {asset_a} → USD → {asset_b}',
                            'legs': [
                                f'Buy {asset_a}/{asset_b} @ {actual_a_per_b:.6f}',
                                f'Sell {asset_a}/USD @ {price_a_usd:.2f}',
                                f'Buy {asset_b}/USD @ {price_b_usd:.2f}'
                            ],
                            'implied_rate': implied_a_per_b,
                            'actual_rate': actual_a_per_b,
                            'spread_pct': spread_pct * 100,
                            'net_profit_pct': net_profit_pct * 100,
                            'timestamp': datetime.now().isoformat()
                        })

                # Strategy 2: B/A market cheaper than implied
                if actual_b_per_a is not None:
                    implied_b_per_a = price_b_usd / price_a_usd
                    spread_pct = (implied_b_per_a - actual_b_per_a) / actual_b_per_a

                    net_profit_pct = spread_pct - (3 * self.transaction_fee)

                    if net_profit_pct >= self.min_profit_pct:
                        opportunities.append({
                            'type': 'triangular',
                            'direction': f'{asset_a} → {asset_b} → USD → {asset_a}',
                            'legs': [
                                f'Buy {asset_b}/{asset_a} @ {actual_b_per_a:.6f}',
                                f'Sell {asset_b}/USD @ {price_b_usd:.2f}',
                                f'Buy {asset_a}/USD @ {price_a_usd:.2f}'
                            ],
                            'implied_rate': implied_b_per_a,
                            'actual_rate': actual_b_per_a,
                            'spread_pct': spread_pct * 100,
                            'net_profit_pct': net_profit_pct * 100,
                            'timestamp': datetime.now().isoformat()
                        })

        return opportunities

    def scan(self) -> List[Dict]:
        """
        Perform a single arbitrage scan.

        Returns:
            List of arbitrage opportunities found
        """
        self._log("Starting arbitrage scan...")
        self.last_scan_time = datetime.now()

        # Fetch current prices
        prices = self.get_latest_prices()

        if not prices:
            self._log("No prices available, skipping scan", "WARNING")
            return []

        # Log current prices
        self._log(f"Current USD prices: {json.dumps({k: f'{v:.2f}' for k, v in prices.items()})}")

        # Calculate arbitrage opportunities
        opportunities = self.calculate_triangular_arbitrage(prices)

        if opportunities:
            self.opportunities_found += len(opportunities)
            self._log(f"Found {len(opportunities)} arbitrage opportunities!")

            for opp in opportunities:
                self._log(
                    f"  {opp['direction']} | "
                    f"Spread: {opp['spread_pct']:.2f}% | "
                    f"Net Profit: {opp['net_profit_pct']:.2f}%",
                    "OPPORTUNITY"
                )
                self._log(f"    Legs: {' → '.join(opp['legs'])}")
        else:
            self._log("No arbitrage opportunities found")

        return opportunities

    def run_continuous(self, interval_seconds: int = 60):
        """
        Run continuous arbitrage scanning.

        Args:
            interval_seconds: Time between scans
        """
        self._log(f"Starting continuous arbitrage monitoring (interval: {interval_seconds}s)")

        try:
            while True:
                opportunities = self.scan()

                # Save opportunities to file
                if opportunities:
                    output_file = Path("logs/arbitrage_opportunities.json")
                    with output_file.open('a') as f:
                        for opp in opportunities:
                            f.write(json.dumps(opp) + '\n')

                # Wait before next scan
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            self._log("Arbitrage scanner stopped by user")
        except Exception as e:
            self._log(f"Scanner error: {e}", "ERROR")
            raise


def main():
    """Run standalone arbitrage scanner."""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto Arbitrage Scanner")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Scan interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=0.01,
        help="Minimum profit percentage (default: 0.01 = 1%%)"
    )
    args = parser.parse_args()

    # Default tickers from main config
    from config_main import TICKER_LIST

    scanner = ArbitrageScanner(
        usd_pairs=TICKER_LIST[:7],  # Use first 7 tickers
        min_profit_pct=args.min_profit
    )

    scanner.run_continuous(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
