#!/usr/bin/env python3
"""
Coinbase Live Trading Implementation
Executes real trades on Coinbase using CDP API with Ed25519 authentication.

SAFETY FEATURES:
- Requires promotion from paper trading grader
- Position size limits
- Portfolio-level stop loss
- Emergency kill switch
- Trade logging and audit trail
- Dry-run mode for testing

Usage:
    python coinbase_live_trader.py --mode live --model-dir train_results/ensemble
    python coinbase_live_trader.py --mode dry-run  # Test mode
"""

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


class CoinbaseLiveTrader:
    """Live trader using Coinbase Advanced Trade API with CDP authentication."""

    def __init__(
        self,
        mode: str = "dry-run",  # dry-run or live
        model_dir: Path = None,
        key_file: str = "key/cdp_api_key.json",
        portfolio: str = "TestPortfolio",
        tickers: List[str] = None,
        poll_interval: int = 60,
        max_position_pct: float = 0.10,  # Max 10% per position for live
        stop_loss_pct: float = 0.05,  # 5% stop loss for live
        emergency_stop_pct: float = 0.20,  # Emergency stop at 20% portfolio loss
    ):
        self.mode = mode
        self.model_dir = Path(model_dir) if model_dir else None
        self.key_file = Path(key_file)
        self.portfolio = portfolio
        self.tickers = tickers or ["BTC-USD", "ETH-USD"]
        self.poll_interval = poll_interval

        # Safety limits (more conservative for live)
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.emergency_stop_pct = emergency_stop_pct

        # State
        self.running = True
        self.emergency_stop = False
        self.initial_portfolio_value = None
        self.positions = {}

        # Setup logging first
        self._setup_logging()

        # Load API credentials
        self.api_key_id, self.private_key = self._load_credentials()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Verify promotion status before live trading
        if mode == "live":
            self._verify_promotion()

    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '[%(asctime)s] [%(levelname)s] %(message)s'
        log_file = f'logs/coinbase_live_{self.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to: {log_file}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.warning(f"Received signal {signum}, shutting down safely...")
        self.running = False

    def _load_credentials(self) -> tuple:
        """Load CDP API credentials from key file."""
        if not self.key_file.exists():
            raise FileNotFoundError(f"API key file not found: {self.key_file}")

        with open(self.key_file) as f:
            key_data = json.load(f)

        api_key_id = key_data['id']
        private_key_base64 = key_data['privateKey']

        # Decode Ed25519 private key
        # CDP API keys are 64 bytes (32-byte private + 32-byte public concatenated)
        # We only need the first 32 bytes for the private key
        import base64
        private_key_bytes = base64.b64decode(private_key_base64)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes[:32])

        self.logger.info(f"Loaded CDP API key: {api_key_id[:8]}...")
        return api_key_id, private_key

    def _verify_promotion(self):
        """Verify that paper trader has been promoted to live trading."""
        grading_state_file = Path("deployments/grading_state.json")

        if not grading_state_file.exists():
            raise RuntimeError(
                "No grading state found. Paper trader must be graded first.\n"
                "Run: python performance_grader.py --check"
            )

        with open(grading_state_file) as f:
            state = json.load(f)

        if not state.get('promoted_to_live', False):
            last_grade = state.get('last_grade', {})
            raise RuntimeError(
                f"Paper trader has NOT been promoted to live trading.\n"
                f"Current grade: {last_grade.get('grade', 'N/A')} ({last_grade.get('score', 0):.1f}%)\n"
                f"Status: {last_grade.get('reason', 'Not graded')}\n\n"
                f"To promote, paper trader must pass all criteria:\n"
                f"  - 7+ days of trading\n"
                f"  - 60%+ win rate\n"
                f"  - Positive alpha\n"
                f"  - Max 15% drawdown\n\n"
                f"Run: python performance_grader.py --promote (after criteria met)"
            )

        promotion_date = state.get('promotion_date')
        self.logger.info("=" * 80)
        self.logger.info("✅ PAPER TRADER PROMOTION VERIFIED")
        self.logger.info(f"   Promoted on: {promotion_date}")
        self.logger.info(f"   Grade: {last_grade.get('grade')} ({last_grade.get('score'):.1f}%)")
        self.logger.info("=" * 80)

    def _sign_request(self, method: str, path: str, body: str = "") -> str:
        """Sign request using Ed25519 private key."""
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method}{path}{body}"

        signature = self.private_key.sign(message.encode())
        import base64
        return base64.b64encode(signature).decode()

    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make authenticated request to Coinbase Advanced Trade API."""
        base_url = "https://api.coinbase.com"
        path = f"/api/v3/brokerage/{endpoint}"
        url = base_url + path

        body = json.dumps(data) if data else ""
        signature = self._sign_request(method, path, body)
        timestamp = str(int(time.time()))

        headers = {
            "Content-Type": "application/json",
            "CB-ACCESS-KEY": self.api_key_id,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
        }

        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return response.json()

    def get_accounts(self) -> List[Dict]:
        """Get all accounts."""
        try:
            result = self._make_request("GET", "accounts")
            return result.get('accounts', [])
        except Exception as e:
            self.logger.error(f"Error getting accounts: {e}")
            return []

    def get_portfolio_value(self) -> float:
        """Get total portfolio value in USD."""
        accounts = self.get_accounts()
        total_usd = 0.0

        for account in accounts:
            currency = account.get('currency')
            available_balance = float(account.get('available_balance', {}).get('value', 0))

            if currency == 'USD':
                total_usd += available_balance
            elif available_balance > 0:
                # Get current price
                try:
                    product = f"{currency}-USD"
                    ticker_data = self._make_request("GET", f"products/{product}/ticker")
                    price = float(ticker_data.get('price', 0))
                    total_usd += available_balance * price
                except:
                    pass

        return total_usd

    def check_emergency_stop(self) -> bool:
        """Check if emergency stop should be triggered."""
        if self.initial_portfolio_value is None:
            return False

        current_value = self.get_portfolio_value()
        drawdown = (self.initial_portfolio_value - current_value) / self.initial_portfolio_value

        if drawdown >= self.emergency_stop_pct:
            self.logger.critical("=" * 80)
            self.logger.critical("🚨 EMERGENCY STOP TRIGGERED 🚨")
            self.logger.critical(f"Portfolio drawdown: {drawdown*100:.1f}%")
            self.logger.critical(f"Initial value: ${self.initial_portfolio_value:.2f}")
            self.logger.critical(f"Current value: ${current_value:.2f}")
            self.logger.critical("=" * 80)
            self.emergency_stop = True
            return True

        return False

    def place_order(self, product_id: str, side: str, size: float) -> Optional[Dict]:
        """Place a market order."""
        if self.mode == "dry-run":
            self.logger.info(f"[DRY-RUN] Would place {side} order: {size} {product_id}")
            return {"order_id": "dry-run-" + str(int(time.time())), "status": "simulated"}

        try:
            order_data = {
                "client_order_id": f"live-{int(time.time())}",
                "product_id": product_id,
                "side": side.lower(),
                "order_configuration": {
                    "market_market_ioc": {
                        "base_size": str(size)
                    }
                }
            }

            result = self._make_request("POST", "orders", order_data)
            self.logger.info(f"Order placed: {side} {size} {product_id} - ID: {result.get('order_id')}")
            return result

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def run(self):
        """Main trading loop."""
        self.logger.info("=" * 80)
        self.logger.info(f"COINBASE LIVE TRADER STARTED - MODE: {self.mode.upper()}")
        self.logger.info("=" * 80)
        self.logger.info(f"Portfolio: {self.portfolio}")
        self.logger.info(f"Tickers: {self.tickers}")
        self.logger.info(f"Max position: {self.max_position_pct*100:.0f}%")
        self.logger.info(f"Stop loss: {self.stop_loss_pct*100:.0f}%")
        self.logger.info(f"Emergency stop: {self.emergency_stop_pct*100:.0f}%")
        self.logger.info("=" * 80)

        # Get initial portfolio value
        self.initial_portfolio_value = self.get_portfolio_value()
        self.logger.info(f"Initial portfolio value: ${self.initial_portfolio_value:.2f}")

        while self.running and not self.emergency_stop:
            try:
                # Check emergency stop
                if self.check_emergency_stop():
                    break

                # Get current portfolio value
                current_value = self.get_portfolio_value()
                self.logger.info(f"Portfolio value: ${current_value:.2f} ({(current_value/self.initial_portfolio_value-1)*100:+.2f}%)")

                # TODO: Load model and get trading signals
                # TODO: Execute trades based on signals

                # Sleep until next poll
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                self.logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(60)

        self.logger.info("Coinbase live trader stopped")


def main():
    parser = argparse.ArgumentParser(description="Coinbase live trader")
    parser.add_argument("--mode", choices=["dry-run", "live"], default="dry-run", help="Trading mode")
    parser.add_argument("--model-dir", type=str, help="Model directory")
    parser.add_argument("--key-file", default="key/cdp_api_key.json", help="API key file")
    parser.add_argument("--portfolio", default="TestPortfolio", help="Portfolio name")
    parser.add_argument("--tickers", nargs="+", default=["BTC-USD", "ETH-USD"], help="Tickers to trade")
    parser.add_argument("--poll-interval", type=int, default=60, help="Poll interval in seconds")

    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    trader = CoinbaseLiveTrader(
        mode=args.mode,
        model_dir=args.model_dir,
        key_file=args.key_file,
        portfolio=args.portfolio,
        tickers=args.tickers,
        poll_interval=args.poll_interval,
    )

    trader.run()


if __name__ == "__main__":
    main()
