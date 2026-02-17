#!/usr/bin/env python3
"""
Coinbase Live Trader
Real money trading on Coinbase Advanced (formerly Coinbase Pro)

Features:
- Maker order optimization (target 70%+ maker ratio)
- Fee tier tracking and volume reporting
- Risk management (daily limits, position limits)
- Discord notifications
- Automatic model loading from ensemble
- Conviction scoring (optional)

Usage:
    python live_trader_coinbase.py \
        --model-dir train_results/best_5min_model \
        --tickers BTC-USD ETH-USD \
        --timeframe 5m \
        --initial-capital 1000 \
        --poll-interval 300

IMPORTANT: Test thoroughly on paper/sandbox before live deployment!
"""

import argparse
import csv
import json
import os
import pickle
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# Coinbase API
try:
    from coinbase.rest import RESTClient
    COINBASE_AVAILABLE = True
except ImportError:
    print("ERROR: coinbase-advanced-py not installed")
    print("Run: pip install coinbase-advanced-py")
    COINBASE_AVAILABLE = False
    sys.exit(1)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from constants import RISK, TRADING, DISCORD
from integrations.discord_notifier import DiscordNotifier
from utils.conviction_scorer import ConvictionFilter


@dataclass
class CoinbaseFeeStatus:
    """Track Coinbase fee tier and volume."""
    current_tier: str
    maker_fee_rate: float
    taker_fee_rate: float
    monthly_volume: float
    next_tier: Optional[str]
    volume_to_next_tier: float


@dataclass
class TradeExecution:
    """Record of a trade execution."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'maker' or 'taker'
    fee: float
    total_value: float


class CoinbaseLiveTrader:
    """Live trading on Coinbase Advanced with maker optimization."""

    def __init__(
        self,
        *,
        tickers: List[str],
        timeframe: str,
        model_dir: Path,
        initial_capital: float,
        poll_interval: int,
        gpu_id: int,
        log_file: Path,
        api_key: str,
        api_secret: str,
        risk_management: Optional[Dict] = None,
        maker_preference: float = 0.7,  # Target 70% maker orders
        max_daily_loss_pct: float = 0.02,  # Stop if down 2% in a day
        max_consecutive_losses: int = 3,
    ):
        self.tickers = tickers
        self.timeframe = timeframe
        self.model_dir = Path(model_dir)
        self.initial_capital = initial_capital
        self.poll_interval = poll_interval
        self.gpu_id = gpu_id
        self.log_file = Path(log_file)
        self.maker_preference = maker_preference
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses

        # Initialize Coinbase client
        self.client = RESTClient(
            api_key=api_key,
            api_secret=api_secret
        )

        # Discord
        self.discord = None
        if DISCORD.ENABLED:
            try:
                self.discord = DiscordNotifier()
            except:
                pass

        # Trading state
        self.running = True
        self.start_of_day_capital = initial_capital
        self.current_capital = initial_capital
        self.consecutive_losses = 0
        self.trades_today = []
        self.maker_count = 0
        self.taker_count = 0

        # Fee tier tracking
        self.fee_status = self._get_current_fee_tier()

        # Load model
        self._load_model()

        # Conviction scoring
        self.conviction_filter = ConvictionFilter(min_score=0.6)

        # Portfolio state
        self.positions = {ticker: 0.0 for ticker in tickers}
        self.cash = initial_capital

        # Setup logging
        self._setup_logging()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown gracefully."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False

    def _setup_logging(self):
        """Setup CSV logging for trades."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create CSV header if new file
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'quantity', 'price',
                    'order_type', 'fee', 'total_value', 'portfolio_value',
                    'daily_pnl_pct', 'maker_ratio'
                ])

    def _load_model(self):
        """Load trained model."""
        print(f"Loading model from {self.model_dir}")

        # Load actor model
        actor_path = self.model_dir / "actor.pth"
        if not actor_path.exists():
            raise FileNotFoundError(f"Model not found: {actor_path}")

        # Setup device
        if self.gpu_id >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu_id}')
        else:
            self.device = torch.device('cpu')

        # Load checkpoint to determine architecture
        checkpoint = torch.load(actor_path, map_location='cpu', weights_only=True)

        # Extract dimensions from checkpoint
        if 'net.4.weight' in checkpoint:
            # Standard actor architecture
            num_assets = checkpoint['net.4.weight'].shape[0]
            state_dim = checkpoint['net.0.weight'].shape[1]
        else:
            raise ValueError("Unexpected checkpoint structure")

        # Import actor class
        try:
            from elegantrl.agents import AgentPPO
            agent = AgentPPO()
            agent.act.net.to(self.device)
            agent.act.net.load_state_dict(checkpoint)
            agent.act.net.eval()
            self.act = agent.act.net
        except ImportError:
            print("⚠️  elegantrl not installed, using direct torch network")
            # Create simple actor network manually if needed
            import torch.nn as nn
            self.act = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, num_assets),
                nn.Tanh()
            )
            self.act.load_state_dict(checkpoint)
            self.act.to(self.device)
            self.act.eval()

        self.state_dim = state_dim
        self.num_assets = num_assets

        print(f"✓ Model loaded successfully")
        print(f"  State dim: {state_dim}, Assets: {num_assets}")
        print(f"  Device: {self.device}")

    def _get_current_fee_tier(self) -> CoinbaseFeeStatus:
        """Get current fee tier from Coinbase."""
        try:
            # Get account info (includes fee tier)
            accounts = self.client.get_accounts()

            # Get 30-day volume (placeholder - implement actual API call)
            monthly_volume = 0.0  # TODO: Get from Coinbase API

            # Fee tier logic (Coinbase Advanced fees as of 2024)
            fee_tiers = [
                ("Intro 1", 0.006, 0.012, 0),
                ("Intro 2", 0.004, 0.008, 10_000),
                ("Advanced 1", 0.0025, 0.005, 25_000),
                ("Advanced 2", 0.00125, 0.0025, 75_000),
                ("Advanced 3", 0.00075, 0.0015, 250_000),
                ("VIP 1", 0.0006, 0.00125, 500_000),
                ("VIP 2", 0.0005, 0.001, 1_000_000),
                ("VIP 3", 0.0004, 0.00085, 5_000_000),
            ]

            # Determine current tier
            current_tier = None
            next_tier = None
            for i, (name, maker, taker, min_vol) in enumerate(fee_tiers):
                if monthly_volume >= min_vol:
                    current_tier = (name, maker, taker)
                    if i + 1 < len(fee_tiers):
                        next_tier = fee_tiers[i + 1]

            if current_tier:
                name, maker, taker = current_tier
                if next_tier:
                    next_name, _, _, next_vol = next_tier
                    volume_to_next = next_vol - monthly_volume
                else:
                    next_name = None
                    volume_to_next = 0

                return CoinbaseFeeStatus(
                    current_tier=name,
                    maker_fee_rate=maker,
                    taker_fee_rate=taker,
                    monthly_volume=monthly_volume,
                    next_tier=next_name,
                    volume_to_next_tier=volume_to_next
                )

        except Exception as e:
            print(f"⚠️  Could not get fee tier: {e}")

        # Default to Intro 1
        return CoinbaseFeeStatus(
            current_tier="Intro 1",
            maker_fee_rate=0.006,
            taker_fee_rate=0.012,
            monthly_volume=0,
            next_tier="Intro 2",
            volume_to_next_tier=10_000
        )

    def check_daily_limits(self) -> bool:
        """Check if daily trading limits exceeded."""
        # Daily loss limit
        daily_pnl_pct = (self.current_capital - self.start_of_day_capital) / self.start_of_day_capital

        if daily_pnl_pct < -self.max_daily_loss_pct:
            print(f"🛑 Daily loss limit hit: {daily_pnl_pct*100:.2f}% (limit: {-self.max_daily_loss_pct*100:.1f}%)")
            if self.discord:
                self.discord.send_message(
                    content="🛑 **TRADING HALTED: Daily Loss Limit**",
                    embed={
                        "title": "Risk Management Alert",
                        "color": 0xff0000,
                        "fields": [
                            {"name": "Daily P&L", "value": f"{daily_pnl_pct*100:+.2f}%", "inline": True},
                            {"name": "Limit", "value": f"{-self.max_daily_loss_pct*100:.1f}%", "inline": True},
                        ]
                    }
                )
            return False

        # Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            print(f"🛑 Consecutive loss limit hit: {self.consecutive_losses} losses")
            if self.discord:
                self.discord.send_message(
                    content="🛑 **TRADING HALTED: Consecutive Losses**",
                    embed={
                        "title": "Risk Management Alert",
                        "color": 0xff0000,
                        "fields": [
                            {"name": "Consecutive Losses", "value": str(self.consecutive_losses), "inline": True},
                            {"name": "Limit", "value": str(self.max_consecutive_losses), "inline": True},
                        ]
                    }
                )
            return False

        return True

    def execute_trade_maker_optimized(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        conviction: float = 1.0
    ) -> Optional[TradeExecution]:
        """Execute trade with maker order preference."""

        # Adjust quantity based on conviction
        adjusted_quantity = quantity * conviction

        if adjusted_quantity < 0.001:  # Minimum order size
            return None

        try:
            # Try maker order first (70% of the time based on preference)
            use_maker = np.random.random() < self.maker_preference

            if use_maker:
                # Limit order strategy
                if side == 'buy':
                    # Place slightly below market
                    limit_price = current_price * 0.9995  # 0.05% below
                else:
                    # Place slightly above market
                    limit_price = current_price * 1.0005  # 0.05% above

                # Create limit order
                order = self.client.create_limit_order(
                    product_id=symbol,
                    side=side,
                    limit_price=str(limit_price),
                    base_size=str(adjusted_quantity)
                )

                # Wait for fill (up to 2 minutes)
                filled = self._wait_for_fill(order['order_id'], timeout=120)

                if filled:
                    # Success - maker order filled
                    fill_price = float(order['filled_price'])  # Actual fill price
                    fee = fill_price * adjusted_quantity * self.fee_status.maker_fee_rate

                    self.maker_count += 1

                    return TradeExecution(
                        timestamp=datetime.now(timezone.utc),
                        symbol=symbol,
                        side=side,
                        quantity=adjusted_quantity,
                        price=fill_price,
                        order_type='maker',
                        fee=fee,
                        total_value=fill_price * adjusted_quantity
                    )
                else:
                    # Not filled - cancel and use taker
                    self.client.cancel_order(order['order_id'])
                    use_maker = False

            if not use_maker:
                # Market order (taker)
                order = self.client.create_market_order(
                    product_id=symbol,
                    side=side,
                    base_size=str(adjusted_quantity)
                )

                fill_price = float(order['filled_price'])
                fee = fill_price * adjusted_quantity * self.fee_status.taker_fee_rate

                self.taker_count += 1

                return TradeExecution(
                    timestamp=datetime.now(timezone.utc),
                    symbol=symbol,
                    side=side,
                    quantity=adjusted_quantity,
                    price=fill_price,
                    order_type='taker',
                    fee=fee,
                    total_value=fill_price * adjusted_quantity
                )

        except Exception as e:
            print(f"⚠️  Trade execution failed: {e}")
            return None

    def _wait_for_fill(self, order_id: str, timeout: int = 120) -> bool:
        """Wait for limit order to fill."""
        start = time.time()

        while time.time() - start < timeout:
            try:
                order = self.client.get_order(order_id)

                if order['status'] == 'FILLED':
                    return True
                elif order['status'] in ['CANCELLED', 'EXPIRED', 'FAILED']:
                    return False

                time.sleep(2)  # Check every 2 seconds

            except Exception as e:
                print(f"⚠️  Order status check failed: {e}")
                return False

        return False

    def log_trade(self, trade: TradeExecution):
        """Log trade to CSV."""
        maker_ratio = self.maker_count / (self.maker_count + self.taker_count) if (self.maker_count + self.taker_count) > 0 else 0
        daily_pnl_pct = (self.current_capital - self.start_of_day_capital) / self.start_of_day_capital * 100

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.timestamp.isoformat(),
                trade.symbol,
                trade.side,
                f"{trade.quantity:.8f}",
                f"{trade.price:.2f}",
                trade.order_type,
                f"{trade.fee:.4f}",
                f"{trade.total_value:.2f}",
                f"{self.current_capital:.2f}",
                f"{daily_pnl_pct:+.2f}",
                f"{maker_ratio:.2%}"
            ])

    def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch recent market data from Coinbase."""
        market_data = {}

        # Map timeframe to Coinbase granularity
        granularity_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
        }
        granularity = granularity_map.get(self.timeframe, 300)

        # Fetch last 100 candles for each ticker
        for ticker in self.tickers:
            try:
                candles = self.client.get_candles(
                    product_id=ticker,
                    granularity=granularity,
                    limit=100
                )

                # Convert to DataFrame
                df = pd.DataFrame(candles['candles'])
                df['timestamp'] = pd.to_datetime(df['start'], unit='s')
                df = df.sort_values('timestamp')

                market_data[ticker] = df

            except Exception as e:
                print(f"⚠️  Failed to fetch data for {ticker}: {e}")

        return market_data

    def construct_state(
        self,
        market_data: Dict[str, pd.DataFrame],
        positions: Dict[str, float],
        cash: float
    ) -> np.ndarray:
        """
        Construct state vector for model.

        Similar to paper trader state construction.
        """
        # This is a simplified version - adapt to your actual state space
        features = []

        # Portfolio features
        total_value = cash
        for ticker, quantity in positions.items():
            if ticker in market_data and len(market_data[ticker]) > 0:
                current_price = float(market_data[ticker].iloc[-1]['close'])
                total_value += quantity * current_price

        # Normalize by initial capital
        features.append(total_value / self.initial_capital)
        features.append(cash / total_value if total_value > 0 else 0)

        # Asset features (prices, returns, positions)
        for ticker in self.tickers:
            if ticker not in market_data or len(market_data[ticker]) == 0:
                # Missing data - use zeros
                features.extend([0.0] * 10)
                continue

            df = market_data[ticker]
            current_price = float(df.iloc[-1]['close'])

            # Price features
            features.append(current_price / 10000.0)  # Normalize

            # Returns (last 5 periods)
            prices = df['close'].values[-6:]
            returns = np.diff(np.log(prices)) if len(prices) > 1 else [0.0]
            returns = list(returns[-5:])
            returns.extend([0.0] * (5 - len(returns)))  # Pad if needed
            features.extend(returns)

            # Position weight
            position_value = positions.get(ticker, 0) * current_price
            position_weight = position_value / total_value if total_value > 0 else 0
            features.append(position_weight)

            # Technical indicators (simple moving averages)
            if len(df) >= 20:
                ma_short = df['close'].rolling(5).mean().iloc[-1]
                ma_long = df['close'].rolling(20).mean().iloc[-1]
                features.append((current_price - ma_short) / current_price)
                features.append((current_price - ma_long) / current_price)
            else:
                features.extend([0.0, 0.0])

        state = np.array(features, dtype=np.float32)

        # Pad or truncate to expected state_dim
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]

        return state

    def get_model_prediction(self, state: np.ndarray) -> np.ndarray:
        """Get model prediction (actions)."""
        state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_tensor = self.act(state_tensor)

        action = action_tensor.cpu().numpy()[0]
        return action

    def run(self):
        """Main trading loop."""
        print("="*80)
        print("COINBASE LIVE TRADER STARTED")
        print("="*80)
        print(f"Tickers: {self.tickers}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Poll interval: {self.poll_interval}s")
        print(f"\nFee Tier: {self.fee_status.current_tier}")
        print(f"  Maker: {self.fee_status.maker_fee_rate*100:.3f}%")
        print(f"  Taker: {self.fee_status.taker_fee_rate*100:.3f}%")
        print(f"  Monthly volume: ${self.fee_status.monthly_volume:,.2f}")
        if self.fee_status.next_tier:
            print(f"  Next tier: {self.fee_status.next_tier} (${self.fee_status.volume_to_next_tier:,.0f} more volume)")
        print("="*80)
        print("\n⚠️  WARNING: LIVE TRADING WITH REAL MONEY! ⚠️\n")
        print("Press Ctrl+C to stop.\n")

        # Send startup notification
        if self.discord:
            self.discord.send_message(
                content="🚀 **Coinbase Live Trader Started**",
                embed={
                    "title": "Live Trading Session",
                    "color": 0x00ff00,
                    "fields": [
                        {"name": "Capital", "value": f"${self.initial_capital:,.2f}", "inline": True},
                        {"name": "Fee Tier", "value": self.fee_status.current_tier, "inline": True},
                        {"name": "Timeframe", "value": self.timeframe, "inline": True},
                    ]
                }
            )

        iteration = 0
        while self.running:
            try:
                iteration += 1
                print(f"\n[Iteration #{iteration}] {datetime.now(timezone.utc).isoformat()}")

                # Check daily limits
                if not self.check_daily_limits():
                    print("Trading halted due to risk limits")
                    break

                # 1. Fetch market data
                print("  Fetching market data...")
                market_data = self.get_market_data()

                if not market_data:
                    print("  ⚠️  No market data available, skipping")
                    time.sleep(self.poll_interval)
                    continue

                # 2. Construct state
                state = self.construct_state(market_data, self.positions, self.cash)

                # 3. Get model prediction
                raw_actions = self.get_model_prediction(state)

                # Map actions to tickers
                raw_action_dict = {
                    ticker: float(raw_actions[i])
                    for i, ticker in enumerate(self.tickers)
                }

                print(f"  Raw actions: {raw_action_dict}")

                # 4. Calculate portfolio value
                portfolio_value = self.cash
                for ticker, quantity in self.positions.items():
                    if ticker in market_data and len(market_data[ticker]) > 0:
                        current_price = float(market_data[ticker].iloc[-1]['close'])
                        portfolio_value += quantity * current_price

                # Build state dict for conviction scorer
                prices = {
                    ticker: float(market_data[ticker].iloc[-1]['close'])
                    for ticker in self.tickers
                    if ticker in market_data and len(market_data[ticker]) > 0
                }

                state_dict = {
                    'positions': self.positions.copy(),
                    'cash': self.cash,
                    'portfolio_value': portfolio_value,
                    'prices': prices,
                    'price_history': [float(x) for x in market_data[self.tickers[0]]['close'].values[-50:]]
                    if self.tickers[0] in market_data else []
                }

                # 5. Filter by conviction
                filtered_actions = self.conviction_filter.filter_actions(
                    raw_action_dict,
                    state_dict
                )

                print(f"  Filtered actions: {filtered_actions}")

                # 6. Execute trades
                trades_executed = 0
                for ticker, action in filtered_actions.items():
                    if ticker not in prices:
                        continue

                    current_price = prices[ticker]
                    current_position = self.positions.get(ticker, 0)

                    # Calculate target position (action is -1 to +1, scale to portfolio weight)
                    max_position_value = portfolio_value * 0.3  # Max 30% per asset
                    target_position_value = max_position_value * abs(action)
                    target_position = target_position_value / current_price

                    # Determine trade
                    if action > 0.1:  # Buy signal
                        # Buy up to target
                        quantity_to_buy = target_position - current_position

                        if quantity_to_buy > 0 and self.cash > quantity_to_buy * current_price * 1.01:
                            trade = self.execute_trade_maker_optimized(
                                symbol=ticker,
                                side='buy',
                                quantity=quantity_to_buy,
                                current_price=current_price,
                                conviction=abs(action)
                            )

                            if trade:
                                self.positions[ticker] = self.positions.get(ticker, 0) + trade.quantity
                                self.cash -= trade.total_value + trade.fee
                                self.log_trade(trade)
                                trades_executed += 1

                                print(f"  ✓ BUY {trade.quantity:.6f} {ticker} @ ${trade.price:.2f}")

                    elif action < -0.1:  # Sell signal
                        # Sell down to target (or all if strong sell)
                        quantity_to_sell = current_position - target_position

                        if quantity_to_sell > 0:
                            trade = self.execute_trade_maker_optimized(
                                symbol=ticker,
                                side='sell',
                                quantity=min(quantity_to_sell, current_position),
                                current_price=current_price,
                                conviction=abs(action)
                            )

                            if trade:
                                self.positions[ticker] = self.positions.get(ticker, 0) - trade.quantity
                                self.cash += trade.total_value - trade.fee
                                self.log_trade(trade)
                                trades_executed += 1

                                print(f"  ✓ SELL {trade.quantity:.6f} {ticker} @ ${trade.price:.2f}")

                # 7. Update portfolio state
                self.current_capital = self.cash
                for ticker, quantity in self.positions.items():
                    if ticker in prices:
                        self.current_capital += quantity * prices[ticker]

                # Update conviction scorer performance
                if iteration > 1:
                    portfolio_return = (self.current_capital / portfolio_value - 1.0)
                    self.conviction_filter.update_performance(portfolio_return)

                # Log status
                print(f"  Portfolio: ${self.current_capital:,.2f} ({(self.current_capital/self.initial_capital-1)*100:+.2f}%)")
                print(f"  Cash: ${self.cash:,.2f} ({self.cash/self.current_capital*100:.1f}%)")
                print(f"  Trades executed: {trades_executed}")
                print(f"  Maker ratio: {self.maker_count/(self.maker_count + self.taker_count)*100:.1f}%" if self.maker_count + self.taker_count > 0 else "")

                # Sleep until next poll
                print(f"  Sleeping {self.poll_interval}s until next poll...")
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                print("\nShutting down...")
                break

            except Exception as e:
                print(f"Error in trading loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)

        print("Live trader stopped")


def main():
    parser = argparse.ArgumentParser(description="Coinbase live trader")
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--tickers', nargs='+', default=['BTC-USD', 'ETH-USD'])
    parser.add_argument('--timeframe', type=str, default='5m')
    parser.add_argument('--initial-capital', type=float, default=1000)
    parser.add_argument('--poll-interval', type=int, default=300)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--log-file', type=Path, default=Path('live_trades/coinbase_session.csv'))

    args = parser.parse_args()

    # Get API credentials
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')

    if not api_key or not api_secret:
        print("ERROR: COINBASE_API_KEY and COINBASE_API_SECRET environment variables required")
        sys.exit(1)

    trader = CoinbaseLiveTrader(
        tickers=args.tickers,
        timeframe=args.timeframe,
        model_dir=args.model_dir,
        initial_capital=args.initial_capital,
        poll_interval=args.poll_interval,
        gpu_id=args.gpu,
        log_file=args.log_file,
        api_key=api_key,
        api_secret=api_secret
    )

    trader.run()


if __name__ == '__main__':
    main()
