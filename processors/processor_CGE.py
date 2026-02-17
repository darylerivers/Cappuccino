"""
CGE Synthetic Data Processor

Loads CGE-generated synthetic data for training augmentation.
Prioritizes bear market and crisis scenarios to improve model robustness.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Tuple


class CGEProcessor:
    """Load and process CGE-generated synthetic crypto data"""

    def __init__(self, cge_data_path: str = './data/cge_synthetic'):
        self.cge_data_path = Path(cge_data_path)
        self.assets = ['AAVE', 'AVAX', 'BTC', 'LINK', 'ETH', 'LTC', 'UNI']

    def load_scenario(self, scenario_id: int) -> Tuple[np.ndarray, dict]:
        """Load a single scenario by ID"""
        # Load price data
        price_file = self.cge_data_path / f'synthetic_{scenario_id:04d}.npy'
        prices = np.load(price_file)

        # Load metadata
        meta_file = self.cge_data_path / f'synthetic_meta_{scenario_id:04d}.json'
        with open(meta_file, 'r') as f:
            metadata = json.load(f)

        return prices, metadata

    def load_macro_scenarios(self) -> pd.DataFrame:
        """Load macro scenario metadata"""
        macro_file = self.cge_data_path / 'macro_scenarios.csv'
        return pd.read_csv(macro_file)

    def classify_scenario_by_macro(self, macro_row: pd.Series) -> str:
        """Classify economic scenario into regime"""
        stress = macro_row.get('financial_stress', 0)
        gdp = macro_row.get('GDP_Growth', 2.5)
        rates = macro_row.get('Interest_Rate', 2.5)
        risk = macro_row.get('Risk_Appetite', 50.0)

        if stress > 0.5:
            return 'crisis'
        elif risk > 70 and rates < 3.5:
            return 'bull'
        elif rates > 5.5 or gdp < 0:
            return 'bear'
        else:
            return 'normal'

    def get_scenarios_by_regime(self, regime: str = 'bear', max_count: int = None) -> List[int]:
        """Get scenario IDs filtered by regime"""
        macro_df = self.load_macro_scenarios()

        # Classify all scenarios
        macro_df['regime'] = macro_df.apply(self.classify_scenario_by_macro, axis=1)

        # Filter by regime
        filtered = macro_df[macro_df['regime'] == regime]
        scenario_ids = filtered.index.tolist()

        if max_count is not None:
            scenario_ids = scenario_ids[:max_count]

        return scenario_ids

    def load_multiple_scenarios(self, scenario_ids: List[int]) -> Tuple[np.ndarray, List[dict]]:
        """Load multiple scenarios and concatenate them"""
        all_prices = []
        all_metadata = []

        for sid in scenario_ids:
            try:
                prices, metadata = self.load_scenario(sid)

                # Normalize prices to realistic ranges
                # Reset each scenario to start from typical market prices
                prices_normalized = self._normalize_price_levels(prices)

                all_prices.append(prices_normalized)
                all_metadata.append(metadata)
            except FileNotFoundError:
                print(f"Warning: Scenario {sid} not found, skipping")
                continue

        # Stack all scenarios: (n_scenarios * timesteps, n_assets)
        combined_prices = np.vstack(all_prices)

        return combined_prices, all_metadata

    def _normalize_price_levels(self, prices: np.ndarray) -> np.ndarray:
        """
        Normalize price levels to realistic ranges

        Strategy: Rescale each asset to match typical market price ranges
        while preserving the pattern of price movements
        """
        n_timesteps, n_assets = prices.shape

        # Typical price ranges for each asset (approximate 2023-2024 levels)
        typical_ranges = {
            'AAVE': (100.0, 400.0),
            'AVAX': (20.0, 80.0),
            'BTC': (50000.0, 150000.0),
            'LINK': (5.0, 30.0),
            'ETH': (1500.0, 5000.0),
            'LTC': (50.0, 150.0),
            'UNI': (4.0, 15.0)
        }

        normalized = np.zeros_like(prices)

        for i, asset in enumerate(self.assets):
            asset_prices = prices[:, i]

            # Get current min/max
            p_min = asset_prices.min()
            p_max = asset_prices.max()

            # Target range
            target_min, target_max = typical_ranges.get(asset, (p_min, p_max))

            # Rescale to target range
            if p_max > p_min:  # Avoid division by zero
                normalized[:, i] = target_min + (asset_prices - p_min) * (target_max - target_min) / (p_max - p_min)
            else:
                normalized[:, i] = target_min

        return normalized

    def convert_to_cappuccino_format(self, prices: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert CGE price data to Cappuccino's expected format

        Returns:
            data_from_processor: DataFrame (placeholder)
            price_array: (timesteps, n_assets) close prices
            tech_array: (timesteps, n_assets * n_indicators) technical indicators
            time_array: (timesteps,) timestamps
        """
        n_timesteps, n_assets = prices.shape

        # Price array is just the close prices (timesteps, assets)
        price_array = prices

        # Compute technical indicators
        tech_array = self._compute_technical_indicators(prices)

        # Time array
        time_array = np.arange(n_timesteps)

        # DataFrame is not used in training, just placeholder
        df = pd.DataFrame()

        return df, price_array, tech_array, time_array

    def _compute_technical_indicators(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute basic technical indicators to match Cappuccino format

        Expected format: 14 indicators per asset × 7 assets = 98 features
        Indicators: open, high, low, close, volume, macd, macd_signal, macd_hist, rsi, cci, dx,
                   atr_regime_shift, range_breakout_volume, trend_reacceleration
        """
        n_timesteps, n_assets = prices.shape

        n_indicators = 14  # Match Cappuccino's TECHNICAL_INDICATORS_LIST
        tech_array = np.zeros((n_timesteps, n_assets * n_indicators))

        for i in range(n_assets):
            closes = prices[:, i]

            for t in range(n_timesteps):
                base_idx = i * n_indicators

                # OHLCV (synthetic from close)
                close = closes[t]
                open_price = closes[t-1] if t > 0 else close
                high = close * 1.005  # Synthetic
                low = close * 0.995
                volume = 1000000

                tech_array[t, base_idx + 0] = open_price
                tech_array[t, base_idx + 1] = high
                tech_array[t, base_idx + 2] = low
                tech_array[t, base_idx + 3] = close
                tech_array[t, base_idx + 4] = volume

                # MACD
                if t >= 26:
                    ema12 = closes[max(0,t-12):t+1].mean()
                    ema26 = closes[max(0,t-26):t+1].mean()
                    macd = ema12 - ema26
                    tech_array[t, base_idx + 5] = macd

                    if t >= 35:
                        macd_signal = tech_array[max(0,t-9):t+1, base_idx + 5].mean()
                        tech_array[t, base_idx + 6] = macd_signal
                        tech_array[t, base_idx + 7] = macd - macd_signal

                # RSI
                if t >= 14:
                    changes = np.diff(closes[max(0,t-14):t+1])
                    gains = changes[changes > 0].sum() if len(changes[changes > 0]) > 0 else 0
                    losses = -changes[changes < 0].sum() if len(changes[changes < 0]) > 0 else 0
                    if losses > 0:
                        rs = gains / losses
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 100 if gains > 0 else 50
                    tech_array[t, base_idx + 8] = rsi

                # CCI and DX - simplified placeholders
                tech_array[t, base_idx + 9] = 0   # CCI
                tech_array[t, base_idx + 10] = 0  # DX

                # NEW INDICATORS
                # ATR regime shift - placeholder (would need ATR calculation)
                tech_array[t, base_idx + 11] = 0  # atr_regime_shift

                # Range breakout + volume - placeholder
                tech_array[t, base_idx + 12] = 0  # range_breakout_volume

                # Trend strength re-acceleration - placeholder
                tech_array[t, base_idx + 13] = 0  # trend_reacceleration

        return tech_array

    def run(self,
            regime_filter: str = 'bear',
            n_scenarios: int = 60,
            **kwargs) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CGE scenarios with regime filtering

        Args:
            regime_filter: Which regime to load ('bear', 'crisis', 'normal', 'bull', or 'all')
            n_scenarios: How many scenarios to load

        Returns:
            Same format as other processors
        """
        print(f"\n{'='*70}")
        print(f"Loading CGE Synthetic Data")
        print(f"{'='*70}")
        print(f"Regime filter: {regime_filter}")
        print(f"Number of scenarios: {n_scenarios}")

        if regime_filter == 'all':
            scenario_ids = list(range(min(n_scenarios, 200)))
        else:
            scenario_ids = self.get_scenarios_by_regime(regime_filter, n_scenarios)

        print(f"Selected {len(scenario_ids)} scenarios: {scenario_ids[:10]}...")

        # Load scenarios
        combined_prices, metadata = self.load_multiple_scenarios(scenario_ids)

        print(f"Loaded {combined_prices.shape[0]} timesteps across {len(scenario_ids)} scenarios")

        # Convert to Cappuccino format
        df, price_array, tech_array, time_array = self.convert_to_cappuccino_format(combined_prices)

        print(f"✓ CGE data loaded successfully")
        print(f"  Price array shape: {price_array.shape}")
        print(f"  Tech array shape: {tech_array.shape}")

        return df, price_array, tech_array, time_array


def mix_data_sources(
    real_data: Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    synthetic_data: Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    real_ratio: float = 0.7
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mix real and synthetic data

    Args:
        real_data: (df, price_array, tech_array, time_array) from real source
        synthetic_data: (df, price_array, tech_array, time_array) from CGE
        real_ratio: Proportion of real data (0-1)

    Returns:
        Combined data in same format
    """
    print(f"\n{'='*70}")
    print(f"Mixing Data Sources")
    print(f"{'='*70}")
    print(f"Real data ratio: {real_ratio*100:.0f}%")
    print(f"Synthetic data ratio: {(1-real_ratio)*100:.0f}%")

    real_df, real_price, real_tech, real_time = real_data
    synth_df, synth_price, synth_tech, synth_time = synthetic_data

    # Calculate split points
    n_real = real_price.shape[0]
    n_synth = synth_price.shape[0]

    n_real_keep = int(n_real * real_ratio)
    n_synth_keep = int(n_real * (1 - real_ratio))

    print(f"Real data: {n_real} timesteps → keeping {n_real_keep}")
    print(f"Synthetic data: {n_synth} timesteps → keeping {n_synth_keep}")

    # Take subsets
    real_price_subset = real_price[:n_real_keep]
    real_tech_subset = real_tech[:n_real_keep]

    synth_price_subset = synth_price[:n_synth_keep]
    synth_tech_subset = synth_tech[:n_synth_keep]

    # Concatenate
    combined_price = np.vstack([real_price_subset, synth_price_subset])
    combined_tech = np.vstack([real_tech_subset, synth_tech_subset])
    combined_time = np.arange(combined_price.shape[0])

    # DataFrame is harder to mix, just return real_df for now
    # (not used in training, only for visualization)
    combined_df = real_df

    print(f"✓ Combined data shape: {combined_price.shape}")
    print(f"{'='*70}\n")

    return combined_df, combined_price, combined_tech, combined_time
