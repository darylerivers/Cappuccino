#!/usr/bin/env python3
"""
Tiburtina Integration for Cappuccino

Provides access to Tiburtina's AI analysis, market data, and research capabilities
from within the Cappuccino trading dashboard.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Add Tiburtina to path
TIBURTINA_PATH = Path("/home/mrc/experiment/tiburtina")
if TIBURTINA_PATH.exists():
    sys.path.insert(0, str(TIBURTINA_PATH))


class CachedData:
    """Simple cache with TTL (time-to-live)."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.data = None
        self.timestamp = None

    def is_stale(self) -> bool:
        """Check if cached data is stale."""
        if self.data is None or self.timestamp is None:
            return True
        age = time.time() - self.timestamp
        return age > self.ttl_seconds

    def set(self, data):
        """Cache data with current timestamp."""
        self.data = data
        self.timestamp = time.time()

    def get(self):
        """Get cached data if not stale, otherwise None."""
        if self.is_stale():
            return None
        return self.data


class TiburtinaClient:
    """Client for accessing Tiburtina functionality from Cappuccino."""

    def __init__(self):
        """Initialize Tiburtina client."""
        self.available = False
        self.analyst = None
        self.hub = None
        self.error = None

        # Caches with different TTLs
        self._macro_cache = CachedData(ttl_seconds=1800)  # 30 minutes
        self._crypto_cache = CachedData(ttl_seconds=300)  # 5 minutes
        self._news_cache = CachedData(ttl_seconds=600)    # 10 minutes

        self._initialize()

    def _initialize(self) -> None:
        """Try to initialize Tiburtina modules."""
        if not TIBURTINA_PATH.exists():
            self.error = f"Tiburtina not found at {TIBURTINA_PATH}"
            return

        try:
            # Import Tiburtina modules
            from src.ai.analyst import Analyst
            from src.ingestion import get_hub

            self.analyst = Analyst()
            self.hub = get_hub()
            self.available = True

        except ImportError as e:
            self.error = f"Failed to import Tiburtina modules: {e}"
        except Exception as e:
            self.error = f"Failed to initialize Tiburtina: {e}"

    def is_available(self) -> bool:
        """Check if Tiburtina is available."""
        return self.available

    def get_error(self) -> Optional[str]:
        """Get initialization error if any."""
        return self.error

    def get_market_brief(self) -> str:
        """Get AI-generated market brief."""
        if not self.available:
            return f"Tiburtina not available: {self.error}"

        try:
            return self.analyst.market_brief()
        except Exception as e:
            return f"Error generating market brief: {e}"

    def get_macro_snapshot(self, use_cache: bool = True) -> Dict:
        """Get macro economic snapshot (cached)."""
        if not self.available:
            return {"error": self.error}

        # Return cached data if available and not stale
        if use_cache:
            cached = self._macro_cache.get()
            if cached is not None:
                return cached

        try:
            # Try to get macro data with a timeout
            # Note: FRED API can be slow or have pandas compatibility issues
            data = self.hub.get_macro()

            # Format for display
            snapshot = {}
            priority = ["fed_funds", "treasury_10y", "unemployment", "cpi", "vix"]
            for key in priority:
                if key in data and "value" in data[key]:
                    snapshot[key] = {
                        "value": data[key]["value"],
                        "date": data[key].get("date", ""),
                    }

            result = snapshot if snapshot else {"error": "No macro data available"}

            # Cache successful results
            if "error" not in result:
                self._macro_cache.set(result)

            return result
        except KeyboardInterrupt:
            # Return cached data if available, even if stale
            cached = self._macro_cache.data
            if cached is not None:
                return cached
            return {"error": "Macro data fetch interrupted by user"}
        except Exception as e:
            # Return cached data if available, even if stale
            cached = self._macro_cache.data
            if cached is not None:
                return cached

            # Return a friendly error message
            error_msg = str(e)
            if "pandas" in error_msg.lower() or "datetime" in error_msg.lower():
                return {"error": "FRED API compatibility issue (pandas/datetime)"}
            return {"error": f"Failed to fetch macro data: {error_msg[:100]}"}

    def get_crypto_overview(self, use_cache: bool = True) -> List[Dict]:
        """Get top crypto overview (cached)."""
        if not self.available:
            return [{"error": self.error}]

        # Return cached data if available and not stale
        if use_cache:
            cached = self._crypto_cache.get()
            if cached is not None:
                return cached

        try:
            markets = self.hub.crypto.get_markets(per_page=10)
            result = markets if markets else [{"error": "No crypto data available"}]

            # Cache successful results
            if result and "error" not in result[0]:
                self._crypto_cache.set(result)

            return result
        except KeyboardInterrupt:
            # Return cached data if available, even if stale
            cached = self._crypto_cache.data
            if cached is not None:
                return cached
            return [{"error": "Crypto data fetch interrupted by user"}]
        except Exception as e:
            # Return cached data if available, even if stale
            cached = self._crypto_cache.data
            if cached is not None:
                return cached
            return [{"error": f"Failed to fetch crypto data: {str(e)[:100]}"}]

    def get_portfolio_analysis(self, tickers: List[str]) -> str:
        """Get AI analysis for a list of tickers."""
        if not self.available:
            return f"Tiburtina not available: {self.error}"

        try:
            # Get quotes for tickers
            context = {"quotes": {}}
            for ticker in tickers:
                try:
                    quote = self.hub.get_quote(ticker)
                    context["quotes"][ticker] = quote
                except:
                    pass

            # Add macro context
            try:
                context["macro"] = self.hub.get_macro()
            except:
                pass

            # Generate analysis
            query = f"Analyze the current market position for these assets: {', '.join(tickers)}"
            return self.analyst.analyze(query, context=context)
        except Exception as e:
            return f"Error analyzing portfolio: {e}"

    def get_news_summary(self, query: str = None, use_cache: bool = True) -> List[Dict]:
        """Get news headlines (cached)."""
        if not self.available:
            return [{"error": self.error}]

        # Return cached data if available and not stale (only for non-query requests)
        if use_cache and query is None:
            cached = self._news_cache.get()
            if cached is not None:
                return cached

        try:
            if query:
                articles = self.hub.news.search(query)
            else:
                articles = self.hub.news.get_rss_news()

            # Return top 10
            result = articles[:10] if articles else [{"error": "No news available"}]

            # Cache successful results (only for non-query requests)
            if query is None and result and "error" not in result[0]:
                self._news_cache.set(result)

            return result
        except KeyboardInterrupt:
            # Return cached data if available, even if stale
            cached = self._news_cache.data
            if cached is not None:
                return cached
            return [{"error": "News fetch interrupted by user"}]
        except Exception as e:
            # Return cached data if available, even if stale
            cached = self._news_cache.data
            if cached is not None:
                return cached
            return [{"error": f"Failed to fetch news: {str(e)[:100]}"}]

    def analyze_strategy(self, strategy_desc: str, market_conditions: Dict) -> str:
        """Get AI analysis of a trading strategy given market conditions."""
        if not self.available:
            return f"Tiburtina not available: {self.error}"

        try:
            query = f"""Analyze this trading strategy:
{strategy_desc}

Given current market conditions:
{market_conditions}

Provide:
1. Strategy fit for current conditions
2. Key risks
3. Potential improvements
"""
            return self.analyst.analyze(query)
        except Exception as e:
            return f"Error analyzing strategy: {e}"

    def prefetch_data(self):
        """Pre-fetch and cache all data in background (non-blocking on errors)."""
        if not self.available:
            return

        print("Prefetching Tiburtina data...")

        # Try to prefetch macro data
        try:
            self.get_macro_snapshot(use_cache=False)
            print("  ✓ Macro data cached")
        except:
            print("  ✗ Macro data failed")

        # Try to prefetch crypto data
        try:
            self.get_crypto_overview(use_cache=False)
            print("  ✓ Crypto data cached")
        except:
            print("  ✗ Crypto data failed")

        # Try to prefetch news
        try:
            self.get_news_summary(use_cache=False)
            print("  ✓ News data cached")
        except:
            print("  ✗ News data failed")

    def get_cache_status(self) -> Dict:
        """Get status of all caches."""
        return {
            "macro": {
                "has_data": self._macro_cache.data is not None,
                "is_stale": self._macro_cache.is_stale(),
                "age_seconds": time.time() - self._macro_cache.timestamp if self._macro_cache.timestamp else None,
            },
            "crypto": {
                "has_data": self._crypto_cache.data is not None,
                "is_stale": self._crypto_cache.is_stale(),
                "age_seconds": time.time() - self._crypto_cache.timestamp if self._crypto_cache.timestamp else None,
            },
            "news": {
                "has_data": self._news_cache.data is not None,
                "is_stale": self._news_cache.is_stale(),
                "age_seconds": time.time() - self._news_cache.timestamp if self._news_cache.timestamp else None,
            },
        }

    def get_asset_performance(self) -> Dict:
        """Get historical performance for major asset classes."""
        if not self.available:
            return {"error": self.error}

        try:
            performance = {}

            # Get crypto performance from cached data
            cryptos = self.get_crypto_overview(use_cache=True)
            if cryptos and isinstance(cryptos, list) and len(cryptos) > 0 and "error" not in cryptos[0]:
                # Average crypto performance (top 5)
                changes = [c.get("change_24h", 0) for c in cryptos[:5] if c.get("change_24h")]
                if changes:
                    performance["crypto_24h"] = sum(changes) / len(changes)

            # Get stock indices (if available via Tiburtina)
            try:
                # Try to get SPY (S&P 500 ETF) as proxy
                spy_quote = self.hub.get_quote("SPY")
                if spy_quote and "change_percent" in spy_quote:
                    performance["stocks_daily"] = spy_quote["change_percent"]
            except:
                pass

            # Get macro context
            macro = self.get_macro_snapshot(use_cache=True)
            if macro and "error" not in macro:
                performance["macro"] = macro

            return performance if performance else {"error": "No performance data available"}

        except Exception as e:
            return {"error": f"Failed to get asset performance: {str(e)[:100]}"}

    def get_market_correlations(self, tickers: List[str] = None) -> Dict:
        """Get correlation analysis between assets."""
        if not self.available:
            return {"error": self.error}

        # Default tickers for correlation analysis
        if tickers is None:
            tickers = ["BTC/USD", "ETH/USD", "SPY", "GLD", "TLT"]

        try:
            # This would require historical data - simplified for now
            # In full implementation, would fetch price history and calculate correlations
            return {
                "note": "Correlation analysis requires historical data",
                "tickers": tickers,
                "suggestion": "Use Tiburtina terminal for detailed correlation analysis"
            }
        except Exception as e:
            return {"error": f"Failed to get correlations: {str(e)[:100]}"}

    def get_market_analysis_detailed(self) -> str:
        """Get comprehensive AI-powered market analysis."""
        if not self.available:
            return f"Tiburtina not available: {self.error}"

        try:
            # Gather all available data
            macro = self.get_macro_snapshot(use_cache=True)
            crypto = self.get_crypto_overview(use_cache=True)
            performance = self.get_asset_performance()

            # Build comprehensive context
            context_parts = []

            if macro and "error" not in macro:
                context_parts.append(f"Macro indicators: {macro}")

            if crypto and isinstance(crypto, list) and "error" not in crypto[0]:
                top_crypto = crypto[:5]
                context_parts.append(f"Top crypto: {[c['symbol'] for c in top_crypto]}")
                perf_list = [f"{c['symbol']}:{c.get('change_24h', 0):+.1f}%" for c in top_crypto]
                context_parts.append(f"Crypto performance (24h): {perf_list}")

            if performance and "error" not in performance:
                context_parts.append(f"Asset performance: {performance}")

            # Generate AI analysis
            prompt = f"""Analyze the current market conditions and provide insights:

{chr(10).join(context_parts)}

Provide:
1. Overall market sentiment
2. Key themes and trends
3. Asset class positioning (which assets are strong/weak)
4. Risk factors to watch
5. Opportunities or areas of concern

Keep analysis concise (3-4 paragraphs max).
"""

            return self.analyst.analyze(prompt)

        except Exception as e:
            return f"Error generating market analysis: {e}"


# Singleton instance
_tiburtina_client = None


def get_tiburtina_client() -> TiburtinaClient:
    """Get singleton TiburtinaClient instance."""
    global _tiburtina_client
    if _tiburtina_client is None:
        _tiburtina_client = TiburtinaClient()
    return _tiburtina_client


if __name__ == "__main__":
    # Test the integration
    client = get_tiburtina_client()

    if client.is_available():
        print("✓ Tiburtina is available")

        # Prefetch data
        client.prefetch_data()

        print("\nCache Status:")
        status = client.get_cache_status()
        for source, info in status.items():
            age = f"{info['age_seconds']:.0f}s ago" if info['age_seconds'] else "never"
            stale = "STALE" if info['is_stale'] else "FRESH"
            print(f"  {source}: {stale} ({age})")

        print("\nMacro Snapshot (from cache):")
        macro = client.get_macro_snapshot()
        if "error" in macro:
            print(f"  Error: {macro['error']}")
        else:
            for key, val in macro.items():
                print(f"  {key}: {val['value']}")
    else:
        print(f"✗ Tiburtina not available: {client.get_error()}")
