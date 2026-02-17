"""
Tiburtina Integration Bridge for Cappuccino

Provides access to Tiburtina's market data, news, macro indicators, and AI analysis.
"""
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3
from datetime import datetime

# Add Tiburtina to path
TIBURTINA_PATH = Path("/home/mrc/experiment/tiburtina")
if str(TIBURTINA_PATH) not in sys.path:
    sys.path.insert(0, str(TIBURTINA_PATH))

logger = logging.getLogger(__name__)


class TiburtinaBridge:
    """Bridge between Cappuccino and Tiburtina systems"""

    def __init__(self):
        """Initialize connection to Tiburtina"""
        self.tiburtina_available = False
        self.hub = None
        self._analyst = None
        self._summarizer = None
        self._analyst_class = None
        self._summarizer_class = None
        self.db = None
        self.alpaca_api = None

        # Caching for slow API calls
        self._macro_cache = None
        self._macro_cache_time = 0
        self._macro_cache_ttl = 300  # 5 minutes (macro data doesn't change quickly)

        try:
            from src.ingestion import get_hub
            from src.ai.analyst import Analyst
            from src.ai.summarizer import Summarizer

            self.hub = get_hub()
            # Store classes for lazy loading (don't initialize yet - they load Ollama models which is slow!)
            self._analyst_class = Analyst
            self._summarizer_class = Summarizer
            self.db = sqlite3.connect(str(TIBURTINA_PATH / "db/tiburtina.db"))

            self.tiburtina_available = True
            logger.info("✅ Tiburtina integration initialized successfully (AI models will lazy-load)")

        except Exception as e:
            logger.error(f"⚠️  Tiburtina integration failed: {e}")
            logger.warning("Continuing without Tiburtina features...")

        # Initialize Alpaca News API (primary)
        try:
            import os
            import alpaca_trade_api as tradeapi

            api_key = os.getenv("ALPACA_API_KEY", "")
            api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY", "")

            if api_key and api_secret:
                self.alpaca_api = tradeapi.REST(
                    api_key, api_secret,
                    "https://paper-api.alpaca.markets",
                    api_version='v2'
                )
                logger.info("✅ Alpaca News API initialized")
            else:
                logger.warning("⚠️  Alpaca credentials not found - news API unavailable")
        except Exception as e:
            logger.warning(f"⚠️  Alpaca News API initialization failed: {e}")

        # Initialize alternative news APIs
        self.finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        self.newsapi_key = os.getenv("NEWSAPI_KEY", "")

        if self.finnhub_key:
            logger.info("✅ Finnhub API key found")
        if self.newsapi_key:
            logger.info("✅ NewsAPI.org key found")

    @property
    def analyst(self):
        """Lazy-load Analyst (loads Ollama models on first access)"""
        if self._analyst is None and self._analyst_class is not None:
            logger.info("Loading Analyst AI models (this may take a moment)...")
            self._analyst = self._analyst_class()
        return self._analyst

    @property
    def summarizer(self):
        """Lazy-load Summarizer (loads Ollama models on first access)"""
        if self._summarizer is None and self._summarizer_class is not None:
            logger.info("Loading Summarizer AI models (this may take a moment)...")
            self._summarizer = self._summarizer_class()
        return self._summarizer

    def is_available(self) -> bool:
        """Check if Tiburtina is available"""
        return self.tiburtina_available

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote for symbol"""
        if not self.tiburtina_available:
            return None

        try:
            return self.hub.get_quote(symbol)
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_macro(self) -> Optional[Dict]:
        """Get current macro indicators (VIX, Fed Funds, yields, etc.) - cached for 5 minutes"""
        if not self.tiburtina_available:
            return None

        # Check cache first
        import time
        current_time = time.time()
        if self._macro_cache is not None and (current_time - self._macro_cache_time) < self._macro_cache_ttl:
            return self._macro_cache

        # Cache miss - fetch fresh data
        try:
            macro_data = self.hub.get_macro()
            # Update cache
            self._macro_cache = macro_data
            self._macro_cache_time = current_time
            return macro_data
        except Exception as e:
            logger.error(f"Failed to get macro indicators: {e}")
            # Return stale cache if available rather than None
            return self._macro_cache

    def get_macro_regime(self) -> str:
        """
        Detect current macro regime

        Returns:
            'high_volatility', 'low_volatility', 'rising_rates', or 'normal'
        """
        macro = self.get_macro()
        if not macro:
            return "normal"

        try:
            vix = macro.get('vix', {}).get('value', 15)
            fed_funds = macro.get('fed_funds', {}).get('value', 3.0)

            if vix > 25:
                return "high_volatility"
            elif vix < 15:
                return "low_volatility"
            elif fed_funds > 4.5:
                return "rising_rates"
            else:
                return "normal"

        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return "normal"

    def get_position_size_multiplier(self, base_size: float) -> tuple[float, str]:
        """
        Get position size adjustment based on macro conditions

        Returns:
            (adjusted_size, reason)
        """
        if not self.tiburtina_available:
            return (base_size, "tiburtina_unavailable")

        regime = self.get_macro_regime()
        macro = self.get_macro()

        if not macro:
            return (base_size, "no_macro_data")

        try:
            vix = macro.get('vix', {}).get('value', 15)
            fed_funds = macro.get('fed_funds', {}).get('value', 3.0)

            # High volatility - reduce size
            if vix > 30:
                return (base_size * 0.4, f"high_vix_{vix:.1f}")
            elif vix > 25:
                return (base_size * 0.5, f"elevated_vix_{vix:.1f}")
            elif vix > 20:
                return (base_size * 0.75, f"moderate_vix_{vix:.1f}")

            # Very low volatility - slightly increase
            elif vix < 12:
                return (base_size * 1.1, f"low_vix_{vix:.1f}")

            # Rising rates - reduce leverage
            if fed_funds > 5.0:
                return (base_size * 0.7, f"high_rates_{fed_funds:.2f}")
            elif fed_funds > 4.5:
                return (base_size * 0.8, f"elevated_rates_{fed_funds:.2f}")

            return (base_size, "normal_conditions")

        except Exception as e:
            logger.error(f"Error calculating position multiplier: {e}")
            return (base_size, "calculation_error")

    def get_news(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """
        Get recent news, optionally filtered by symbol

        Args:
            symbol: Optional symbol to filter news
            limit: Maximum number of articles
        """
        if not self.tiburtina_available:
            return []

        try:
            if symbol:
                return self.hub.news.search(symbol)[:limit]
            else:
                return self.hub.get_news()[:limit]
        except Exception as e:
            logger.error(f"Failed to get news: {e}")
            return []

    def check_news_alerts(self, symbols: List[str]) -> List[Dict]:
        """
        Check for important news on symbols

        Returns list of alerts with symbol, title, source, url
        """
        if not self.tiburtina_available:
            return []

        alerts = []

        # Keywords that indicate important news
        alert_keywords = [
            'earnings', 'downgrade', 'upgrade', 'fda', 'merger',
            'acquisition', 'ceo', 'lawsuit', 'recall', 'investigation',
            'bankruptcy', 'dividend', 'split', 'buyback'
        ]

        try:
            for symbol in symbols:
                news = self.get_news(symbol, limit=10)

                for article in news:
                    title_lower = article.get('title', '').lower()

                    if any(keyword in title_lower for keyword in alert_keywords):
                        alerts.append({
                            'symbol': symbol,
                            'title': article.get('title'),
                            'source': article.get('source'),
                            'url': article.get('url'),
                            'published': article.get('published'),
                            'detected_keywords': [kw for kw in alert_keywords if kw in title_lower]
                        })

        except Exception as e:
            logger.error(f"Error checking news alerts: {e}")

        return alerts

    def get_alpaca_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get recent news from Alpaca News API for a symbol

        Args:
            symbol: Trading symbol (e.g., 'BTC/USD', 'AAPL')
            limit: Maximum number of articles

        Returns:
            List of news articles with headline, summary, created_at, source, symbols
        """
        if not self.alpaca_api:
            return []

        try:
            # Clean symbol for Alpaca (remove / for crypto)
            clean_symbol = symbol.replace('/USD', '').replace('/', '')

            # Alpaca news API
            news = self.alpaca_api.get_news(clean_symbol, limit=limit)

            articles = []
            for article in news:
                articles.append({
                    'headline': article.headline,
                    'summary': article.summary if hasattr(article, 'summary') else '',
                    'created_at': article.created_at.isoformat() if hasattr(article, 'created_at') else '',
                    'source': article.source if hasattr(article, 'source') else 'Unknown',
                    'url': article.url if hasattr(article, 'url') else '',
                    'symbols': article.symbols if hasattr(article, 'symbols') else [clean_symbol]
                })

            return articles

        except Exception as e:
            logger.error(f"Failed to get Alpaca news for {symbol}: {e}")
            return []

    def get_finnhub_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get news from Finnhub API (sentiment + comprehensive coverage)

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'AAPL')
            limit: Maximum number of articles

        Returns:
            List of news with headline, summary, sentiment, source
        """
        if not self.finnhub_key:
            return []

        try:
            import requests
            from datetime import datetime, timedelta

            # Clean symbol
            clean_symbol = symbol.replace('/USD', '').replace('/', '').upper()

            # Finnhub news endpoint
            # For crypto: use general news and filter
            # For stocks: use company news
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': clean_symbol,
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            articles = []
            for article in response.json()[:limit]:
                articles.append({
                    'headline': article.get('headline', ''),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', 'Finnhub'),
                    'url': article.get('url', ''),
                    'datetime': article.get('datetime', 0),
                    'sentiment': article.get('sentiment', 'neutral')  # Finnhub provides sentiment
                })

            return articles

        except Exception as e:
            logger.error(f"Failed to get Finnhub news for {symbol}: {e}")
            return []

    def get_newsapi_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """
        Get general market news from NewsAPI.org

        Args:
            symbol: Trading symbol (used as keyword)
            limit: Maximum number of articles

        Returns:
            List of news with headline, description, source
        """
        if not self.newsapi_key:
            return []

        try:
            import requests

            # Clean symbol for search
            clean_symbol = symbol.replace('/USD', '').replace('/', '')

            # NewsAPI everything endpoint
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{clean_symbol} OR crypto OR cryptocurrency" if 'BTC' in symbol or 'ETH' in symbol else clean_symbol,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'apiKey': self.newsapi_key
            }

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            articles = []
            for article in response.json().get('articles', [])[:limit]:
                articles.append({
                    'headline': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'source': article.get('source', {}).get('name', 'NewsAPI'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', '')
                })

            return articles

        except Exception as e:
            logger.error(f"Failed to get NewsAPI news for {symbol}: {e}")
            return []

    def get_aggregated_news(self, symbol: str, limit: int = 20) -> List[Dict]:
        """
        Aggregate news from all available sources

        Returns combined and deduplicated news from:
        - Alpaca News API (primary)
        - Finnhub (if key available)
        - NewsAPI (if key available)
        - Tiburtina RSS (backup)

        Returns:
            Combined list of news articles, sorted by recency
        """
        all_news = []

        # 1. Alpaca News (primary - trading specific)
        alpaca_news = self.get_alpaca_news(symbol, limit=limit // 2)
        all_news.extend([{**n, 'source_api': 'alpaca'} for n in alpaca_news])

        # 2. Finnhub (sentiment + broad coverage)
        if self.finnhub_key:
            finnhub_news = self.get_finnhub_news(symbol, limit=limit // 3)
            all_news.extend([{**n, 'source_api': 'finnhub'} for n in finnhub_news])

        # 3. NewsAPI (general market news)
        if self.newsapi_key:
            newsapi_news = self.get_newsapi_news(symbol, limit=limit // 3)
            all_news.extend([{**n, 'source_api': 'newsapi'} for n in newsapi_news])

        # 4. Tiburtina RSS (backup)
        if self.tiburtina_available:
            rss_news = self.get_news(symbol, limit=limit // 4)
            all_news.extend([{**n, 'source_api': 'rss'} for n in rss_news])

        # Simple deduplication by headline similarity
        seen_headlines = set()
        unique_news = []

        for article in all_news:
            headline = article.get('headline') or article.get('title', '')
            # Use first 50 chars as fingerprint
            fingerprint = headline[:50].lower().strip()

            if fingerprint and fingerprint not in seen_headlines:
                seen_headlines.add(fingerprint)
                unique_news.append(article)

        return unique_news[:limit]

    def check_pre_trade_news(self, symbol: str) -> Dict:
        """
        Comprehensive pre-trade news check combining Alpaca + RSS

        Returns:
            {
                'has_news': bool,
                'bearish_signals': List[str],
                'bullish_signals': List[str],
                'recommendation': str,  # 'proceed', 'reduce', 'skip'
                'reason': str
            }
        """
        result = {
            'has_news': False,
            'bearish_signals': [],
            'bullish_signals': [],
            'recommendation': 'proceed',
            'reason': 'no_significant_news'
        }

        # Bearish keywords (high priority)
        bearish_keywords = [
            'downgrade', 'cut', 'lower', 'warning', 'concern', 'worry',
            'lawsuit', 'investigation', 'probe', 'fraud', 'scandal',
            'bankruptcy', 'default', 'debt', 'loss', 'miss', 'misses',
            'decline', 'fall', 'drop', 'plunge', 'crash', 'sell',
            'bearish', 'negative', 'weak', 'disappointing'
        ]

        # Bullish keywords
        bullish_keywords = [
            'upgrade', 'raise', 'beat', 'exceed', 'strong', 'growth',
            'acquisition', 'buyout', 'partnership', 'deal', 'launch',
            'bullish', 'positive', 'surge', 'rally', 'gain'
        ]

        try:
            # Get aggregated news from all available sources
            # This includes: Alpaca, Finnhub, NewsAPI, and RSS
            all_news = self.get_aggregated_news(symbol, limit=15)

            if len(all_news) == 0:
                return result

            result['has_news'] = True

            # Analyze each article
            for article in all_news[:10]:  # Check up to 10 most recent
                # Get text to analyze
                text = ''
                if 'headline' in article:
                    text = article['headline'].lower()
                elif 'title' in article:
                    text = article.get('title', '').lower()

                if 'summary' in article and article['summary']:
                    text += ' ' + article['summary'].lower()

                # Check for bearish signals
                for keyword in bearish_keywords:
                    if keyword in text:
                        result['bearish_signals'].append(f"{keyword}: {article.get('headline') or article.get('title', 'N/A')[:50]}")

                # Check for bullish signals
                for keyword in bullish_keywords:
                    if keyword in text:
                        result['bullish_signals'].append(f"{keyword}: {article.get('headline') or article.get('title', 'N/A')[:50]}")

            # Make recommendation
            bearish_count = len(result['bearish_signals'])
            bullish_count = len(result['bullish_signals'])

            if bearish_count >= 2:
                result['recommendation'] = 'skip'
                result['reason'] = f"{bearish_count} bearish signals detected"
            elif bearish_count == 1 and bullish_count == 0:
                result['recommendation'] = 'reduce'
                result['reason'] = "1 bearish signal, no bullish counterbalance"
            elif bullish_count >= 2:
                result['recommendation'] = 'proceed'
                result['reason'] = f"{bullish_count} bullish signals"
            elif bearish_count == 0 and bullish_count == 0:
                result['recommendation'] = 'proceed'
                result['reason'] = "neutral news"

        except Exception as e:
            logger.error(f"Error in pre-trade news check: {e}")
            # Default to proceed on error (don't block trading on news API issues)
            result['recommendation'] = 'proceed'
            result['reason'] = f"news_check_error: {str(e)[:50]}"

        return result

    def analyze_trade(self, symbol: str, action: str, reasoning: str, timeout: int = 10) -> Optional[str]:
        """
        Get AI analysis of proposed trade

        Args:
            symbol: Trading symbol
            action: 'LONG' or 'SHORT'
            reasoning: Why the ML model suggested this trade
            timeout: Max seconds to wait for AI (default 10)

        Returns:
            AI analysis text or None if unavailable/timeout
        """
        if not self.tiburtina_available:
            return None

        try:
            quote = self.get_quote(symbol)
            macro = self.get_macro()
            news = self.get_news(symbol, limit=5)

            context = {
                'quotes': {symbol: quote} if quote else {},
                'macro': macro or {},
                'recent_news': [n.get('title') for n in news[:3]]
            }

            query = f"""
            Proposed trade: {action} {symbol}
            ML Model reasoning: {reasoning}

            Current price: ${quote.get('price', 'N/A') if quote else 'N/A'}
            Recent news: {', '.join(context['recent_news']) if context['recent_news'] else 'None'}

            Given current market context, is this a good trade?
            Consider risks and provide a brief recommendation (2-3 sentences max).
            """

            # AI analysis (note: may be slow, 5-15 seconds)
            analysis = self.analyst.analyze(query, context=context)
            return analysis

        except Exception as e:
            logger.error(f"AI trade analysis failed: {e}")
            return None

    def get_market_context(self, symbols: List[str]) -> Dict:
        """
        Get comprehensive market context for symbols

        Returns dict with quotes, macro, news
        """
        if not self.tiburtina_available:
            return {'quotes': {}, 'macro': {}, 'news': []}

        try:
            return {
                'quotes': {s: self.get_quote(s) for s in symbols},
                'macro': self.get_macro(),
                'news': self.get_news(limit=20)
            }
        except Exception as e:
            logger.error(f"Failed to get market context: {e}")
            return {'quotes': {}, 'macro': {}, 'news': []}

    def summarize_news(self, symbol: str) -> Optional[str]:
        """Get AI summary of recent news for symbol"""
        if not self.tiburtina_available:
            return None

        try:
            news = self.get_news(symbol, limit=10)
            if not news:
                return None

            summary = self.summarizer.summarize_news(news, focus=symbol)
            return summary

        except Exception as e:
            logger.error(f"Failed to summarize news for {symbol}: {e}")
            return None

    def get_status(self) -> Dict:
        """Get Tiburtina integration status for monitoring"""
        status = {
            'available': self.tiburtina_available,
            'timestamp': datetime.now().isoformat(),
        }

        if self.tiburtina_available:
            try:
                # Test basic connectivity
                macro = self.get_macro()
                status['macro_working'] = macro is not None
                status['vix'] = macro.get('vix', {}).get('value') if macro else None
                status['regime'] = self.get_macro_regime()

            except Exception as e:
                status['error'] = str(e)

        return status


# Singleton instance
_bridge = None


def get_tiburtina_bridge() -> TiburtinaBridge:
    """Get or create Tiburtina bridge singleton"""
    global _bridge
    if _bridge is None:
        _bridge = TiburtinaBridge()
    return _bridge


def check_tiburtina_status() -> Dict:
    """Convenience function to check integration status"""
    bridge = get_tiburtina_bridge()
    return bridge.get_status()
