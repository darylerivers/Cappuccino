"""
Momentum Live Trading Engine
Runs on the hour, computes 24h cross-sectional momentum ranks,
rebalances weekly (when due), applies BTC 200h MA regime filter.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import uuid

import ccxt
import numpy as np
import pandas as pd
import yaml

import audit as _audit
import alerts as _alerts

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CFG_FILE    = BASE_DIR / "config.yaml"
VERSION_FILE = BASE_DIR / "version.json"


# ── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CFG_FILE) as f:
        cfg = yaml.safe_load(f)
    # Inject strategy version into config so alerts can access it
    if VERSION_FILE.exists():
        with open(VERSION_FILE) as f:
            v = json.load(f)
        cfg["strategy_version"] = v.get("version", "unknown")
    else:
        cfg["strategy_version"] = "unknown"
    return cfg


# ── Structured JSON logger ────────────────────────────────────────────────────
def make_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"{today}.jsonl"

    logger = logging.getLogger("momentum")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    logger.addHandler(sh)

    return logger


def log_event(logger: logging.Logger, event: str, **kwargs):
    record = dict(
        ts=datetime.now(timezone.utc).isoformat(),
        event=event,
        **kwargs,
    )
    logger.debug(json.dumps(record))
    logger.info(f"[{event}] " + "  ".join(f"{k}={v}" for k, v in kwargs.items()))


# ── State persistence ─────────────────────────────────────────────────────────
def load_state(state_file: Path, cfg: dict) -> dict:
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    # Fresh state — seed paper portfolio from config capital
    return dict(
        positions={},           # asset -> qty held
        paper_usdt=float(cfg["capital_usdt"]),  # cash in paper portfolio
        portfolio_value=None,
        peak_value=None,
        last_rebal_ts=None,
        consecutive_errors=0,
        circuit_breaker=None,
        inception_value=None,
        rebal_count=0,
        pnl_history=[],
    )


def save_state(state: dict, state_file: Path):
    tmp = state_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.rename(tmp, state_file)


# ── Coinbase Advanced Trade adapter ───────────────────────────────────────────
class CoinbaseExchange:
    """
    Thin adapter around coinbase-advanced-py that presents the same interface
    as ccxt for the operations the engine uses:
        fetch_ohlcv / fetch_balance / create_market_order
    Coinbase pairs use '-' separator and USD quote (BTC-USD, ETH-USD, ...).
    """

    GRANULARITY = "ONE_HOUR"

    def __init__(self, api_key: str, api_secret: str, portfolio_uuid: str = ""):
        from coinbase.rest import RESTClient
        self._client       = RESTClient(api_key=api_key, api_secret=api_secret)
        self._portfolio_id = portfolio_uuid

    @staticmethod
    def _to_cb_symbol(ccxt_symbol: str) -> str:
        """'BTC/USD' → 'BTC-USD'"""
        return ccxt_symbol.replace("/", "-")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 210) -> list:
        """Returns list of [timestamp_ms, o, h, l, close, volume] — ccxt format."""
        import math
        cb_sym  = self._to_cb_symbol(symbol)
        now_ts  = int(time.time())
        # Coinbase max 300 candles per request; 1h = 3600s
        start   = now_ts - (limit + 5) * 3600
        resp    = self._client.get_candles(
            product_id=cb_sym,
            start=str(start),
            end=str(now_ts),
            granularity=self.GRANULARITY,
        )
        # SDK returns a response object; access .candles attribute directly
        candles = resp.candles if hasattr(resp, "candles") else resp.get("candles", [])
        # Coinbase returns newest-first; reverse to oldest-first
        candles = list(reversed(candles))
        result  = []
        for c in candles[-limit:]:
            ts_ms = int(c["start"]) * 1000
            result.append([
                ts_ms,
                float(c["open"]),
                float(c["high"]),
                float(c["low"]),
                float(c["close"]),
                float(c["volume"]),
            ])
        return result

    def fetch_balance(self) -> dict:
        """Returns balance dict in ccxt format: {asset: {free, used, total}}.
        Uses portfolio breakdown when portfolio_uuid is set (reads correct portfolio).
        Falls back to get_accounts() for default portfolio.
        """
        bal: dict = {}

        if self._portfolio_id:
            resp = self._client.get_portfolio_breakdown(
                portfolio_uuid=self._portfolio_id
            )
            # SDK returns typed object; PortfolioPosition has direct attributes
            breakdown = resp.breakdown if hasattr(resp, "breakdown") else resp
            spot_positions = getattr(breakdown, "spot_positions", [])
            for pos in spot_positions:
                asset = getattr(pos, "asset", "")
                if not asset:
                    continue
                free       = float(getattr(pos, "available_to_trade_crypto", 0) or 0)
                total      = float(getattr(pos, "total_balance_crypto", 0) or 0)
                used       = max(total - free, 0.0)
                total_fiat = float(getattr(pos, "total_balance_fiat", 0) or 0)
                # USDC and USD both represent cash — merge additively so
                # futures-settled USD is not invisible to the engine
                if asset in ("USDC", "USD"):
                    existing = bal.get("USD", {"free": 0.0, "used": 0.0, "total": 0.0, "total_fiat": 0.0})
                    merged = {
                        "free":       existing["free"]              + free,
                        "used":       existing["used"]              + used,
                        "total":      existing["total"]             + total,
                        "total_fiat": existing.get("total_fiat", 0) + total_fiat,
                    }
                    bal["USD"]  = merged
                    bal["USDC"] = merged
                else:
                    bal[asset] = {"free": free, "used": used, "total": total, "total_fiat": total_fiat}
        else:
            accounts = self._client.get_accounts()
            acct_list = getattr(accounts, "accounts", None) or accounts.get("accounts", [])
            for acct in acct_list:
                cur  = getattr(acct, "currency", "") or acct.get("currency", "")
                avb  = getattr(acct, "available_balance", None)
                hld  = getattr(acct, "hold", None)
                free = float(getattr(avb, "value", None) or (avb or {}).get("value", 0))
                hold = float(getattr(hld, "value", None) or (hld or {}).get("value", 0))
                bal[cur] = {"free": free, "used": hold, "total": free + hold}

        # Always ensure USD key exists
        if "USD" not in bal:
            bal["USD"] = {"free": 0.0, "used": 0.0, "total": 0.0}
        # Expose USD as USDT alias so engine cash logic works unchanged
        bal["USDT"] = bal.get("USD", {"free": 0.0, "used": 0.0, "total": 0.0})
        return bal

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """
        Places a market order.
        Buys use quote_size (USD amount); sells use base_size (coin amount).
        Returns dict with 'average' price and 'id'.
        """
        cb_sym   = self._to_cb_symbol(symbol)
        order_id = str(uuid.uuid4())

        if side == "buy":
            # quote_size = USD to spend (amount is coin qty; convert to USD notional)
            # We pass qty*px but we don't have px here — use quote_size from caller
            # The engine passes qty (coin amount); for Coinbase buys we need USD notional.
            # The trade dict has px; we reconstruct here via a best-bid lookup.
            best   = self._client.get_best_bid_ask(product_ids=[cb_sym])
            px     = float(best["pricebooks"][0]["asks"][0]["price"])
            quote  = str(round(amount * px, 2))
            resp   = self._client.market_order_buy(
                client_order_id=order_id,
                product_id=cb_sym,
                quote_size=quote,
            )
        else:
            resp = self._client.market_order_sell(
                client_order_id=order_id,
                product_id=cb_sym,
                base_size=str(round(amount, 8)),
            )

        sr = resp.success_response if hasattr(resp, "success_response") else resp
        return {
            "id":      getattr(sr, "order_id", order_id) if hasattr(sr, "order_id") else order_id,
            "average": None,   # filled async; engine falls back to mid-price
            "price":   None,
        }


# ── Exchange setup ─────────────────────────────────────────────────────────────
def make_exchange(cfg: dict):
    """Returns a CoinbaseExchange or a ccxt exchange depending on config."""
    exchange = cfg["exchange"].lower()

    if exchange == "coinbase":
        import json
        key_path = Path(cfg.get("cdp_key_file", "")).expanduser()
        if key_path.exists():
            with open(key_path) as f:
                key = json.load(f)
            api_key    = key["name"]
            api_secret = key["privateKey"]
        else:
            api_key    = cfg.get("api_key", "")
            api_secret = cfg.get("api_secret", "")
        portfolio_uuid = cfg.get("portfolio_uuid", "")
        return CoinbaseExchange(api_key, api_secret, portfolio_uuid=portfolio_uuid)

    # Fallback: ccxt (Kraken etc.)
    params: dict = {
        "apiKey":          cfg.get("api_key", ""),
        "secret":          cfg.get("api_secret", ""),
        "options":         {"defaultType": "spot"},
        "enableRateLimit": True,
    }
    if cfg.get("api_passphrase"):
        params["password"] = cfg["api_passphrase"]
    ex_cls = getattr(ccxt, exchange)
    return ex_cls(params)


# ── Market data ───────────────────────────────────────────────────────────────
def fetch_ohlcv_data(ex: ccxt.Exchange, symbol: str, limit: int = 210) -> tuple:
    """Fetch last `limit` 1h bars. Returns (close_series, volume_series) indexed by datetime (UTC)."""
    ohlcv = ex.fetch_ohlcv(symbol, "1h", limit=limit)
    df    = pd.DataFrame(ohlcv, columns=["ts", "o", "h", "l", "close", "vol"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")
    return df["close"], df["vol"]


def fetch_all_ohlcv(
    ex, assets: list[str], limit: int = 210
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (closes_df, volumes_df) for all assets."""
    quote   = "USD" if isinstance(ex, CoinbaseExchange) else "USDT"
    closes  = {}
    volumes = {}
    for asset in assets:
        symbol = f"{asset}/{quote}"
        c, v   = fetch_ohlcv_data(ex, symbol, limit=limit)
        closes[asset]  = c
        volumes[asset] = v
        time.sleep(0.25)
    idx = pd.DataFrame(closes).dropna().index
    return pd.DataFrame(closes).loc[idx], pd.DataFrame(volumes).loc[idx]


# ── Signal & weights ──────────────────────────────────────────────────────────
def compute_target_weights(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    cfg: dict,
    last_regime: bool = True,
) -> tuple:
    """
    Composite signal: equal-weight z-score of ret_24h + vol_accel.
      ret_24h  : 24h price return (momentum)
      vol_accel: current 24h volume / prior 24h volume (volume confirmation)
    Returns (weights, in_regime, btc_px, btc_ma, scores_dict).
    """
    lb   = cfg["lookback_bars"]
    ma_n = cfg["regime_ma_bars"]
    top  = cfg["top_n"]
    mw   = cfg["max_position_pct"]

    # Regime check with ±0.5% hysteresis
    btc_ma = closes["BTC"].rolling(ma_n).mean().iloc[-1]
    btc_px = closes["BTC"].iloc[-1]
    buffer = cfg.get("regime_buffer", 0.005)
    if last_regime:
        in_regime = btc_px >= btc_ma * (1 - buffer)
    else:
        in_regime = btc_px >= btc_ma * (1 + buffer)

    if not in_regime:
        return {a: 0.0 for a in closes.columns}, False, btc_px, btc_ma, {}

    # Factor 1: 24h return
    ret_24h = closes.pct_change(lb).iloc[-1]

    # Factor 2: volume acceleration — current 24h vol / prior 24h vol
    vol_now  = volumes.rolling(lb).sum().iloc[-1]
    vol_prev = volumes.rolling(lb).sum().shift(lb).iloc[-1]
    vol_accel = (vol_now / vol_prev.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # Z-score each factor cross-sectionally then average
    def _zscore(s):
        s = s.dropna()
        std = s.std()
        return (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)

    z_ret   = _zscore(ret_24h)
    z_vaccel = _zscore(vol_accel)
    composite = (z_ret.add(z_vaccel, fill_value=0)) / 2.0

    ranked  = composite.rank(ascending=False, na_option="bottom")
    weights = {a: (mw if ranked.get(a, 999) <= top else 0.0) for a in closes.columns}
    scores  = composite.to_dict()
    return weights, True, btc_px, btc_ma, scores


# ── Portfolio valuation ───────────────────────────────────────────────────────
def get_portfolio_value(
    ex: ccxt.Exchange,
    assets: list[str],
    last_prices: dict[str, float],
    state: dict,
    paper_mode: bool,
) -> tuple[float, dict]:
    """Returns (total_usdt, holdings={asset: qty}).
    Paper mode: reads from state (no exchange call).
    Live mode:  fetches real balance from exchange.
    """
    if paper_mode:
        usdt     = float(state.get("paper_usdt", 0))
        holdings = {a: float(state.get("positions", {}).get(a, 0)) for a in assets}
        total    = usdt + sum(holdings[a] * last_prices.get(a, 0) for a in assets)
        return total, holdings

    balance  = ex.fetch_balance()
    cash_key = "USD" if isinstance(ex, CoinbaseExchange) else "USDT"
    holdings = {}

    # Prefer Coinbase-reported fiat values (match what the app shows) over
    # crypto_qty × stale OHLCV price.
    if isinstance(ex, CoinbaseExchange) and any(
        "total_fiat" in balance.get(a, {}) for a in [cash_key] + assets
    ):
        total = sum(
            float(balance.get(k, {}).get("total_fiat", 0))
            for k in [cash_key] + assets
        )
        for asset in assets:
            holdings[asset] = float(balance.get(asset, {}).get("total", 0))
    else:
        usdt  = float(balance.get(cash_key, {}).get("free", 0))
        total = usdt
        for asset in assets:
            qty = float(balance.get(asset, {}).get("total", 0))
            holdings[asset] = qty
            total += qty * last_prices.get(asset, 0)

    return total, holdings


# ── Rebalance trades ──────────────────────────────────────────────────────────
def compute_rebal_trades(
    current_holdings: dict[str, float],
    target_weights: dict[str, float],
    portfolio_value: float,
    last_prices: dict[str, float],
    fee_rate: float,
) -> list[dict]:
    """
    Returns list of trades: {asset, side, qty, reason}.
    Sells first, then buys (to free USDT before entering new positions).
    """
    trades = []

    for asset, target_w in target_weights.items():
        px          = last_prices[asset]
        target_usdt = portfolio_value * target_w
        target_qty  = target_usdt / px if px > 0 else 0
        current_qty = current_holdings.get(asset, 0)

        delta_qty  = target_qty - current_qty
        delta_usdt = abs(delta_qty) * px

        # Skip tiny trades (< $1 or < 0.1% of portfolio)
        if delta_usdt < max(1.0, portfolio_value * 0.001):
            continue

        side = "buy" if delta_qty > 0 else "sell"
        trades.append(dict(
            asset=asset,
            side=side,
            qty=abs(delta_qty),
            delta_usdt=delta_usdt,
            px=px,
        ))

    # Sell first
    trades.sort(key=lambda t: 0 if t["side"] == "sell" else 1)
    return trades


# ── Order execution ───────────────────────────────────────────────────────────
def execute_trades(
    ex,
    trades: list[dict],
    paper_mode: bool,
    state: dict,
    logger: logging.Logger,
) -> list[dict]:
    quote  = "USD" if isinstance(ex, CoinbaseExchange) else "USDT"
    fills = []
    for t in trades:
        symbol = f"{t['asset']}/{quote}"
        try:
            if paper_mode:
                slippage = 0.0005 if t["side"] == "buy" else -0.0005
                fill_px  = t["px"] * (1 + slippage)
                notional = fill_px * t["qty"]
                fee      = notional * (state.get("fee_rate", 0.0026))

                # Update paper portfolio
                if t["side"] == "buy":
                    state["paper_usdt"]  = state.get("paper_usdt", 0) - notional - fee
                    state["positions"][t["asset"]] = (
                        state.get("positions", {}).get(t["asset"], 0) + t["qty"]
                    )
                else:
                    state["paper_usdt"]  = state.get("paper_usdt", 0) + notional - fee
                    state["positions"][t["asset"]] = max(
                        0.0,
                        state.get("positions", {}).get(t["asset"], 0) - t["qty"]
                    )

                fill = dict(
                    asset=t["asset"], side=t["side"], qty=round(t["qty"], 8),
                    fill_px=round(fill_px, 6), mid_px=round(t["px"], 6),
                    notional_usdt=round(notional, 4),
                    fee_usdt=round(fee, 4),
                    slippage_bps=round(slippage * 10000, 2),
                    paper=True,
                )
            else:
                order   = ex.create_market_order(symbol, t["side"], t["qty"])
                fill_px = float(order.get("average") or order.get("price") or t["px"])
                slip    = (fill_px - t["px"]) / t["px"] * 10000 * (1 if t["side"] == "buy" else -1)
                fill = dict(
                    asset=t["asset"], side=t["side"], qty=t["qty"],
                    fill_px=fill_px, mid_px=t["px"],
                    slippage_bps=round(slip, 2),
                    order_id=order.get("id"), paper=False,
                )

            fills.append(fill)
            log_event(logger, "order_filled", **fill)
            time.sleep(0.3)

        except Exception as e:
            log_event(logger, "order_error", asset=t["asset"], side=t["side"],
                      qty=t["qty"], error=str(e))
            raise
    return fills


# ── Rolling Sharpe (168-bar = 7 days) ─────────────────────────────────────────
def rolling_sharpe(pnl_history: list[dict], window: int = 168) -> Optional[float]:
    if len(pnl_history) < 10:
        return None
    vals = [p["value"] for p in pnl_history[-window:]]
    rets = np.diff(vals) / np.array(vals[:-1])
    if rets.std() == 0:
        return None
    return round(rets.mean() / rets.std() * np.sqrt(8760), 3)


# ── Circuit breakers ──────────────────────────────────────────────────────────
def check_circuit_breakers(
    state: dict,
    portfolio_value: float,
    last_prices: dict[str, float],
    prev_prices: dict[str, float],
    cfg: dict,
) -> Optional[str]:
    # 1. Portfolio drawdown from peak
    peak = state.get("peak_value") or portfolio_value
    dd   = (portfolio_value - peak) / peak * 100
    if dd <= -cfg["max_drawdown_pct"]:
        return f"drawdown {dd:.1f}% exceeds -{cfg['max_drawdown_pct']}%"

    # 2. Single-asset 1h drop
    for asset in cfg["assets"]:
        if asset in last_prices and asset in prev_prices:
            drop = (last_prices[asset] - prev_prices[asset]) / prev_prices[asset] * 100
            if drop <= -cfg["single_asset_drop_pct"]:
                return f"{asset} dropped {drop:.1f}% in 1h"

    return None


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    cfg        = load_config()
    log_dir    = BASE_DIR / cfg["log_dir"]
    state_file = BASE_DIR / cfg["state_file"]
    logger     = make_logger(log_dir)
    state      = load_state(state_file, cfg)
    # Carry fee_rate and strategy_version into state
    state["fee_rate"]         = cfg["fee_rate"]
    state["strategy_version"] = cfg.get("strategy_version", "unknown")

    _audit.init_db()

    log_event(logger, "engine_start",
              paper_mode=cfg["paper_mode"],
              exchange=cfg["exchange"],
              assets=cfg["assets"],
              strategy_version=cfg.get("strategy_version", "unknown"),
              capital=cfg["capital_usdt"])
    _audit.log_event("engine_start", {
        "paper_mode": cfg["paper_mode"],
        "exchange":   cfg["exchange"],
        "assets":     cfg["assets"],
        "strategy_version": cfg.get("strategy_version"),
    }, strategy_version=cfg.get("strategy_version", ""))
    _alerts.alert_engine_start(cfg, paper=cfg["paper_mode"])

    if state.get("circuit_breaker"):
        logger.error(f"CIRCUIT BREAKER ACTIVE: {state['circuit_breaker']}")
        logger.error("Clear state.json circuit_breaker field to resume.")
        sys.exit(1)

    ex         = make_exchange(cfg)
    assets     = cfg["assets"]
    prev_prices: dict[str, float] = {}
    _last_pnl_alert_pct: float = 0.0   # track last alerted pnl level

    while True:
        now = datetime.now(timezone.utc)

        # ── Sleep until next top of hour ──────────────────────────────────
        next_hour = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
        )
        sleep_secs = (next_hour - now).total_seconds()
        logger.info(f"Sleeping {sleep_secs/60:.1f} min until {next_hour.strftime('%H:%M')} UTC")
        time.sleep(max(sleep_secs - 5, 0))   # wake up 5s early
        time.sleep(5)                          # let exchange settle

        loop_start = datetime.now(timezone.utc)
        try:
            # ── Fetch market data ────────────────────────────────────────
            closes, volumes = fetch_all_ohlcv(ex, assets, limit=210)
            last_prices = {a: float(closes[a].iloc[-1]) for a in assets}

            # ── Portfolio value ──────────────────────────────────────────
            portfolio_value, holdings = get_portfolio_value(
                ex, assets, last_prices, state, cfg["paper_mode"]
            )

            # Initialise inception value
            if state["inception_value"] is None:
                state["inception_value"] = portfolio_value
            if state["peak_value"] is None or portfolio_value > state["peak_value"]:
                state["peak_value"] = portfolio_value

            # Track P&L for rolling Sharpe
            state["pnl_history"].append(dict(
                ts=loop_start.isoformat(), value=portfolio_value
            ))
            state["pnl_history"] = state["pnl_history"][-500:]  # keep last 500
            save_state(state, state_file)

            # ── Circuit breaker check ─────────────────────────────────────
            cb = check_circuit_breakers(
                state, portfolio_value, last_prices, prev_prices, cfg
            )
            if cb:
                log_event(logger, "circuit_breaker_triggered", reason=cb,
                          portfolio_value=portfolio_value)
                _audit.log_event("circuit_breaker_triggered",
                                 {"reason": cb, "portfolio_value": portfolio_value},
                                 strategy_version=cfg.get("strategy_version", ""))
                _alerts.alert_circuit_breaker(cfg, cb, portfolio_value)
                state["circuit_breaker"] = cb
                save_state(state, state_file)
                logger.error(f"CIRCUIT BREAKER: {cb}. Halting. Fix state.json to resume.")
                sys.exit(1)

            # ── Compute signal always (for logging) ───────────────────────
            target_weights, in_regime, btc_px, btc_ma, scores = compute_target_weights(
                closes, volumes, cfg, last_regime=state.get("last_regime", True)
            )
            ret_24h = closes.pct_change(cfg["lookback_bars"]).iloc[-1].to_dict()
            rs      = rolling_sharpe(state["pnl_history"])

            # ── Decide if it's rebalance time ────────────────────────────
            last_rebal = (datetime.fromisoformat(state["last_rebal_ts"])
                          if state["last_rebal_ts"] else None)
            hours_since_rebal = (
                (loop_start - last_rebal).total_seconds() / 3600
                if last_rebal else 9999
            )
            due_for_rebal = hours_since_rebal >= cfg["rebalance_interval_hours"]

            # ── Regime-change emergency exit ──────────────────────────────
            # If BTC just crossed below the MA and we hold positions,
            # exit to cash immediately — don't wait for the weekly rebalance.
            # This matches the backtest, which exits on the next bar after a breach.
            was_in_regime = state.get("last_regime", True)
            regime_breach = was_in_regime and not in_regime
            has_positions = any(float(v) > 0 for v in holdings.values())
            emergency_exit = regime_breach and has_positions
            state["last_regime"] = in_regime

            log_event(logger, "hourly_check",
                      portfolio_value=round(portfolio_value, 2),
                      pnl_pct=round((portfolio_value / state["inception_value"] - 1) * 100, 2),
                      in_regime=in_regime,
                      btc_price=round(btc_px, 2),
                      btc_ma=round(btc_ma, 2),
                      hours_since_rebal=round(hours_since_rebal, 1),
                      due_for_rebal=due_for_rebal,
                      emergency_exit=emergency_exit,
                      rolling_sharpe_7d=rs,
                      rankings={a: round(v, 4) for a, v in ret_24h.items()},
                      composite_scores={a: round(v, 4) for a, v in scores.items()},
                      target_weights=target_weights)

            # ── Audit hourly snapshot ─────────────────────────────────────
            _audit.log_snapshot(
                state, portfolio_value, in_regime, btc_px, btc_ma, holdings,
                strategy_version=cfg.get("strategy_version", ""),
            )

            # ── Daily P&L alert (fires once per ±3% move from inception) ──
            pnl_pct = (portfolio_value / state["inception_value"] - 1) * 100 if state["inception_value"] else 0.0
            if abs(pnl_pct - _last_pnl_alert_pct) >= 3.0:
                _alerts.alert_daily_pnl(cfg, portfolio_value, pnl_pct)
                _last_pnl_alert_pct = round(pnl_pct / 3.0) * 3.0

            if not due_for_rebal and not emergency_exit:
                prev_prices = last_prices
                state["consecutive_errors"] = 0
                save_state(state, state_file)
                continue

            if emergency_exit:
                log_event(logger, "regime_breach_exit",
                          btc_price=round(btc_px, 2),
                          btc_ma=round(btc_ma, 2),
                          holdings={a: round(q, 6) for a, q in holdings.items()})
                _audit.log_event("regime_breach_exit",
                                 {"btc_price": round(btc_px, 2), "btc_ma": round(btc_ma, 2)},
                                 strategy_version=cfg.get("strategy_version", ""))
                _alerts.alert_regime_breach(cfg, btc_px, btc_ma, portfolio_value)
            elif not was_in_regime and in_regime:
                # Regime recovery — BTC crossed back above MA
                _audit.log_event("regime_recovery",
                                 {"btc_price": round(btc_px, 2), "btc_ma": round(btc_ma, 2)},
                                 strategy_version=cfg.get("strategy_version", ""))
                _alerts.alert_regime_recovery(cfg, btc_px, btc_ma)

            # ── Rebalance ─────────────────────────────────────────────────
            trades = compute_rebal_trades(
                holdings, target_weights, portfolio_value,
                last_prices, cfg["fee_rate"]
            )

            log_event(logger, "rebalance_start",
                      rebal_num=state["rebal_count"] + 1,
                      in_regime=in_regime,
                      target_weights=target_weights,
                      n_trades=len(trades),
                      current_holdings={a: round(q, 6) for a, q in holdings.items()})

            fills = execute_trades(ex, trades, cfg["paper_mode"], state, logger)

            # Audit each fill
            sv = cfg.get("strategy_version", "")
            for fill in fills:
                _audit.log_trade(fill, strategy_version=sv)

            # Update state
            state["last_rebal_ts"] = loop_start.isoformat()
            state["rebal_count"]  += 1
            state["consecutive_errors"] = 0

            # Re-value portfolio after simulated trades
            time.sleep(1)
            portfolio_value, holdings = get_portfolio_value(
                ex, assets, last_prices, state, cfg["paper_mode"]
            )
            if portfolio_value > state["peak_value"]:
                state["peak_value"] = portfolio_value

            log_event(logger, "rebalance_complete",
                      rebal_num=state["rebal_count"],
                      fills=fills,
                      portfolio_value=round(portfolio_value, 2),
                      holdings={a: round(q, 6) for a, q in holdings.items()},
                      rolling_sharpe_7d=rs)
            _audit.log_event("rebalance_complete",
                             {"rebal_num": state["rebal_count"],
                              "portfolio_value": round(portfolio_value, 2),
                              "n_fills": len(fills)},
                             strategy_version=sv)
            _alerts.alert_rebalance(cfg, fills, portfolio_value, state["rebal_count"])

        except ccxt.NetworkError as e:
            state["consecutive_errors"] = state.get("consecutive_errors", 0) + 1
            log_event(logger, "network_error", error=str(e),
                      consecutive=state["consecutive_errors"])
            if state["consecutive_errors"] >= cfg["max_consecutive_errors"]:
                state["circuit_breaker"] = f"ccxt network error ×{state['consecutive_errors']}: {e}"
                save_state(state, state_file)
                logger.error("Max consecutive errors reached. Halting.")
                sys.exit(1)

        except Exception as e:
            log_event(logger, "unexpected_error", error=str(e))
            logger.exception("Unexpected error — continuing after 60s backoff")
            time.sleep(60)
            continue

        finally:
            prev_prices = last_prices if "last_prices" in dir() else {}
            save_state(state, state_file)


if __name__ == "__main__":
    run()
