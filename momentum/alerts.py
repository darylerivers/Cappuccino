"""
Telegram alerting for the momentum engine.
Silently no-ops if bot_token or chat_id are not configured.
"""

import logging

log = logging.getLogger("momentum")


def send_alert(message: str, bot_token: str, chat_id: str) -> bool:
    """Send a Telegram message. Returns True on success, False on failure."""
    if not bot_token or not chat_id:
        return False
    try:
        import requests
        url  = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not resp.ok:
            log.warning(f"Telegram alert failed: {resp.status_code} {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        log.warning(f"Telegram alert error: {e}")
        return False


def fmt_currency(val: float) -> str:
    return f"${val:,.2f}"


def alert_engine_start(cfg: dict, paper: bool):
    mode = "PAPER" if paper else "🔴 LIVE"
    send_alert(
        f"*Cappuccino Engine Started* ({mode})\n"
        f"Exchange: {cfg.get('exchange', '?').title()}\n"
        f"Assets: {', '.join(cfg.get('assets', []))}\n"
        f"Strategy v{cfg.get('strategy_version', '?')}",
        cfg.get("telegram_bot_token", ""),
        cfg.get("telegram_chat_id", ""),
    )


def alert_circuit_breaker(cfg: dict, reason: str, portfolio_value: float):
    send_alert(
        f"🚨 *CIRCUIT BREAKER TRIGGERED*\n"
        f"Reason: {reason}\n"
        f"Portfolio: {fmt_currency(portfolio_value)}\n"
        f"Action: Engine halted. Edit `state.json` to resume.",
        cfg.get("telegram_bot_token", ""),
        cfg.get("telegram_chat_id", ""),
    )


def alert_regime_breach(cfg: dict, btc_px: float, btc_ma: float, portfolio_value: float):
    send_alert(
        f"⚠️ *Regime Breach — Emergency Exit*\n"
        f"BTC: {fmt_currency(btc_px)} | MA: {fmt_currency(btc_ma)}\n"
        f"Portfolio: {fmt_currency(portfolio_value)}\n"
        f"All positions closed. Waiting for regime recovery.",
        cfg.get("telegram_bot_token", ""),
        cfg.get("telegram_chat_id", ""),
    )


def alert_regime_recovery(cfg: dict, btc_px: float, btc_ma: float):
    send_alert(
        f"✅ *Regime Recovery*\n"
        f"BTC: {fmt_currency(btc_px)} | MA: {fmt_currency(btc_ma)}\n"
        f"Back in regime — next rebalance will open positions.",
        cfg.get("telegram_bot_token", ""),
        cfg.get("telegram_chat_id", ""),
    )


def alert_rebalance(cfg: dict, fills: list, portfolio_value: float, rebal_num: int):
    if not fills:
        return
    lines = [f"*Rebalance #{rebal_num}* — {len(fills)} trade(s)"]
    for f in fills:
        side_emoji = "🟢" if f["side"] == "buy" else "🔴"
        lines.append(
            f"{side_emoji} {f['side'].upper()} {f['asset']}: "
            f"{f['qty']:.6f} @ {fmt_currency(f.get('fill_px', 0))}"
        )
    lines.append(f"Portfolio: {fmt_currency(portfolio_value)}")
    send_alert(
        "\n".join(lines),
        cfg.get("telegram_bot_token", ""),
        cfg.get("telegram_chat_id", ""),
    )


def alert_daily_pnl(cfg: dict, portfolio_value: float, pnl_pct: float):
    emoji = "📈" if pnl_pct >= 0 else "📉"
    send_alert(
        f"{emoji} *Daily P&L Alert*\n"
        f"Portfolio: {fmt_currency(portfolio_value)}\n"
        f"P&L from inception: {pnl_pct:+.2f}%",
        cfg.get("telegram_bot_token", ""),
        cfg.get("telegram_chat_id", ""),
    )
