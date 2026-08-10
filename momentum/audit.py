"""
Append-only SQLite audit log for the momentum engine.
Never UPDATE or DELETE — every trade and snapshot is immutable.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "audit.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT    NOT NULL,
                strategy_version TEXT,
                asset            TEXT,
                side             TEXT,
                qty              REAL,
                fill_px          REAL,
                notional_usd     REAL,
                fee_usd          REAL,
                slippage_bps     REAL,
                order_id         TEXT,
                paper            INTEGER
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT    NOT NULL,
                strategy_version TEXT,
                portfolio_value  REAL,
                inception_value  REAL,
                pnl_pct          REAL,
                peak_value       REAL,
                drawdown_pct     REAL,
                in_regime        INTEGER,
                btc_price        REAL,
                btc_ma           REAL,
                holdings         TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT    NOT NULL,
                strategy_version TEXT,
                event            TEXT,
                payload          TEXT
            );
        """)


def log_trade(fill: dict, strategy_version: str = ""):
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO trades
               (ts, strategy_version, asset, side, qty, fill_px,
                notional_usd, fee_usd, slippage_bps, order_id, paper)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                strategy_version,
                fill.get("asset"),
                fill.get("side"),
                fill.get("qty"),
                fill.get("fill_px"),
                fill.get("notional_usdt"),
                fill.get("fee_usdt"),
                fill.get("slippage_bps"),
                fill.get("order_id"),
                1 if fill.get("paper") else 0,
            ),
        )


def log_snapshot(
    state: dict,
    portfolio_value: float,
    in_regime: bool,
    btc_price: float,
    btc_ma: float,
    holdings: dict,
    strategy_version: str = "",
):
    ts = datetime.now(timezone.utc).isoformat()
    inception = state.get("inception_value") or portfolio_value
    peak = state.get("peak_value") or portfolio_value
    pnl_pct = (portfolio_value / inception - 1) * 100 if inception else 0.0
    dd_pct = (portfolio_value / peak - 1) * 100 if peak else 0.0

    with _connect() as conn:
        conn.execute(
            """INSERT INTO snapshots
               (ts, strategy_version, portfolio_value, inception_value,
                pnl_pct, peak_value, drawdown_pct, in_regime,
                btc_price, btc_ma, holdings)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ts,
                strategy_version,
                round(portfolio_value, 4),
                round(inception, 4),
                round(pnl_pct, 4),
                round(peak, 4),
                round(dd_pct, 4),
                1 if in_regime else 0,
                round(btc_price, 2),
                round(btc_ma, 2),
                json.dumps({k: round(v, 8) for k, v in holdings.items()}),
            ),
        )


def log_event(event: str, payload: dict, strategy_version: str = ""):
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
<<<<<<< HEAD
            "INSERT INTO events (ts, strategy_version, Event, payload) VALUES (?,?,?,?)",
            (ts, strategy_version, event, json.dumps(payload, default=str)),
        )


def reject_proposal(proposal: str, reason: str) -> None:
    log_event("rejected_proposal",
               {"proposal": proposal,
                "reason": reason},
               strategy_version=None
    )
=======
            "INSERT INTO events (ts, strategy_version, event, payload) VALUES (?,?,?,?)",
            (ts, strategy_version, event, json.dumps(payload, default=str)),
        )
>>>>>>> f54edeb (Initial commit)
