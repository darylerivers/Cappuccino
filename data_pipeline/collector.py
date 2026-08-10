"""
Coinalyze Data Pipeline Collector
Collects: funding rates, open interest, L/S ratio, liquidations
Storage:  SQLite at ~/cappuccino/data_pipeline/binance_features.db
Auth:     Coinalyze API key (free tier, set in collector_config.json)

Coinalyze provides aggregated data across all major exchanges.
Symbols: BTCUSDT_PERP.A (.A = aggregated), 1h bars.
History depth: ~60 days on free tier.
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import aiohttp

# Global semaphore: Coinalyze free tier allows ~1 req/sec
_API_SEM = asyncio.Semaphore(1)

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "binance_features.db"
CONFIG_FILE = BASE_DIR / "collector_config.json"
BASE_URL   = "https://api.coinalyze.net/v1"

ASSETS  = ["BTC", "ETH", "DOGE", "ADA", "SOL"]
SYMBOLS = [f"{a}USDT_PERP.A" for a in ASSETS]   # aggregated perp symbols
SYM_STR = ",".join(SYMBOLS)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "collector.log"),
    ],
)
log = logging.getLogger("collector")


def load_api_key() -> str:
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text()).get("coinalyze_api_key", "")
    return ""


# ── Database ───────────────────────────────────────────────────────────────────

def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            symbol       TEXT    NOT NULL,
            funding_time INTEGER NOT NULL,
            funding_rate REAL    NOT NULL,
            PRIMARY KEY (symbol, funding_time)
        );
        CREATE INDEX IF NOT EXISTS idx_fr ON funding_rates(symbol, funding_time);

        CREATE TABLE IF NOT EXISTS open_interest (
            symbol     TEXT    NOT NULL,
            timestamp  INTEGER NOT NULL,
            oi_usd     REAL    NOT NULL,
            volume_usd REAL,
            PRIMARY KEY (symbol, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_oi ON open_interest(symbol, timestamp);

        CREATE TABLE IF NOT EXISTS ls_ratio (
            symbol           TEXT    NOT NULL,
            timestamp        INTEGER NOT NULL,
            long_short_ratio REAL    NOT NULL,
            PRIMARY KEY (symbol, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_ls ON ls_ratio(symbol, timestamp);

        CREATE TABLE IF NOT EXISTS liquidations (
            symbol    TEXT    NOT NULL,
            side      TEXT    NOT NULL,
            quantity  REAL    NOT NULL,
            price     REAL    NOT NULL,
            timestamp INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS _meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    conn.commit()
    return conn


# ── HTTP helper ────────────────────────────────────────────────────────────────

async def cg_get(
    session: aiohttp.ClientSession,
    api_key: str,
    endpoint: str,
    params: dict | None = None,
    _retries: int = 4,
) -> list | None:
    url = BASE_URL + endpoint
    for attempt in range(_retries):
        async with _API_SEM:
            try:
                async with session.get(
                    url, params=params,
                    headers={"api_key": api_key},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as r:
                    if r.status == 429:
                        retry_after = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
                        log.info(f"Coinalyze rate-limited, waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        continue
                    if r.status != 200:
                        body = await r.text()
                        log.warning(f"Coinalyze {endpoint} → HTTP {r.status}: {body[:200]}")
                        return None
                    await asyncio.sleep(1.1)   # stay under ~1 req/sec after each successful call
                    return await r.json()
            except Exception as e:
                log.warning(f"Coinalyze {endpoint} error: {e}")
                await asyncio.sleep(2 ** attempt)
    return None


# ── Backfill helper ────────────────────────────────────────────────────────────

async def fetch_history(
    session: aiohttp.ClientSession,
    api_key: str,
    endpoint: str,
    from_ts: int,
    to_ts: int,
) -> list | None:
    """Fetch a history endpoint for all symbols in one request."""
    return await cg_get(session, api_key, endpoint, {
        "symbols":  SYM_STR,
        "interval": "1hour",
        "from":     from_ts,
        "to":       to_ts,
    })


# ── Funding rates ──────────────────────────────────────────────────────────────

async def poll_funding(session: aiohttp.ClientSession, conn: sqlite3.Connection, api_key: str):
    """Backfill 60 days, then poll every hour."""
    meta = conn.execute("SELECT value FROM _meta WHERE key='funding_backfill_done'").fetchone()
    if not meta:
        log.info("Funding: backfilling 60 days...")
        total = 0
        now   = int(datetime.now(timezone.utc).timestamp())
        step  = 86400 * 7  # 7-day chunks to stay within response limits

        from_ts = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp())
        while from_ts < now:
            to_ts = min(from_ts + step, now)
            data  = await fetch_history(session, api_key, "/funding-rate-history", from_ts, to_ts)
            if data:
                rows = []
                for item in data:
                    sym = item["symbol"]
                    for bar in item.get("history", []):
                        # t in Unix seconds → store as ms for consistency
                        rows.append((sym, bar["t"] * 1000, bar["c"]))
                conn.executemany("INSERT OR IGNORE INTO funding_rates VALUES (?,?,?)", rows)
                conn.commit()
                total += len(rows)
            from_ts = to_ts
            await asyncio.sleep(0.5)

        conn.execute("INSERT OR REPLACE INTO _meta VALUES ('funding_backfill_done', ?)",
                     (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
        log.info(f"Funding backfill complete: {total} rows")

    while True:
        await asyncio.sleep(3600)
        now = int(datetime.now(timezone.utc).timestamp())
        data = await fetch_history(session, api_key, "/funding-rate-history", now - 7200, now)
        if data:
            rows = []
            for item in data:
                for bar in item.get("history", []):
                    rows.append((item["symbol"], bar["t"] * 1000, bar["c"]))
            conn.executemany("INSERT OR IGNORE INTO funding_rates VALUES (?,?,?)", rows)
            conn.commit()
            log.info(f"Funding poll: {len(rows)} rows")


# ── Open interest ──────────────────────────────────────────────────────────────

async def poll_oi(session: aiohttp.ClientSession, conn: sqlite3.Connection, api_key: str):
    """Backfill 60 days, then poll every hour."""
    meta = conn.execute("SELECT value FROM _meta WHERE key='oi_backfill_done'").fetchone()
    if not meta:
        log.info("OI: backfilling 60 days...")
        total   = 0
        now     = int(datetime.now(timezone.utc).timestamp())
        step    = 86400 * 7
        from_ts = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp())
        while from_ts < now:
            to_ts = min(from_ts + step, now)
            data  = await fetch_history(session, api_key, "/open-interest-history", from_ts, to_ts)
            if data:
                rows = []
                for item in data:
                    for bar in item.get("history", []):
                        rows.append((item["symbol"], bar["t"] * 1000, bar["c"], None))
                conn.executemany("INSERT OR IGNORE INTO open_interest VALUES (?,?,?,?)", rows)
                conn.commit()
                total += len(rows)
            from_ts = to_ts
            await asyncio.sleep(0.5)
        conn.execute("INSERT OR REPLACE INTO _meta VALUES ('oi_backfill_done', ?)",
                     (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
        log.info(f"OI backfill complete: {total} rows")

    while True:
        await asyncio.sleep(3600)
        now  = int(datetime.now(timezone.utc).timestamp())
        data = await fetch_history(session, api_key, "/open-interest-history", now - 7200, now)
        if data:
            rows = []
            for item in data:
                for bar in item.get("history", []):
                    rows.append((item["symbol"], bar["t"] * 1000, bar["c"], None))
            conn.executemany("INSERT OR IGNORE INTO open_interest VALUES (?,?,?,?)", rows)
            conn.commit()
            log.info(f"OI poll: {len(rows)} rows")


# ── Long/short ratio ───────────────────────────────────────────────────────────

async def poll_ls(session: aiohttp.ClientSession, conn: sqlite3.Connection, api_key: str):
    """Backfill 60 days, then poll every hour."""
    meta = conn.execute("SELECT value FROM _meta WHERE key='ls_backfill_done'").fetchone()
    if not meta:
        log.info("L/S: backfilling 60 days...")
        total   = 0
        now     = int(datetime.now(timezone.utc).timestamp())
        step    = 86400 * 7
        from_ts = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp())
        while from_ts < now:
            to_ts = min(from_ts + step, now)
            data  = await fetch_history(session, api_key, "/long-short-ratio-history", from_ts, to_ts)
            if data:
                rows = []
                for item in data:
                    for bar in item.get("history", []):
                        rows.append((item["symbol"], bar["t"] * 1000, bar["r"]))
                conn.executemany("INSERT OR IGNORE INTO ls_ratio VALUES (?,?,?)", rows)
                conn.commit()
                total += len(rows)
            from_ts = to_ts
            await asyncio.sleep(0.5)
        conn.execute("INSERT OR REPLACE INTO _meta VALUES ('ls_backfill_done', ?)",
                     (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
        log.info(f"L/S backfill complete: {total} rows")

    while True:
        await asyncio.sleep(3600)
        now  = int(datetime.now(timezone.utc).timestamp())
        data = await fetch_history(session, api_key, "/long-short-ratio-history", now - 7200, now)
        if data:
            rows = []
            for item in data:
                for bar in item.get("history", []):
                    rows.append((item["symbol"], bar["t"] * 1000, bar["r"]))
            conn.executemany("INSERT OR IGNORE INTO ls_ratio VALUES (?,?,?)", rows)
            conn.commit()
            log.info(f"L/S poll: {len(rows)} rows")


# ── Liquidations ───────────────────────────────────────────────────────────────

async def poll_liquidations(session: aiohttp.ClientSession, conn: sqlite3.Connection, api_key: str):
    """Backfill 60 days, then poll every hour.
    Coinalyze returns l=long liq USD, s=short liq USD per hourly bar.
    Stored as two rows per bar: side=sell (long liq) and side=buy (short liq).
    """
    meta = conn.execute("SELECT value FROM _meta WHERE key='liq_backfill_done'").fetchone()
    if not meta:
        log.info("Liquidations: backfilling 60 days...")
        total   = 0
        now     = int(datetime.now(timezone.utc).timestamp())
        step    = 86400 * 7
        from_ts = int((datetime.now(timezone.utc) - timedelta(days=60)).timestamp())
        while from_ts < now:
            to_ts = min(from_ts + step, now)
            data  = await fetch_history(session, api_key, "/liquidation-history", from_ts, to_ts)
            if data:
                rows = []
                for item in data:
                    for bar in item.get("history", []):
                        ts = bar["t"] * 1000
                        if bar.get("l", 0):
                            rows.append((item["symbol"], "sell", bar["l"], 1.0, ts))
                        if bar.get("s", 0):
                            rows.append((item["symbol"], "buy",  bar["s"], 1.0, ts))
                conn.executemany("INSERT INTO liquidations VALUES (?,?,?,?,?)", rows)
                conn.commit()
                total += len(rows)
            from_ts = to_ts
            await asyncio.sleep(0.5)
        conn.execute("INSERT OR REPLACE INTO _meta VALUES ('liq_backfill_done', ?)",
                     (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
        log.info(f"Liquidations backfill complete: {total} rows")

    while True:
        await asyncio.sleep(3600)
        now  = int(datetime.now(timezone.utc).timestamp())
        data = await fetch_history(session, api_key, "/liquidation-history", now - 7200, now)
        if data:
            rows = []
            for item in data:
                for bar in item.get("history", []):
                    ts = bar["t"] * 1000
                    if bar.get("l", 0):
                        rows.append((item["symbol"], "sell", bar["l"], 1.0, ts))
                    if bar.get("s", 0):
                        rows.append((item["symbol"], "buy",  bar["s"], 1.0, ts))
            conn.executemany("INSERT INTO liquidations VALUES (?,?,?,?,?)", rows)
            conn.commit()
            log.info(f"Liquidations poll: {len(rows)} rows")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    api_key = load_api_key()
    if not api_key:
        log.error(f"No API key found. Add coinalyze_api_key to {CONFIG_FILE}")
        return

    # Delete old stub DB if it existed with wrong schema
    if DB_PATH.exists():
        conn_test = sqlite3.connect(DB_PATH)
        rows = conn_test.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
        conn_test.close()
        if rows < 100:
            log.info("Removing old stub DB, starting fresh.")
            DB_PATH.unlink()
            for wal in [DB_PATH.with_suffix(".db-shm"), DB_PATH.with_suffix(".db-wal")]:
                if wal.exists():
                    wal.unlink()

    conn = init_db(DB_PATH)
    log.info(f"Database: {DB_PATH}")
    log.info(f"Symbols:  {SYMBOLS}")

    async def _staggered(fn, delay):
        await asyncio.sleep(delay)
        await fn(session, conn, api_key)

    async with aiohttp.ClientSession() as session:
        await asyncio.gather(
            _staggered(poll_funding,      0),
            _staggered(poll_oi,          15),
            _staggered(poll_ls,          30),
            _staggered(poll_liquidations, 45),
        )


if __name__ == "__main__":
    asyncio.run(main())
