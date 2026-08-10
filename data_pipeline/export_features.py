"""
Feature Export Utility — Binance Futures
Exports a merged hourly DataFrame of all collected Binance signals,
aligned to the hour bar close, forward-filled, saved as Parquet.

Output columns are identical to the previous schema so XGBoost
feature_builder.py and train.py require no changes.

Usage:
    python export_features.py --start 2025-01-01 --end 2025-03-01
    python export_features.py --start 2025-01-01 --end 2025-03-01 --ccys BTC ETH
    python export_features.py  # defaults: all assets, last 30 days
"""

import argparse
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "binance_features.db"
EXPORT_DIR = BASE_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CCYS       = ["BTC", "ETH", "DOGE", "ADA", "SOL"]
CCY_TO_SYMBOL  = {c: f"{c}USDT_PERP.A" for c in ALL_CCYS}


def load_funding(conn: sqlite3.Connection, ccys: list[str],
                 start_ms: int, end_ms: int) -> pd.DataFrame:
    symbols      = [CCY_TO_SYMBOL[c] for c in ccys]
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(f"""
        SELECT symbol, funding_time, funding_rate
        FROM funding_rates
        WHERE symbol IN ({placeholders})
          AND funding_time BETWEEN ? AND ?
        ORDER BY symbol, funding_time
    """, [*symbols, start_ms, end_ms]).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "ts_ms", "funding_rate"])
    df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("dt")

    out = {}
    for ccy in ccys:
        sym = CCY_TO_SYMBOL[ccy]
        sub = df[df["symbol"] == sym]
        sub = sub[~sub.index.duplicated(keep="last")]
        out[f"{ccy}_funding_rate"] = sub["funding_rate"]
    return pd.DataFrame(out)


def load_oi(conn: sqlite3.Connection, ccys: list[str],
            start_ms: int, end_ms: int) -> pd.DataFrame:
    symbols      = [CCY_TO_SYMBOL[c] for c in ccys]
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(f"""
        SELECT symbol, timestamp, oi_usd, volume_usd
        FROM open_interest
        WHERE symbol IN ({placeholders})
          AND timestamp BETWEEN ? AND ?
        ORDER BY symbol, timestamp
    """, [*symbols, start_ms, end_ms]).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "ts_ms", "oi_usd", "volume_usd"])
    df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("dt")

    out = {}
    for ccy in ccys:
        sym = CCY_TO_SYMBOL[ccy]
        sub = df[df["symbol"] == sym]
        sub = sub[~sub.index.duplicated(keep="last")]
        out[f"{ccy}_oi_usd"]     = sub["oi_usd"]
        out[f"{ccy}_volume_usd"] = sub["volume_usd"]
    return pd.DataFrame(out)


def load_ls(conn: sqlite3.Connection, ccys: list[str],
            start_ms: int, end_ms: int) -> pd.DataFrame:
    symbols      = [CCY_TO_SYMBOL[c] for c in ccys]
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(f"""
        SELECT symbol, timestamp, long_short_ratio
        FROM ls_ratio
        WHERE symbol IN ({placeholders})
          AND timestamp BETWEEN ? AND ?
        ORDER BY symbol, timestamp
    """, [*symbols, start_ms, end_ms]).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "ts_ms", "ls_ratio"])
    df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("dt")

    out = {}
    for ccy in ccys:
        sym = CCY_TO_SYMBOL[ccy]
        sub = df[df["symbol"] == sym]
        sub = sub[~sub.index.duplicated(keep="last")]
        out[f"{ccy}_ls_ratio"] = sub["ls_ratio"]
    return pd.DataFrame(out)


def load_liquidations(conn: sqlite3.Connection, ccys: list[str],
                      start_ms: int, end_ms: int) -> pd.DataFrame:
    """Aggregate to hourly: sum notional USD by side."""
    symbols      = [CCY_TO_SYMBOL[c] for c in ccys]
    placeholders = ",".join("?" * len(symbols))
    rows = conn.execute(f"""
        SELECT symbol, side, quantity, price, timestamp
        FROM liquidations
        WHERE symbol IN ({placeholders})
          AND timestamp BETWEEN ? AND ?
    """, [*symbols, start_ms, end_ms]).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["symbol", "side", "qty", "price", "ts_ms"])
    df["dt"]       = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.floor("h")
    df["notional"] = df["qty"] * df["price"]

    out_frames = []
    for ccy in ccys:
        sym = CCY_TO_SYMBOL[ccy]
        sub = df[df["symbol"] == sym]
        # Binance: side=sell → long position liquidated; side=buy → short liquidated
        long_liq  = (sub[sub["side"] == "sell"].groupby("dt")["notional"].sum()
                     .rename(f"{ccy}_liq_long_usd"))
        short_liq = (sub[sub["side"] == "buy"].groupby("dt")["notional"].sum()
                     .rename(f"{ccy}_liq_short_usd"))
        out_frames.extend([long_liq, short_liq])

    return pd.concat(out_frames, axis=1) if out_frames else pd.DataFrame()


def export(
    start: datetime,
    end: datetime,
    ccys: list[str] = ALL_CCYS,
    output_path: Path | None = None,
) -> pd.DataFrame:
    conn     = sqlite3.connect(DB_PATH)
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp() * 1000)

    print(f"Exporting {start.date()} → {end.date()} | ccys: {ccys}")

    hourly_idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    result     = pd.DataFrame(index=hourly_idx)

    loaders = {
        "funding":  load_funding(conn, ccys, start_ms, end_ms),
        "oi":       load_oi(conn, ccys, start_ms, end_ms),
        "ls_ratio": load_ls(conn, ccys, start_ms, end_ms),
        "liq":      load_liquidations(conn, ccys, start_ms, end_ms),
    }

    for name, df in loaders.items():
        if df.empty:
            print(f"  WARNING: no {name} data in range")
            continue
        df = df.reindex(hourly_idx, method="ffill", limit=25)
        result = result.join(df, how="left")
        print(f"  {name}: {len(df.columns)} cols, "
              f"{df.notna().any(axis=1).sum()}/{len(df)} rows with data")

    liq_cols = [c for c in result.columns if "_liq_" in c]
    result[liq_cols] = result[liq_cols].fillna(0)

    conn.close()

    if output_path is None:
        ccy_str     = "_".join(ccys)
        fname       = f"features_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{ccy_str}.parquet"
        output_path = EXPORT_DIR / fname

    result.to_parquet(output_path)
    print(f"\nExported {len(result)} rows × {len(result.columns)} cols → {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Export Binance features to parquet")
    parser.add_argument("--start",  default=None, help="Start YYYY-MM-DD")
    parser.add_argument("--end",    default=None, help="End YYYY-MM-DD")
    parser.add_argument("--ccys",   nargs="+", default=ALL_CCYS,
                        help="Base currencies e.g. BTC ETH")
    parser.add_argument("--output", default=None, help="Output parquet path")
    args = parser.parse_args()

    now   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = (datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
             if args.start else now - timedelta(days=30))
    end   = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
             if args.end else now)

    export(start, end, args.ccys, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
