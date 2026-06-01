import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from credentials import TV_USERNAME, TV_PASSWORD

# --- config ---
SYMBOLS_YF = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "WIPRO.NS", "BAJFINANCE.NS",
    "AXISBANK.NS", "KOTAKBANK.NS"
]
SYMBOLS_TV = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK",
    "ICICIBANK", "SBIN", "WIPRO", "BAJFINANCE",
    "AXISBANK", "KOTAKBANK"
]

TV_EXCHANGE = "NSE"
TV_BARS     = 50

INTERVAL_YF = "30m"
PERIOD_YF   = "5d"
OUT_FILE    = "data/testdata.parquet"

SCHEMA = pa.schema([
    ("sno",       pa.int32()),
    ("datetime",  pa.timestamp("ns", tz="UTC")),
    ("open",      pa.float64()),
    ("high",      pa.float64()),
    ("low",       pa.float64()),
    ("close",     pa.float64()),
    ("adj_close", pa.float64()),
    ("volume",    pa.float64()),
    ("symbol",    pa.string()),
])

os.makedirs("data", exist_ok=True)


def fetch_yfinance():
    frames = []
    for symbol in SYMBOLS_YF:
        print(f"Fetching {symbol} from yfinance...")
        df = yf.download(symbol, period=PERIOD_YF, interval=INTERVAL_YF,
                         progress=False, auto_adjust=False)

        if df.empty:
            print(f"  WARNING: no data for {symbol}\n")
            continue

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df = df.rename(columns={"adj close": "adj_close"})

        df.index.name = "datetime"
        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["symbol"]   = symbol.replace(".NS", "_NS")

        frames.append(df)
        del df

    return frames


def fetch_tvdatafeed():
    frames = []
    tv = TvDatafeed(TV_USERNAME, TV_PASSWORD)

    for symbol in SYMBOLS_TV:
        print(f"Fetching {symbol} from tvdatafeed...")
        df = tv.get_hist(symbol=symbol, exchange=TV_EXCHANGE,
                         interval=Interval.in_30_minute, n_bars=TV_BARS)

        if df is None or df.empty:
            print(f"  WARNING: no data for {symbol}\n")
            continue

        df.columns = [c.lower() for c in df.columns]
        df.index.name = "datetime"
        df = df.reset_index()
        df["datetime"]  = pd.to_datetime(df["datetime"], utc=True)
        df["symbol"]    = f"{symbol}_NS"
        df["adj_close"] = float("nan")

        frames.append(df)
        del df

    return frames


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("yfinance", "tvdatafeed"):
        print("Usage: python3 testdata.py yfinance | tvdatafeed")
        sys.exit(1)

    source = sys.argv[1]

    if source == "yfinance":
        frames = fetch_yfinance()
    else:
        frames = fetch_tvdatafeed()

    if not frames:
        print("No data fetched. Exiting.")
        sys.exit(1)

    print("Combining and writing to parquet...")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["datetime", "open", "high", "low", "close", "adj_close", "volume", "symbol"]]

    # sort by symbol then datetime, then assign per-symbol sno starting from 1
    combined = combined.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    combined["sno"] = combined.groupby("symbol").cumcount() + 1
    combined = combined[["sno", "datetime", "open", "high", "low", "close", "adj_close", "volume", "symbol"]]

    table = pa.Table.from_pandas(combined, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, OUT_FILE)

    print(f"\nTotal rows : {len(combined)}")
    print(f"Memory     : {combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Saved      -> {OUT_FILE}\n")

    for symbol, group in combined.groupby("symbol"):
        print(f"{symbol} -- {len(group)} bars | sno {group['sno'].iloc[0]} -> {group['sno'].iloc[-1]} | {group['datetime'].iloc[0]} -> {group['datetime'].iloc[-1]}")
    print()
    # print(combined.to_string()) # prints all entries

    del combined, table