import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

# NSE session in IST
NSE_OPEN_H,  NSE_OPEN_M  = 9,  15   # 09:15 IST
NSE_CLOSE_H, NSE_CLOSE_M = 15, 15   # 15:15 IST  (last bar open time)

# Base timeframe stored on disk
BASE_TF_MINUTES = 30
BASE_TF_LABEL   = "30min"

# Bars per full trading day
BARS_PER_DAY = 13   # 09:15, 09:45 … 15:15

# Supported target timeframes: label -> (pandas offset, anchor offset in IST)
TARGET_TIMEFRAMES = {
    "30min": ("30min",  "09:15"),   # passthrough
    "1H":    ("1h",     "09:15"),   # 09:15, 10:15, 11:15 ...
    "2H":    ("2h",     "09:15"),   # 09:15, 11:15, 13:15, 15:15
    "4H":    ("4h",     "09:15"),   # 09:15, 13:15
    "1D":    ("1D",     None),      # one bar per session
    "1W":    ("1W",     None),      # one bar per Mon-Fri week
    "1M":    ("1ME",    None),      # one bar per calendar month
}

# OHLCV aggregation rules
OHLCV_AGG = {
    "open":      "first",
    "high":      "max",
    "low":       "min",
    "close":     "last",
    "volume":    "sum",
    "adj_close": "last",    # dropped silently if absent
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_ist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.tz_convert(IST)
    return df


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.tz_convert(UTC)
    return df


def _prepare(df: pd.DataFrame, symbol) -> pd.DataFrame:
    """
    - Filter by symbol if requested
    - Drop housekeeping columns (sno, symbol)
    - Ensure UTC DatetimeIndex
    - Keep only present OHLCV columns
    - Sort by datetime
    """
    df = df.copy()

    # symbol filter
    if symbol and "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
        if df.empty:
            raise ValueError(f"Symbol '{symbol}' not found in DataFrame.")

    # drop non-OHLCV columns
    drop = {"sno", "symbol"}
    df = df.drop(columns=[c for c in drop if c in df.columns])

    # ensure DatetimeIndex
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame needs a DatetimeIndex or a 'datetime' column.")
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.index.name = "datetime"

    # keep only present OHLCV cols
    keep = [c for c in OHLCV_AGG if c in df.columns]
    df = df[keep].sort_index()

    if df.empty:
        raise ValueError("No OHLCV data left after filtering.")

    return df


def _build_agg(df: pd.DataFrame) -> dict:
    return {k: v for k, v in OHLCV_AGG.items() if k in df.columns}


def _drop_empty_buckets(df: pd.DataFrame) -> pd.DataFrame:
    ohlc_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    return df.dropna(subset=ohlc_cols, how="all")


# ---------------------------------------------------------------------------
# Sub-hourly resampler  (1H, 2H, 4H)
# Anchored to 09:15 IST so buckets align to NSE session open
# ---------------------------------------------------------------------------

def _resample_intraday(df: pd.DataFrame, pandas_freq: str, anchor_ist: str) -> pd.DataFrame:
    """
    Resample to 1H / 2H / 4H.
    Uses origin="start" so buckets are anchored to the first actual
    bar in the data (e.g. 09:00 IST) — no phantom empty buckets.
    """
    agg = _build_agg(df)
    df_ist = _to_ist(df)

    resampled = (
        df_ist
        .resample(pandas_freq, origin="start", closed="left", label="left")
        .agg(agg)
        .pipe(_drop_empty_buckets)
    )

    return _to_utc(resampled)


# ---------------------------------------------------------------------------
# Daily resampler
# ---------------------------------------------------------------------------

def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    One OHLCV bar per trading day.
    Groups by IST calendar date so all bars from 09:15–15:15 IST
    land in the same bucket regardless of UTC offset.
    Bar timestamp is set to 09:15 IST of that day.
    """
    agg = _build_agg(df)
    df_ist = _to_ist(df)

    # group by IST date (midnight-normalised)
    day_key = df_ist.index.normalize()

    resampled = df_ist.groupby(day_key).agg(agg)
    resampled.index.name = "datetime"

    # stamp each daily bar at 09:15 IST (convention)
    resampled.index = resampled.index + pd.Timedelta(hours=9, minutes=15)
    if resampled.index.tz is None:
        resampled.index = resampled.index.tz_localize(IST)

    return _to_utc(_drop_empty_buckets(resampled))


# ---------------------------------------------------------------------------
# Weekly resampler
# ---------------------------------------------------------------------------

def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    One OHLCV bar per ISO week (Mon–Fri).
    Bar timestamp → Monday 09:15 IST of that week.
    """
    agg = _build_agg(df)
    df_ist = _to_ist(df)

    # (year, week_number) tuple as grouper key
    week_key = df_ist.index.to_series().apply(
        lambda dt: dt.isocalendar()[:2]
    )

    resampled = df_ist.groupby(week_key).agg(agg)

    # rebuild index: Monday 09:15 IST
    def week_to_monday_open(yw):
        year, week = yw
        monday = pd.Timestamp.fromisocalendar(year, week, 1)
        return pd.Timestamp(monday, tz=IST) + pd.Timedelta(hours=9, minutes=15)

    resampled.index = pd.DatetimeIndex(
        [week_to_monday_open(yw) for yw in resampled.index]
    )
    resampled.index.name = "datetime"

    return _to_utc(_drop_empty_buckets(resampled))


# ---------------------------------------------------------------------------
# Monthly resampler
# ---------------------------------------------------------------------------

def _resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    One OHLCV bar per calendar month.
    Bar timestamp -> first actual trading bar's datetime of that month.

    Note: to_period() drops timezone info, so we strip tz before
    grouping and re-localize the result index afterward.
    """
    agg = _build_agg(df)
    df_ist = _to_ist(df)

    # strip tz for period grouping (pandas limitation)
    df_naive = df_ist.copy()
    df_naive.index = df_ist.index.tz_localize(None)

    month_key  = df_naive.index.to_period("M")
    resampled  = df_naive.groupby(month_key).agg(agg)
    first_bars = df_naive.groupby(month_key).apply(lambda g: g.index[0])

    # re-attach IST timezone to the first-bar timestamps
    first_ts = pd.DatetimeIndex(first_bars.values).tz_localize(IST)
    resampled.index = first_ts
    resampled.index.name = "datetime"

    return _to_utc(_drop_empty_buckets(resampled))


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class NSEResampler:
    """
    Converts NSE 30-min OHLCV data to a lower frequency.

    Parameters
    ----------
    df      : pd.DataFrame
              30-min OHLCV data. datetime can be index or column (UTC).
              May be a multi-symbol parquet — use symbol= to filter.
    symbol  : str | None
              Symbol to extract from a multi-symbol DataFrame.
              e.g. "RELIANCE_NS"
    """

    def __init__(self, df: pd.DataFrame, symbol=None):
        self.symbol = symbol
        self._df    = _prepare(df, symbol)

    # ------------------------------------------------------------------

    def resample(self, target_tf: str) -> pd.DataFrame:
        """
        Resample to target_tf.

        Parameters
        ----------
        target_tf : str
            One of: "30min", "1H", "2H", "4H", "1D", "1W", "1M"

        Returns
        -------
        pd.DataFrame
            UTC DatetimeIndex | open | high | low | close | volume [| adj_close]
        """
        if target_tf not in TARGET_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe '{target_tf}'. "
                f"Choose from: {list(TARGET_TIMEFRAMES.keys())}"
            )

        pandas_freq, anchor = TARGET_TIMEFRAMES[target_tf]

        if target_tf == "30min":
            return self._df.copy()

        if target_tf in ("1H", "2H", "4H"):
            return _resample_intraday(self._df, pandas_freq, anchor)

        if target_tf == "1D":
            return _resample_daily(self._df)

        if target_tf == "1W":
            return _resample_weekly(self._df)

        if target_tf == "1M":
            return _resample_monthly(self._df)

    def resample_multi(self, target_tfs: list) -> dict:
        """Resample to multiple timeframes. Returns {tf_label: DataFrame}."""
        return {tf: self.resample(tf) for tf in target_tfs}

    # ------------------------------------------------------------------

    @staticmethod
    def convert(df: pd.DataFrame, target_tf: str, symbol=None) -> pd.DataFrame:
        """One-shot conversion without manually instantiating the class."""
        return NSEResampler(df, symbol=symbol).resample(target_tf)

    @staticmethod
    def convert_multi_symbol(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample an entire multi-symbol parquet DataFrame.
        Each symbol is processed independently then concatenated.
        Output has an extra 'symbol' column.
        """
        if "symbol" not in df.columns:
            return NSEResampler.convert(df, target_tf=target_tf)

        results = []
        for sym, group in df.groupby("symbol"):
            resampled = NSEResampler.convert(group, target_tf=target_tf, symbol=None)
            resampled["symbol"] = sym
            results.append(resampled)

        return pd.concat(results).sort_index()

    # ------------------------------------------------------------------

    def info(self) -> dict:
        ist_index = self._df.index.tz_convert(IST)
        return {
            "symbol":            self.symbol,
            "base_tf":           BASE_TF_LABEL,
            "rows":              len(self._df),
            "bars_per_day":      BARS_PER_DAY,
            "start_ist":         str(ist_index.min()),
            "end_ist":           str(ist_index.max()),
            "start_utc":         str(self._df.index.min()),
            "end_utc":           str(self._df.index.max()),
            "columns":           list(self._df.columns),
            "available_targets": list(TARGET_TIMEFRAMES.keys()),
        }

    def __repr__(self) -> str:
        i = self.info()
        return (
            f"<NSEResampler symbol={i['symbol']} rows={i['rows']} "
            f"{i['start_ist']} -> {i['end_ist']} IST>"
        )


# ---------------------------------------------------------------------------
# Self-test  (python resampler.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("NSE Resampler — self-test")
    print("=" * 65)

    def make_nse_30min(symbol: str, days: int = 5) -> pd.DataFrame:
        """Synthetic 30-min NSE bars for N trading days."""
        bars = []
        base_date = pd.Timestamp("2024-01-02", tz=IST)   # Tuesday
        trading_day, current_date = 0, base_date

        while trading_day < days:
            if current_date.weekday() < 5:   # Mon–Fri
                for h, m in [
                    (9,15),(9,45),(10,15),(10,45),(11,15),(11,45),
                    (12,15),(12,45),(13,15),(13,45),(14,15),(14,45),(15,15)
                ]:
                    bars.append(
                        current_date.replace(hour=h, minute=m, second=0, microsecond=0)
                    )
                trading_day += 1
            current_date += pd.Timedelta(days=1)

        np.random.seed(42)
        n = len(bars)
        close  = 1000 + np.cumsum(np.random.randn(n) * 2)
        open_  = close + np.random.randn(n) * 0.5
        high   = np.maximum(open_, close) + np.abs(np.random.randn(n) * 0.5)
        low    = np.minimum(open_, close) - np.abs(np.random.randn(n) * 0.5)
        volume = np.random.randint(10_000, 500_000, n).astype(float)

        return pd.DataFrame({
            "datetime":  pd.DatetimeIndex(bars).tz_convert("UTC"),
            "open":      open_,
            "high":      high,
            "low":       low,
            "close":     close,
            "adj_close": close * 0.98,
            "volume":    volume,
            "symbol":    symbol,
        })

    df = pd.concat(
        [make_nse_30min("RELIANCE_NS", days=22),   # ~1 month
         make_nse_30min("TCS_NS",      days=22)],
        ignore_index=True,
    )

    print(f"\nInput  : {len(df)} rows | 2 symbols | base=30min")
    print(f"Range  : {df['datetime'].min()} -> {df['datetime'].max()}\n")

    rs = NSEResampler(df, symbol="RELIANCE_NS")
    print(rs, "\n")

    # resample to every supported TF and print a summary row
    print(f"  {'TF':>5}  | {'Bars':>5} | {'First bar (IST)':^22} | O        H        L        C")
    print("  " + "-" * 80)
    for tf in ["30min", "1H", "2H", "4H", "1D", "1W", "1M"]:
        out = rs.resample(tf)
        first     = out.iloc[0]
        first_ist = out.index.tz_convert(IST)[0].strftime("%Y-%m-%d %H:%M")
        print(
            f"  {tf:>5}  | {len(out):>5} | {first_ist:^22} | "
            f"{first['open']:>7.2f}  {first['high']:>7.2f}  "
            f"{first['low']:>7.2f}  {first['close']:>7.2f}"
        )

    # verify 1H bucket alignment
    print("\n  1H bar timestamps for day 1 (IST):")
    df_1h = rs.resample("1H")
    day1 = df_1h.index.tz_convert(IST)
    day1_bars = day1[day1.date == day1.date[0]]
    for ts in day1_bars:
        print(f"    {ts.strftime('%H:%M IST')}")
    # expected: 09:15 10:15 11:15 12:15 13:15 14:15 15:15

    # 4H bucket check
    print("\n  4H bar timestamps for day 1 (IST):")
    df_4h = rs.resample("4H")
    day1_4h = df_4h.index.tz_convert(IST)
    day1_4h_bars = day1_4h[day1_4h.date == day1_4h.date[0]]
    for ts in day1_4h_bars:
        print(f"    {ts.strftime('%H:%M IST')}")
    # expected: 09:15  13:15  (15:15 bar is its own bucket)

    # multi-symbol batch
    print("\n  convert_multi_symbol (30min -> 1D):")
    daily_all = NSEResampler.convert_multi_symbol(df, target_tf="1D")
    print(daily_all.groupby("symbol").size().rename("daily_bars"))

    # error guard
    print("\n  Error guard:")
    try:
        rs.resample("5min")
    except ValueError as e:
        print(f"    Caught -> {e}")

    print("\n=== All tests passed ===\n")

    # "from resampler import NSEResampler" use this to call the resampler(the freq changer)