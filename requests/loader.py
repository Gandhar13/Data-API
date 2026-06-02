import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
from config import DATA_FILE



def load_symbol(symbol: str, date_from=None, date_to=None, bars=None):
    """
    Load 30min bars for a symbol into a DataFrame, then apply date filters.
    Bar count validation and trimming is handled by the caller after resampling.
    date_from and date_to are always UTC-aware datetimes (converted from IST in models.py).

    Parameters
    ----------
    symbol    : ticker string
    date_from : optional UTC datetime lower bound (inclusive)
    date_to   : optional UTC datetime upper bound (inclusive)
    bars      : whether a bars-based format is requested (used to decide filter strategy)

    Returns
    -------
    (DataFrame | None, reason: str | None, avail_from: Timestamp | None, avail_to: Timestamp | None)
    - DataFrame is None, avail_from/to are None when symbol is completely missing
    - DataFrame is None with reason when no data found at requested boundary
    - DataFrame with rows when data found
    """
    # load full symbol in one read
    table = pq.read_table(DATA_FILE, filters=[("symbol", "=", symbol)])

    if table.num_rows == 0:
        return None, None, None, None  # symbol completely missing, check_empty handles this

    df = table.to_pandas()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # capture full available range before any filtering
    avail_from = df["datetime"].min()
    avail_to   = df["datetime"].max()

    if bars:
        if date_to and not date_from:
            # format 5: all bars up to date_to, caller trims to N after resampling
            df = df[df["datetime"] <= pd.Timestamp(date_to)]
            if df.empty:
                return None, f"no data found at or before {date_to} for {symbol}", avail_from, avail_to

        elif date_from:
            # format 4: all bars from date_from, caller trims to N after resampling
            df = df[df["datetime"] >= pd.Timestamp(date_from)]
            if df.empty:
                return None, f"no data found at or after {date_from} for {symbol}", avail_from, avail_to

        # format 2: no date filter, load everything, caller takes last N after resampling

    else:
        # formats 1 and 3: datetime range filters only
        if date_from:
            df = df[df["datetime"] >= pd.Timestamp(date_from)]
        if date_to:
            df = df[df["datetime"] <= pd.Timestamp(date_to)]

    if df.empty:
        return None, None, avail_from, avail_to

    return df.reset_index(drop=True), None, avail_from, avail_to