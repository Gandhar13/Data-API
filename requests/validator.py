import pandas as pd
from zoneinfo import ZoneInfo
from models import DataRequest, UnavailableInfo

IST = ZoneInfo("Asia/Kolkata")


def check_empty(symbol: str, avail_from, avail_to) -> UnavailableInfo:
    """
    Called when load_symbol returns no rows and no unavailable_reason.
    Distinguishes between symbol completely missing vs range not available.
    avail_from/avail_to come directly from loader, no extra parquet read needed.
    """
    if avail_from is None:
        return UnavailableInfo(
            symbol = symbol,
            reason = f"no data found for {symbol}, use fetch service to populate data"
        )

    return UnavailableInfo(
        symbol       = symbol,
        reason       = (
            f"requested range not available for {symbol}, "
            f"available data is from {avail_from.tz_convert(IST).date()} to {avail_to.tz_convert(IST).date()}, "
            f"use fetch service to get missing data"
        )
    )


def check_range_coverage(symbol: str, avail_from, avail_to, req: DataRequest) -> UnavailableInfo | None:
    """
    Called after resampling for formats 1 and 3 (date-based, no bars).
    Compares IST dates only to avoid false gaps from time-of-day differences.
    Returns UnavailableInfo if data doesn't cover the requested range, else None.
    """
    if req.bars:
        return None  # bars-based formats handled in router after resampling

    actual_from_ist = avail_from.tz_convert(IST).date()
    actual_to_ist   = avail_to.tz_convert(IST).date()

    gaps = []
    if req.date_from:
        requested_from_ist = req.date_from.astimezone(IST).date()
        if actual_from_ist > requested_from_ist:
            gaps.append(f"data starts at {actual_from_ist} IST, requested from {requested_from_ist} IST")

    if req.date_to:
        requested_to_ist = req.date_to.astimezone(IST).date()
        if actual_to_ist < requested_to_ist:
            gaps.append(f"data ends at {actual_to_ist} IST, requested to {requested_to_ist} IST")

    if gaps:
        return UnavailableInfo(
            symbol       = symbol,
            reason       = (
                f"partial data for {symbol}: " + "; ".join(gaps) +
                ", use fetch service to get missing data"
            )
        )

    return None