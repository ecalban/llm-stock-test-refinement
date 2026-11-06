"""Load and validate daily OHLCV bars for a single stock; never fill missing prices."""

from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def normalize_symbol(symbol: str) -> str:
    """Return a BIST symbol such as ``THYAO.IS`` from ``thyao`` or ``THYAO.IS``."""
    value = symbol.strip().upper()
    if value.endswith(".IS"):
        value = value[:-3]
    if not value or not value.isascii() or not value.isalnum():
        raise ValueError("symbol must be a BIST ticker, e.g. THYAO or THYAO.IS")
    return value + ".IS"


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with a unique daily DatetimeIndex named ``date``.

    Accept a date column or a DatetimeIndex and case-insensitive OHLCV names.
    ``adj_close`` defaults to ``close`` for already-adjusted input. If supplied,
    it is used to adjust opening prices consistently with splits/dividends.
    Multi-stock input, invalid bars, missing values and duplicates are errors.
    """
    if frame.empty:
        raise ValueError("price data must not be empty")
    data = frame.copy()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("duplicate column names")
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"missing price columns: {', '.join(sorted(missing))}")
    if "date" in data:
        dates = pd.DatetimeIndex(pd.to_datetime(data.pop("date"), errors="raise"))
    elif isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    else:
        raise ValueError("a date column or DatetimeIndex is required")
    if dates.hasnans:
        raise ValueError("dates must not be missing")
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    data.index = dates.normalize().rename("date")
    if data.index.has_duplicates:
        raise ValueError("duplicate trading dates")
    if "symbol" in data:
        if data["symbol"].isna().any():
            raise ValueError("symbol must not be missing")
        data["symbol"] = data["symbol"].map(normalize_symbol)
        if data["symbol"].nunique() != 1:
            raise ValueError("price data must contain exactly one stock")
    if "adj_close" not in data:
        data["adj_close"] = data["close"]
    numeric = [*PRICE_COLUMNS, "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("prices and volume must be finite and non-missing")
    if (data[list(PRICE_COLUMNS)] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (data["volume"] < 0).any() or (data["volume"] % 1 != 0).any():
        raise ValueError("volume must be a non-negative integer")
    if (data["high"] < data[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be at least open, close and low")
    if (data["low"] > data[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be at most open, close and high")
    return data.sort_index()


def load_prices(path: str | Path, *, symbol: str | None = None) -> pd.DataFrame:
    """Load a local daily CSV; optionally attach/check its BIST ticker."""
    data = validate_prices(pd.read_csv(path))
    if symbol is not None:
        ticker = normalize_symbol(symbol)
        if "symbol" in data and data["symbol"].iloc[0] != ticker:
            raise ValueError("requested symbol does not match the CSV")
        data["symbol"] = ticker
    return data


def adjusted_open(prices: pd.DataFrame) -> pd.Series:
    """Opening price in the same adjusted units as ``adj_close``."""
    return (prices["open"] * prices["adj_close"] / prices["close"]).rename("adj_open")
