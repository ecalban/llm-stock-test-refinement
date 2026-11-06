"""Causal technical indicators. Warm-up observations are NaN, never backfilled."""

from numbers import Integral

import numpy as np
import pandas as pd


def _window(window: int) -> None:
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("window must be a positive integer")


def _prices(prices: pd.Series) -> pd.Series:
    values = pd.Series(prices, dtype=float)
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any():
        raise ValueError("prices must be finite and positive")
    return values


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average, available after ``window`` observations."""
    _window(window)
    return _prices(prices).rolling(window, min_periods=window).mean().rename("sma")


def ema(prices: pd.Series, window: int) -> pd.Series:
    """Exponential average seeded with the first price; no hidden warm-up."""
    _window(window)
    return _prices(prices).ewm(span=window, adjust=False).mean().rename("ema")


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI, seeded by the first ``window`` changes' simple averages.

    First valid value is at index ``window``. Flat prices give 50, gains without
    losses give 100, and losses without gains give 0.
    """
    _window(window)
    values = _prices(prices)
    result = pd.Series(np.nan, index=values.index, dtype=float, name="rsi")
    if len(values) <= window:
        return result
    delta = values.diff()
    gains, losses = delta.clip(lower=0), -delta.clip(upper=0)
    gain = gains.iloc[1 : window + 1].mean()
    loss = losses.iloc[1 : window + 1].mean()
    for i in range(window, len(values)):
        if i > window:
            gain = (gain * (window - 1) + gains.iloc[i]) / window
            loss = (loss * (window - 1) + losses.iloc[i]) / window
        if gain == 0 and loss == 0:
            result.iloc[i] = 50.0
        elif loss == 0:
            result.iloc[i] = 100.0
        else:
            result.iloc[i] = 100.0 - 100.0 / (1.0 + gain / loss)
    return result


def macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, EMA signal line and their difference (histogram)."""
    for window in (fast, slow, signal):
        _window(window)
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")
    line = ema(prices, fast) - ema(prices, slow)
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": signal_line, "histogram": line - signal_line})


def bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Moving mean plus/minus population standard deviation times ``num_std``."""
    _window(window)
    if not np.isfinite(num_std) or num_std <= 0:
        raise ValueError("num_std must be finite and positive")
    values = _prices(prices)
    middle = sma(values, window)
    spread = values.rolling(window).std(ddof=0) * num_std
    return pd.DataFrame({"lower": middle - spread, "middle": middle, "upper": middle + spread})
