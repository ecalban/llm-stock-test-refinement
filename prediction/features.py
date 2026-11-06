"""Features available at today's close and labels for next-open execution."""

import numpy as np
import pandas as pd

from prediction.data import adjusted_open, validate_prices
from prediction.indicators import bollinger_bands, ema, macd, rsi, sma


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build scale-independent features; retain NaN during the 20-bar warm-up.

    No future rows are read. Zero volume and flat bands have neutral ratios.
    Callers must remove warm-up rows before fitting or predicting.
    """
    data = validate_prices(prices)
    close = data["adj_close"]
    returns = close.pct_change(fill_method=None)
    bands = bollinger_bands(close)
    width = bands["upper"] - bands["lower"]
    band_position = (close - bands["lower"]) / width.replace(0, np.nan)
    band_position = band_position.mask(width == 0, 0.5)
    volume_mean = data["volume"].rolling(20).mean()
    volume_ratio = data["volume"] / volume_mean.replace(0, np.nan)
    volume_ratio = volume_ratio.mask(volume_mean == 0, 0.0)
    return pd.DataFrame({
        "return_1": returns,
        "return_5": close.pct_change(5, fill_method=None),
        "sma_ratio": sma(close, 5) / sma(close, 20) - 1,
        "ema_ratio": ema(close, 12) / ema(close, 26) - 1,
        "rsi_14": rsi(close) / 100,
        "macd_histogram": macd(close)["histogram"] / close,
        "band_position": band_position,
        "volatility_20": returns.rolling(20).std(ddof=0),
        "volume_ratio": volume_ratio,
    }, index=data.index)


def build_target(prices: pd.DataFrame) -> pd.Series:
    """1 when adjusted open[t+2] > open[t+1], else 0; last two labels are NaN.

    A close[t] signal executes at open[t+1]. Its label is only known at
    open[t+2], so walk-forward training must purge labels not yet observable.
    """
    opening = adjusted_open(validate_prices(prices))
    future_return = opening.shift(-2) / opening.shift(-1) - 1
    return (future_return > 0).astype(float).where(future_return.notna()).rename("target")


def build_class_target(prices: pd.DataFrame, neutral_band: float = 0.003) -> pd.Series:
    """Next tradable open-to-open class: buy above +band, sell below -band.

    The inclusive neutral interval is hold. The configurable default 0.3%
    is a baseline label definition, not an optimized trading parameter.
    Last two outcomes remain unknown. Keep this band fixed across all folds.
    """
    if not np.isfinite(neutral_band) or not 0 <= neutral_band < 1:
        raise ValueError("neutral_band must be finite and in [0, 1)")
    opening = adjusted_open(validate_prices(prices))
    returns = opening.shift(-2) / opening.shift(-1) - 1
    labels = pd.Series("hold", index=returns.index, dtype=object, name="actual_class")
    # Tolerance only handles floating point error at the inclusive boundaries.
    labels.loc[(returns > neutral_band) & ~np.isclose(returns, neutral_band, rtol=0, atol=1e-12)] = "buy"
    labels.loc[(returns < -neutral_band) & ~np.isclose(returns, -neutral_band, rtol=0, atol=1e-12)] = "sell"
    return labels.where(returns.notna(), None)
