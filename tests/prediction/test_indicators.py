import numpy as np
import pandas as pd
import pytest

from prediction.indicators import bollinger_bands, ema, macd, rsi, sma


def test_sma_exact_values_and_warmup():
    result = sma(pd.Series([1, 2, 3, 4]), 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == [2, 3]


def test_ema_uses_first_price_seed():
    assert ema(pd.Series([1, 2, 3, 4]), 3).tolist() == [1, 1.5, 2.25, 3.125]


@pytest.mark.parametrize("function", [sma, ema, rsi, bollinger_bands])
@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_reject_invalid_window(function, window):
    with pytest.raises(ValueError, match="positive integer"):
        function(pd.Series([1, 2, 3]), window)


@pytest.mark.parametrize("values,expected", [([1, 2, 3, 4], 100), ([4, 3, 2, 1], 0), ([2, 2, 2, 2], 50)])
def test_rsi_boundaries(values, expected):
    result = rsi(pd.Series(values), 3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == expected


def test_rsi_wilder_seed_and_smoothing():
    # Seed gains=2/3, losses=1/3; next gain=(2*(2/3)+2)/3=10/9,
    # next loss=(2*(1/3)+0)/3=2/9 -> RSI=100-100/6.
    result = rsi(pd.Series([1, 2, 3, 2, 4]), 3)
    assert result.iloc[3] == pytest.approx(100 - 100 / 3)
    assert result.iloc[4] == pytest.approx(100 - 100 / 6)


def test_macd_exact_values():
    result = macd(pd.Series([1, 2, 3]), fast=1, slow=3, signal=2)
    assert result.macd.tolist() == [0, 0.5, 0.75]
    assert result.signal.tolist() == pytest.approx([0, 1 / 3, 11 / 18])
    pd.testing.assert_series_equal(result.histogram, result.macd - result.signal, check_names=False)


def test_macd_requires_fast_smaller_than_slow():
    with pytest.raises(ValueError):
        macd(pd.Series([1, 2]), fast=3, slow=3)


def test_bollinger_population_standard_deviation():
    result = bollinger_bands(pd.Series([1, 2, 3]), window=3, num_std=2)
    assert result.iloc[:2].isna().all().all()
    assert result.iloc[-1].tolist() == pytest.approx([2 - 2 * np.sqrt(2 / 3), 2, 2 + 2 * np.sqrt(2 / 3)])


@pytest.mark.parametrize("num_std", [0, -1, np.inf, np.nan])
def test_bands_reject_invalid_deviation(num_std):
    with pytest.raises(ValueError):
        bollinger_bands(pd.Series([1, 2]), num_std=num_std)


@pytest.mark.parametrize("function", [sma, ema, rsi, bollinger_bands])
def test_empty_indicators(function):
    assert function(pd.Series([], dtype=float), 3).empty


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan])
def test_invalid_price(value):
    with pytest.raises(ValueError):
        sma(pd.Series([1, value]), 2)


def test_indicators_are_causal(prices):
    for function in (sma, ema, rsi, bollinger_bands):
        complete = function(prices.adj_close, 14).iloc[:60]
        prefix = function(prices.adj_close.iloc[:60], 14)
        if isinstance(complete, pd.Series):
            pd.testing.assert_series_equal(complete, prefix)
        else:
            pd.testing.assert_frame_equal(complete, prefix)
