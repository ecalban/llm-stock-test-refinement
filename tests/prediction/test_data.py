import numpy as np
import pandas as pd
import pytest

from prediction.data import adjusted_open, load_prices, normalize_symbol, validate_prices


@pytest.mark.parametrize("ticker", ["thyao", " THYAO.IS ", "ThYaO.is"])
def test_normalize_symbol(ticker):
    assert normalize_symbol(ticker) == "THYAO.IS"


@pytest.mark.parametrize("ticker", ["", ".IS", "THYAO.US", "THY AO", "İŞCTR", "../THYAO"])
def test_reject_invalid_symbols(ticker):
    with pytest.raises(ValueError):
        normalize_symbol(ticker)


def test_load_sorts_normalizes_columns_and_attaches_symbol(prices, tmp_path):
    csv = prices.iloc[::-1].reset_index().rename(columns=str.title).drop(columns="Adj_Close")
    path = tmp_path / "prices.csv"
    csv.to_csv(path, index=False)
    result = load_prices(path, symbol="thyao")
    assert result.index.equals(prices.index)
    assert result["symbol"].eq("THYAO.IS").all()
    pd.testing.assert_series_equal(result.adj_close, result.close, check_names=False)


def test_validate_does_not_modify_input(prices):
    original = prices.copy(deep=True)
    validate_prices(prices)
    pd.testing.assert_frame_equal(prices, original)


@pytest.mark.parametrize("column,value", [
    ("open", 0), ("adj_close", -1), ("close", np.nan), ("volume", np.inf),
    ("volume", -1), ("volume", 1.5), ("high", 1), ("low", 10000),
])
def test_reject_invalid_bar(prices, column, value):
    data = prices.astype({column: float})
    data.iloc[0, data.columns.get_loc(column)] = value
    with pytest.raises(ValueError):
        validate_prices(data)


def test_reject_duplicate_daily_dates(prices):
    extra = prices.iloc[[0]].copy()
    extra.index = extra.index + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="duplicate trading dates"):
        validate_prices(pd.concat([prices, extra]))


def test_reject_duplicate_normalized_columns(prices):
    data = prices.copy()
    data["Open"] = data["open"]
    with pytest.raises(ValueError, match="duplicate column"):
        validate_prices(data)


@pytest.mark.parametrize("kind", ["empty", "missing_column", "missing_date", "missing_date_value"])
def test_reject_incomplete_data(prices, kind):
    if kind == "empty":
        data = prices.iloc[:0]
    elif kind == "missing_column":
        data = prices.drop(columns="volume")
    elif kind == "missing_date":
        data = prices.reset_index(drop=True)
    else:
        data = prices.reset_index()
        data.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError):
        validate_prices(data)


def test_reject_multiple_stocks(prices):
    prices["symbol"] = ["THYAO", *(["ASELS"] * (len(prices) - 1))]
    with pytest.raises(ValueError, match="one stock"):
        validate_prices(prices)


def test_reject_symbol_mismatch(prices, tmp_path):
    prices["symbol"] = "THYAO"
    path = tmp_path / "bars.csv"
    prices.to_csv(path)
    with pytest.raises(ValueError, match="does not match"):
        load_prices(path, symbol="ASELS")


def test_adjusted_open_removes_split_jump(prices):
    data = prices.iloc[:4].copy()
    data["open"] = [100, 100, 50, 50]
    data["close"] = data["open"]
    data["high"] = data["open"]
    data["low"] = data["open"]
    data["adj_close"] = 50.0
    assert adjusted_open(validate_prices(data)).tolist() == [50, 50, 50, 50]
