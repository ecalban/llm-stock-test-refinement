import numpy as np
import pandas as pd
import pytest

from prediction.backtest import run_backtest
from prediction.signals import generate_signals, target_positions


def bars(openings):
    values = np.array(openings, dtype=float)
    return pd.DataFrame({
        "open": values, "high": values, "low": values, "close": values,
        "adj_close": values, "volume": 100,
    }, index=pd.bdate_range("2024-01-01", periods=len(values), name="date"))


def actions(data, values):
    return pd.Series(values, index=data.index)


def test_probability_threshold_boundaries():
    probabilities = pd.Series([0, 0.45, 0.5, 0.55, 1])
    assert generate_signals(probabilities).tolist() == ["sell", "sell", "hold", "buy", "buy"]


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
def test_reject_invalid_probabilities(value):
    with pytest.raises(ValueError):
        generate_signals(pd.Series([value]))


@pytest.mark.parametrize("sell,buy", [(0.5, 0.5), (0.7, 0.6), (-0.1, 0.9), (0.1, 1.1), (np.nan, 0.6)])
def test_reject_invalid_thresholds(sell, buy):
    with pytest.raises(ValueError):
        generate_signals(pd.Series([0.5]), sell_threshold=sell, buy_threshold=buy)


def test_hold_keeps_position_and_repeated_orders_do_not_leverage():
    result = target_positions(pd.Series(["hold", "sell", "buy", "buy", "hold", "sell", "hold"]))
    assert result.tolist() == [0, 0, 1, 1, 1, 0, 0]


def test_next_open_execution_ignores_same_day_rally():
    data = bars([10, 100, 110, 121])
    result = run_backtest(data, actions(data, ["buy", "hold", "hold", "hold"]),
                          initial_cash=100, commission_bps=0, slippage_bps=0)
    assert result.daily.position.tolist() == [0, 1, 1, 0]
    assert result.daily.equity.tolist() == pytest.approx([100, 110, 121, 121])
    assert result.metrics["total_return"] == pytest.approx(0.21)
    assert result.metrics["buy_hold_return"] == pytest.approx(0.21)
    assert result.metrics["trades"] == 2


def test_entry_exit_costs_exact_compounding_and_repeated_buy():
    data = bars([10, 10, 11, 12])
    result = run_backtest(data, actions(data, ["buy", "buy", "buy", "hold"]),
                          initial_cash=100, commission_bps=60, slippage_bps=40)
    assert result.daily.turnover.tolist() == [0, 1, 0, 1]
    assert result.daily.cost_fraction.tolist() == [0, 0.01, 0, 0.01]
    assert result.daily.equity.iloc[-1] == pytest.approx(100 * 0.99 * 1.2 * 0.99)


def test_sell_executes_next_day_and_cash_avoids_later_loss():
    data = bars([100, 100, 110, 55, 55])
    result = run_backtest(data, actions(data, ["buy", "sell", "hold", "hold", "hold"]),
                          commission_bps=0, slippage_bps=0)
    assert result.daily.position.tolist() == [0, 1, 0, 0, 0]
    assert result.metrics["total_return"] == pytest.approx(0.1)


def test_missing_signal_dates_mean_hold():
    data = bars([100, 100, 110, 121])
    result = run_backtest(data, pd.Series(["buy"], index=data.index[:1]),
                          commission_bps=0, slippage_bps=0)
    assert result.metrics["total_return"] == pytest.approx(0.21)


def test_late_buy_cannot_create_unobservable_trade():
    data = bars([100, 110, 120])
    result = run_backtest(data, actions(data, ["hold", "buy", "buy"]))
    assert result.metrics["total_return"] == 0
    assert result.metrics["trades"] == 0


def test_drawdown_includes_initial_loss():
    data = bars([100, 100, 90])
    result = run_backtest(data, actions(data, ["buy", "hold", "hold"]),
                          commission_bps=0, slippage_bps=0)
    assert result.metrics["max_drawdown"] == pytest.approx(-0.1)


def test_cash_only_has_zero_risk_metrics(prices):
    result = run_backtest(prices, actions(prices, ["hold"] * len(prices)))
    assert result.metrics["total_return"] == 0
    assert result.metrics["sharpe_ratio"] == 0
    assert result.metrics["max_drawdown"] == 0
    assert result.metrics["exposure"] == 0


def test_risk_metrics_count_intervals_and_merge_terminal_cost():
    data = bars([100, 100, 110])
    result = run_backtest(data, actions(data, ["buy", "hold", "hold"]),
                          commission_bps=100, slippage_bps=0)
    # Two time intervals: cash return 0, then entry + 10% gain + exit.
    last_return = 0.99 * 1.1 * 0.99 - 1
    expected_std = np.std([0, last_return], ddof=0)
    assert result.metrics["annualized_volatility"] == pytest.approx(expected_std * np.sqrt(252))
    assert result.metrics["sharpe_ratio"] == pytest.approx((last_return / 2) / expected_std * np.sqrt(252))
    assert result.metrics["annualized_return"] == pytest.approx((1 + last_return) ** 126 - 1)


def test_adjusted_prices_avoid_false_split_loss():
    data = bars([100, 100, 50, 50])
    data["adj_close"] = 50.0
    result = run_backtest(data, actions(data, ["buy", "hold", "hold", "hold"]),
                          commission_bps=0, slippage_bps=0)
    assert result.metrics["total_return"] == 0


@pytest.mark.parametrize("kwargs", [
    {"initial_cash": 0}, {"initial_cash": np.inf}, {"commission_bps": -1},
    {"commission_bps": np.nan}, {"slippage_bps": -1}, {"slippage_bps": 10000},
    {"commission_bps": 6000, "slippage_bps": 4000}, {"trading_days": 0},
])
def test_reject_invalid_backtest_parameters(prices, kwargs):
    with pytest.raises(ValueError):
        run_backtest(prices, actions(prices, ["hold"] * len(prices)), **kwargs)


@pytest.mark.parametrize("kind", ["unknown_action", "duplicate_date", "outside_date", "non_date_index", "missing_action"])
def test_reject_invalid_backtest_actions(prices, kind):
    series = pd.Series(["buy"], index=prices.index[:1])
    if kind == "unknown_action":
        series.iloc[0] = "short"
    elif kind == "duplicate_date":
        series = pd.concat([series, series])
    elif kind == "outside_date":
        series.index = series.index - pd.Timedelta(days=10)
    elif kind == "non_date_index":
        series = series.reset_index(drop=True)
    else:
        series.iloc[0] = None
    with pytest.raises(ValueError):
        run_backtest(prices, series)


def test_reject_too_short_backtest(prices):
    with pytest.raises(ValueError, match="three trading days"):
        run_backtest(prices.iloc[:2], pd.Series(["buy"], index=prices.index[:1]))
