import numpy as np
import pandas as pd
import pytest

from prediction.backtest import drawdown_statistics, run_backtest
from prediction.pipeline import run_strategy_backtest
from prediction.strategies import strategy_signals


def bars(values):
    return pd.DataFrame({**{c: values for c in ("open", "high", "low", "close", "adj_close")},
                         "volume": 100},
                        index=pd.bdate_range("2024-01-01", periods=len(values), name="date"))


def test_ledger_matches_fractional_share_cash_and_fees():
    data = bars([100, 100, 110, 110])
    report = run_backtest(data, pd.Series(["buy", "sell", "hold", "hold"], index=data.index),
                          initial_cash=100, commission_bps=60, slippage_bps=40)
    trade = report.trades.iloc[0]
    assert trade.entry_date == data.index[1] and trade.exit_date == data.index[2]
    assert trade.quantity == pytest.approx(0.99)
    assert trade.entry_cost == pytest.approx(1)
    assert trade.exit_cost == pytest.approx(1.089)
    assert trade.capital_after == pytest.approx(107.811)
    assert trade.net_return == pytest.approx(0.07811)
    assert trade.total_cost == pytest.approx(2.089)
    assert trade.duration_sessions == 1
    assert trade.direction == "long"
    assert report.metrics["order_count"] == 2
    assert report.metrics["closed_trade_count"] == 1
    assert report.metrics["win_rate"] == 1
    assert report.metrics["total_cost"] == pytest.approx(report.orders.total_cost.sum())
    assert report.daily.equity.iloc[-1] == pytest.approx(trade.capital_after)


def test_compounded_closed_positions_reconcile_final_equity():
    data = bars([100, 100, 110, 100, 90, 95, 100])
    actions = pd.Series(["buy", "sell", "buy", "sell", "buy", "hold", "hold"], index=data.index)
    result = run_backtest(data, actions, initial_cash=100)
    assert len(result.trades) == 3
    assert result.metrics["win_rate"] == pytest.approx(2 / 3)
    assert 100 * (1 + result.trades.net_return).prod() == pytest.approx(result.daily.equity.iloc[-1])
    assert result.trades.pnl.sum() == pytest.approx(result.daily.equity.iloc[-1] - 100)
    assert result.trades.total_cost.sum() == pytest.approx(result.orders.total_cost.sum())
    assert result.trades.exit_date.iloc[-1] == data.index[-1]


@pytest.mark.parametrize("equity,recovery,longest", [
    ([90, 80, 90, 100, 110, 105], 2, 3),
    ([100, 90, 80], None, 2),
    ([100, 100, 100], 0, 0),
    ([110, 100, 105, 110], 2, 2),
])
def test_drawdown_recovery_is_measured_not_estimated(equity, recovery, longest):
    stats = drawdown_statistics(pd.Series(equity), 100)
    assert stats == {"recovery_sessions": recovery, "max_underwater_sessions": longest}


def test_bist100_is_independent_of_stock_buy_hold():
    data = bars([100, 100, 110, 120])
    index = pd.Series([1000, 1000, 900, 800], index=data.index)
    result = run_backtest(data, pd.Series("buy", index=data.index),
                          commission_bps=0, slippage_bps=0, benchmark_open=index)
    assert result.metrics["buy_hold_return"] == pytest.approx(0.2)
    assert result.metrics["bist100_return"] == pytest.approx(-0.2)
    assert result.daily.bist100_equity.iloc[-1] == pytest.approx(80000)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "zero"])
def test_benchmark_must_cover_exact_evaluation_dates(prices, kind):
    benchmark = prices.open.copy()
    if kind == "missing":
        benchmark = benchmark.iloc[1:]
    elif kind == "duplicate":
        benchmark = pd.concat([benchmark, benchmark.iloc[:1]])
    else:
        benchmark.iloc[5] = 0
    with pytest.raises(ValueError):
        run_backtest(prices, pd.Series("hold", index=prices.index), benchmark_open=benchmark)


def test_absent_benchmark_and_empty_ledger_are_explicit(prices):
    result = run_backtest(prices, pd.Series("hold", index=prices.index))
    assert result.metrics["bist100_return"] is None
    assert result.metrics["win_rate"] is None
    assert result.trades.empty and result.orders.empty
    assert "entry_date" in result.trades.columns


def test_buy_hold_strategy_uses_same_engine_and_benchmark(prices):
    result = run_strategy_backtest(prices, strategy="buy_hold", start=prices.index[130])
    assert result.daily.index.equals(prices.index[130:])
    assert result.metrics["total_return"] == pytest.approx(result.metrics["buy_hold_return"])
    assert result.metrics["order_count"] == 2 and result.metrics["closed_trade_count"] == 1


def test_sma_strategy_retains_warmup_before_requested_period():
    data = bars(list(range(100, 150)))
    result = run_strategy_backtest(data, strategy="sma", start=data.index[30])
    assert result.daily.index[0] == data.index[30]
    assert result.daily.position.iloc[1] == 1
    assert result.metrics["closed_trade_count"] == 1


def test_sma_signals_are_causal_and_have_no_execution_lag():
    data = bars([100, 101, 102, 103, 90, 80, 75, 90, 100])
    actions = strategy_signals(data, "sma", fast=2, slow=3)
    assert actions.iloc[:2].eq("hold").all()
    assert actions.iloc[2] == "buy" and actions.iloc[4] == "sell"
    pd.testing.assert_series_equal(actions.iloc[:6], strategy_signals(data.iloc[:6], "sma", fast=2, slow=3))


@pytest.mark.parametrize("kwargs", [{"fast": 20, "slow": 20}, {"fast": 0}, {"strategy": "short"}])
def test_invalid_strategy_settings(prices, kwargs):
    options = {"strategy": "sma", **kwargs}
    with pytest.raises(ValueError):
        strategy_signals(prices, **options)
