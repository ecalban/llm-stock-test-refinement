"""Long/cash backtest: close signals, next-open execution and per-side costs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediction.data import adjusted_open, validate_prices
from prediction.signals import target_positions


@dataclass(frozen=True)
class BacktestResult:
    daily: pd.DataFrame
    metrics: dict[str, float | int | None]
    trades: pd.DataFrame
    orders: pd.DataFrame


def drawdown_statistics(equity: pd.Series, initial_cash: float) -> dict:
    """Session counts: longest underwater episode and recovery from worst trough.

    No drawdown gives zero recovery. An unrecovered worst trough gives None,
    never an estimated duration. Input observations are end-of-period values.
    """
    values = np.r_[initial_cash, equity.to_numpy(dtype=float)]
    peaks = np.maximum.accumulate(values)
    drawdowns = values / peaks - 1
    trough = int(np.argmin(drawdowns))
    recovery = 0
    if drawdowns[trough] < 0:
        recovered = np.flatnonzero(values[trough + 1:] >= peaks[trough])
        recovery = int(recovered[0] + 1) if len(recovered) else None
    longest = current = 0
    for value, peak in zip(values[1:], peaks[1:]):
        current = current + 1 if value < peak else 0
        longest = max(longest, current)
    return {"recovery_sessions": recovery, "max_underwater_sessions": longest}


def _ledger(opening, positions, equity, initial_cash, commission_rate, slippage_rate):
    """Reconcile fractional-share entries/exits with the portfolio simulation."""
    trades, orders = [], []
    entry = None
    for i in np.flatnonzero(positions.diff().fillna(positions.iloc[0]).to_numpy()):
        price = float(opening.iloc[i])
        before = float(initial_cash if i == 0 else equity.iloc[i - 1])
        commission, slippage = before * commission_rate, before * slippage_rate
        fee = commission + slippage
        side = "buy" if positions.iloc[i] else "sell"
        quantity = (before - fee) / price if side == "buy" else entry["quantity"]
        orders.append({"date": opening.index[i], "side": side, "price": price,
                       "quantity": quantity, "commission": commission,
                       "slippage": slippage, "total_cost": fee,
                       "capital_before": before, "capital_after_cost": before - fee})
        if side == "buy":
            entry = {"entry_date": opening.index[i], "entry_price": price,
                     "quantity": quantity, "capital_before": before,
                     "entry_cost": fee, "entry_commission": commission,
                     "entry_slippage": slippage, "entry_index": i}
        else:
            after = before - fee
            trade = {key: value for key, value in entry.items() if key != "entry_index"}
            trade.update({"exit_date": opening.index[i], "exit_price": price,
                          "direction": "long", "exit_cost": fee,
                          "exit_commission": commission, "exit_slippage": slippage,
                          "total_cost": entry["entry_cost"] + fee,
                          "capital_after": after, "pnl": after - entry["capital_before"],
                          "gross_return": price / entry["entry_price"] - 1,
                          "net_return": after / entry["capital_before"] - 1,
                          "duration_sessions": int(i - entry["entry_index"]),
                          "duration_calendar_days": (opening.index[i] - entry["entry_date"]).days})
            trades.append(trade)
            entry = None
    trade_columns = ["entry_date", "entry_price", "quantity", "capital_before", "entry_cost",
                     "entry_commission", "entry_slippage", "exit_date", "exit_price", "direction",
                     "exit_cost", "exit_commission", "exit_slippage", "total_cost", "capital_after",
                     "pnl", "gross_return", "net_return", "duration_sessions", "duration_calendar_days"]
    order_columns = ["date", "side", "price", "quantity", "commission", "slippage",
                     "total_cost", "capital_before", "capital_after_cost"]
    return pd.DataFrame(trades, columns=trade_columns), pd.DataFrame(orders, columns=order_columns)


def run_backtest(
    prices: pd.DataFrame,
    actions: pd.Series,
    *,
    initial_cash: float = 100_000.0,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
    trading_days: int = 252,
    benchmark_open: pd.Series | None = None,
) -> BacktestResult:
    """Simulate fractional, fully invested long positions or cash (no shorting).

    A signal on day t sets position at open[t+1]; returns run from that open
    to the following open. ``hold`` preserves position. Charge commission plus
    slippage, in basis points, on entry and exit. Repeated buy/sell costs zero
    when position is unchanged. Liquidate at the final observed open and ignore
    instructions that would enter there, since no later price is available.

    The benchmark buys at the second observed open and liquidates at the last,
    with the same costs. Equity in row t includes the return to open[t+1]
    (except the final row, which records only the liquidation cost).
    """
    data = validate_prices(prices)
    if len(data) < 3:
        raise ValueError("backtest needs at least three trading days")
    if not np.isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("initial_cash must be finite and positive")
    costs = np.array([commission_bps, slippage_bps], dtype=float)
    if not np.isfinite(costs).all() or (costs < 0).any() or costs.sum() >= 10_000:
        raise ValueError("costs must be non-negative and total less than 10000 bps")
    if isinstance(trading_days, bool) or not isinstance(trading_days, int) or trading_days <= 0:
        raise ValueError("trading_days must be a positive integer")
    if not isinstance(actions.index, pd.DatetimeIndex) or actions.index.has_duplicates:
        raise ValueError("actions must have a unique DatetimeIndex")
    if not actions.index.isin(data.index).all():
        raise ValueError("action dates must occur in price data")
    target_positions(actions)  # validate before filling dates without instructions
    instructions = actions.reindex(data.index, fill_value="hold").rename("action")
    positions = target_positions(instructions).shift(1, fill_value=0.0).rename("position")
    positions.iloc[-1] = 0.0
    opening = adjusted_open(data)
    asset_returns = (opening.shift(-1) / opening - 1).fillna(0.0).rename("asset_return")
    cost_rate = float(costs.sum() / 10_000)
    turnover = positions.diff().abs().fillna(positions.iloc[0]).rename("turnover")
    transaction_cost = (turnover * cost_rate).rename("cost_fraction")
    net_returns = ((1 - transaction_cost) * (1 + positions * asset_returns) - 1).rename("net_return")
    equity = (initial_cash * (1 + net_returns).cumprod()).rename("equity")

    benchmark_position = pd.Series(1.0, index=data.index)
    benchmark_position.iloc[[0, -1]] = 0.0
    benchmark_turnover = benchmark_position.diff().abs().fillna(0.0)
    benchmark_returns = (1 - benchmark_turnover * cost_rate) * (1 + benchmark_position * asset_returns) - 1
    benchmark_equity = (initial_cash * (1 + benchmark_returns).cumprod()).rename("benchmark_equity")

    bist_equity = None
    if benchmark_open is not None:
        if not isinstance(benchmark_open.index, pd.DatetimeIndex) or benchmark_open.index.has_duplicates:
            raise ValueError("benchmark must have a unique DatetimeIndex")
        aligned = benchmark_open.reindex(data.index).astype(float)
        missing = aligned.index[aligned.isna()]
        if len(missing):
            dates = ", ".join(str(day.date()) for day in missing[:5])
            raise ValueError(f"benchmark is missing evaluation dates: {dates}; use a complete source or a narrower period")
        if not np.isfinite(aligned.to_numpy()).all() or (aligned <= 0).any():
            raise ValueError("benchmark must cover every evaluation date with positive prices")
        bist_return = (aligned.shift(-1) / aligned - 1).fillna(0.0)
        bist_net = (1 - benchmark_turnover * cost_rate) * (1 + benchmark_position * bist_return) - 1
        bist_equity = (initial_cash * (1 + bist_net).cumprod()).rename("bist100_equity")

    trades, orders = _ledger(opening, positions, equity, initial_cash,
                             float(commission_bps / 10_000), float(slippage_bps / 10_000))

    # Include initial capital as a peak so an entry cost/loss is a drawdown.
    drawdown = equity / equity.cummax().clip(lower=initial_cash) - 1
    # The last row is an exit cost at the same open where the preceding
    # interval ends, not an additional trading period. Fold it into that
    # interval for volatility/Sharpe, matching the annualization period count.
    period_returns = net_returns.iloc[:-1].copy()
    period_returns.iloc[-1] = (1 + period_returns.iloc[-1]) * (1 + net_returns.iloc[-1]) - 1
    std = float(period_returns.std(ddof=0))
    total_return = float(equity.iloc[-1] / initial_cash - 1)
    metrics = {
        "total_return": total_return,
        "annualized_return": float((1 + total_return) ** (trading_days / (len(data) - 1)) - 1),
        "annualized_volatility": std * float(np.sqrt(trading_days)),
        "sharpe_ratio": float(period_returns.mean() / std * np.sqrt(trading_days)) if std > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "buy_hold_return": float(benchmark_equity.iloc[-1] / initial_cash - 1),
        "trades": int(turnover.sum()),
        "order_count": len(orders),
        "closed_trade_count": len(trades),
        "win_rate": float(trades.net_return.gt(0).mean()) if len(trades) else None,
        "total_cost": float(orders.total_cost.sum()),
        "bist100_return": float(bist_equity.iloc[-1] / initial_cash - 1) if bist_equity is not None else None,
        "exposure": float(positions.iloc[:-1].mean()),
    }
    interval_equity = initial_cash * (1 + period_returns).cumprod()
    metrics.update(drawdown_statistics(interval_equity, initial_cash))
    daily = pd.concat([
        opening, instructions, positions, asset_returns, turnover, transaction_cost,
        net_returns, equity, benchmark_equity, drawdown.rename("drawdown"),
    ], axis=1)
    daily["valuation_date"] = pd.Series(data.index, index=data.index).shift(-1).fillna(data.index[-1])
    if bist_equity is not None:
        daily["bist100_equity"] = bist_equity
    return BacktestResult(daily=daily, metrics=metrics, trades=trades, orders=orders)
