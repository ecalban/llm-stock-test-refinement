"""Causal long/cash strategies sharing one execution and cost engine."""

import pandas as pd

from prediction.data import validate_prices
from prediction.indicators import sma, _window

STRATEGIES = ("ml", "buy_hold", "sma")


def strategy_signals(
    prices: pd.DataFrame, strategy: str, *, ml_actions: pd.Series | None = None,
    fast: int = 5, slow: int = 20,
) -> pd.Series:
    """Signals at close, with execution lag applied only by the backtest engine.

    SMA targets long whenever fast > slow, otherwise cash. Its first usable
    row can buy even if the crossover occurred before the evaluation period.
    """
    data = validate_prices(prices)
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if strategy == "ml":
        if ml_actions is None or not ml_actions.index.equals(data.index):
            raise ValueError("ML actions must match the evaluation dates")
        from prediction.signals import target_positions
        target_positions(ml_actions)
        return ml_actions.rename("action").copy()
    result = pd.Series("hold", index=data.index, name="action")
    if strategy == "buy_hold":
        result.iloc[0] = "buy"
        return result
    _window(fast)
    _window(slow)
    if fast >= slow:
        raise ValueError("SMA fast window must be smaller than slow window")
    short, long = sma(data.adj_close, fast), sma(data.adj_close, slow)
    ready = short.notna() & long.notna()
    result.loc[ready & (short > long)] = "buy"
    result.loc[ready & (short <= long)] = "sell"
    return result
