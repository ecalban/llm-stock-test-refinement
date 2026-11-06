"""Turn direction probabilities into long/cash trading instructions."""

import numpy as np
import pandas as pd


def generate_signals(
    probabilities: pd.Series, *, buy_threshold: float = 0.55, sell_threshold: float = 0.45
) -> pd.Series:
    """Buy at/above the upper threshold, sell at/below the lower, otherwise hold.

    ``hold`` preserves the existing position, ``buy`` targets long and ``sell``
    targets cash. Repeated instructions never add leverage or short positions.
    """
    if not 0 <= sell_threshold < buy_threshold <= 1:
        raise ValueError("thresholds must satisfy 0 <= sell < buy <= 1")
    values = pd.Series(probabilities, dtype=float)
    if not np.isfinite(values.to_numpy()).all() or not values.between(0, 1).all():
        raise ValueError("probabilities must be finite and between 0 and 1")
    actions = pd.Series("hold", index=values.index, name="action")
    actions.loc[values >= buy_threshold] = "buy"
    actions.loc[values <= sell_threshold] = "sell"
    return actions


def target_positions(actions: pd.Series) -> pd.Series:
    """Long/cash target at signal close; execution still requires a one-bar lag."""
    if not actions.isin(["buy", "sell", "hold"]).all():
        raise ValueError("actions must contain only buy, sell or hold")
    return actions.map({"buy": 1.0, "sell": 0.0}).ffill().fillna(0.0).rename("target_position")
