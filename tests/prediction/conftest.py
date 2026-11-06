import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    """A deterministic single-stock series; no network or database needed."""
    rng = np.random.default_rng(42)
    opening = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, 160)))
    close = opening * np.exp(rng.normal(0, 0.006, len(opening)))
    return pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, close) * 1.01,
        "low": np.minimum(opening, close) * 0.99,
        "close": close,
        "adj_close": close,
        "volume": rng.integers(1000, 10000, len(opening)),
    }, index=pd.bdate_range("2024-01-01", periods=len(opening), name="date"))
