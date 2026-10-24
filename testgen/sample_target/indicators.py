"""Small technical-indicator functions used as a practice target for the test layer."""


def sma(prices: list[float], window: int) -> list[float | None]:
    """Simple moving average. The first window-1 values are None."""
    if window <= 0:
        raise ValueError("window must be positive")
    result: list[float | None] = []
    for i in range(len(prices)):
        if i + 1 < window:
            result.append(None)
        else:
            result.append(sum(prices[i + 1 - window : i + 1]) / window)
    return result


def ema(prices: list[float], window: int) -> list[float]:
    """Exponential moving average seeded with the first price."""
    if window <= 0:
        raise ValueError("window must be positive")
    if not prices:
        return []
    alpha = 2 / (window + 1)
    result = [prices[0]]
    for price in prices[1:]:
        result.append(alpha * price + (1 - alpha) * result[-1])
    return result


def rsi(prices: list[float], window: int = 14) -> float | None:
    """Relative Strength Index of the last `window` price changes (0-100)."""
    if len(prices) <= window:
        return None
    changes = [prices[i] - prices[i - 1] for i in range(len(prices) - window, len(prices))]
    gains = sum(c for c in changes if c > 0)
    losses = -sum(c for c in changes if c < 0)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100 - 100 / (1 + rs)


def crossover_signal(short: list[float | None], long: list[float | None]) -> str:
    """'buy' when short crosses above long on the last step, 'sell' when it crosses below,
    otherwise 'hold'."""
    if len(short) < 2 or len(long) < 2:
        return "hold"
    prev_s, prev_l, cur_s, cur_l = short[-2], long[-2], short[-1], long[-1]
    if None in (prev_s, prev_l, cur_s, cur_l):
        return "hold"
    if prev_s <= prev_l and cur_s > cur_l:
        return "buy"
    if prev_s >= prev_l and cur_s < cur_l:
        return "sell"
    return "hold"
