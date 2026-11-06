"""Three-class walk-forward prediction, honest evaluation and strategy reports."""

from dataclasses import dataclass

import pandas as pd

from prediction.backtest import BacktestResult, run_backtest
from prediction.data import adjusted_open, validate_prices
from prediction.evaluation import evaluate_predictions
from prediction.features import build_class_target, build_features
from prediction.model import CLASSES, SignalModel
from prediction.strategies import strategy_signals


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame


@dataclass(frozen=True)
class PredictionResult:
    predictions: pd.DataFrame
    folds: pd.DataFrame
    backtest: BacktestResult
    classification_metrics: dict
    config: dict


def _period(data, start=None, end=None):
    start = pd.Timestamp(start).normalize() if start is not None else None
    end = pd.Timestamp(end).normalize() if end is not None else None
    if (start is not None and pd.isna(start)) or (end is not None and pd.isna(end)):
        raise ValueError("start/end must be valid dates")
    if start is not None and end is not None and start > end:
        raise ValueError("start must be on or before end")
    return data.loc[:end] if end is not None else data, start


def walk_forward_predict(
    prices: pd.DataFrame, *, train_size: int = 252, test_size: int = 21,
    neutral_band: float = 0.003, start=None, end=None,
) -> WalkForwardResult:
    """Refit expanding history; purge two rows before each test fold.

    Band is fixed ex ante. Features use close[t]; labels describe adjusted
    open[t+1] -> open[t+2]. Every training outcome is observable strictly
    before the first test day. A fresh training-only scaler is fitted per fold.
    'start' filters decisions, retaining preceding data for training. 'end'
    truncates source data as well as output, so later outcomes remain unknown.
    """
    for name, value in (("train_size", train_size), ("test_size", test_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    data, start_date = _period(validate_prices(prices), start, end)
    if data.empty:
        raise ValueError("no price data on or before end")
    target = build_class_target(data, neutral_band)
    features = build_features(data).dropna()
    first_test = max(train_size + 2, int(features.index.searchsorted(start_date))
                     if start_date is not None else 0)
    if len(features) <= first_test:
        raise ValueError("not enough data after indicator warm-up, training and two-row purge")
    predictions, folds = [], []
    for fold, first in enumerate(range(first_test, len(features), test_size)):
        train = features.iloc[:first - 2]
        test = features.iloc[first:first + test_size]
        model = SignalModel().fit(train, target.loc[train.index])
        probabilities = model.predict_proba(test)
        # Hold wins exact probability ties, avoiding an arbitrary buy on ties.
        predicted = probabilities[["hold", "buy", "sell"]].idxmax(axis=1)
        batch = probabilities.rename(columns={c: f"prob_{c}" for c in CLASSES})
        batch["predicted_class"] = predicted
        batch["probability"] = probabilities.max(axis=1)
        batch["fold"] = fold
        predictions.append(batch)
        last = data.index.get_loc(train.index[-1])
        folds.append({
            "fold": fold, "train_start": train.index[0], "train_end": train.index[-1],
            "last_label_date": data.index[last + 2], "test_start": test.index[0],
            "test_end": test.index[-1], "train_samples": len(train), "test_samples": len(test),
        })
    result = pd.concat(predictions)
    result["action"] = result.predicted_class
    result["actual_class"] = target.reindex(result.index)
    result["correct"] = result.predicted_class.eq(result.actual_class).astype("boolean")
    result.loc[result.actual_class.isna(), "correct"] = pd.NA
    dates = pd.Series(data.index, index=data.index)
    result["execution_date"] = dates.shift(-1).reindex(result.index)
    result["label_available_date"] = dates.shift(-2).reindex(result.index)
    result["close"] = data.close.reindex(result.index)
    result["model_name"] = SignalModel.name
    if "symbol" in data:
        result["symbol"] = data.symbol.iloc[0]
    return WalkForwardResult(result, pd.DataFrame(folds))


def run_prediction(
    prices: pd.DataFrame, *, train_size: int = 252, test_size: int = 21,
    neutral_band: float = 0.003, initial_cash: float = 100_000.0,
    commission_bps: float = 10.0, slippage_bps: float = 5.0,
    start=None, end=None, benchmark_prices: pd.DataFrame | None = None,
) -> PredictionResult:
    """Out-of-sample ML signals, classification metrics and long/cash report."""
    data, _ = _period(validate_prices(prices), start, end)
    walk = walk_forward_predict(data, train_size=train_size, test_size=test_size,
                                neutral_band=neutral_band, start=start)
    predictions = walk.predictions
    evaluation = data.loc[predictions.index]
    benchmark = adjusted_open(validate_prices(benchmark_prices)) if benchmark_prices is not None else None
    backtest = run_backtest(
        evaluation, predictions.action, initial_cash=initial_cash,
        commission_bps=commission_bps, slippage_bps=slippage_bps, benchmark_open=benchmark,
    )
    config = {
        "model": SignalModel.name, "strategy": "ml", "position_mode": "long_cash",
        "decision_rule": "argmax_hold_first_on_ties", "neutral_band": neutral_band,
        "label_horizon": "next_open_to_following_open", "price_basis": "adjusted",
        "train_size": train_size, "test_size": test_size, "purge_rows": 2,
        "initial_cash": initial_cash, "commission_bps": commission_bps, "slippage_bps": slippage_bps,
        "requested_start": str(start) if start is not None else None,
        "requested_end": str(end) if end is not None else None,
        "evaluation_start": str(predictions.index[0].date()),
        "evaluation_end": str(predictions.index[-1].date()),
    }
    return PredictionResult(predictions, walk.folds, backtest,
                            evaluate_predictions(predictions), config)


def run_strategy_backtest(
    prices: pd.DataFrame, *, strategy: str, start=None, end=None,
    fast: int = 5, slow: int = 20, initial_cash: float = 100_000.0,
    commission_bps: float = 10.0, slippage_bps: float = 5.0,
    benchmark_prices: pd.DataFrame | None = None,
) -> BacktestResult:
    """Non-ML strategies need no ML training; SMA retains pre-period history."""
    if strategy not in ("buy_hold", "sma"):
        raise ValueError("use run_prediction for the ml strategy")
    data, start_date = _period(validate_prices(prices), start, end)
    if data.empty:
        raise ValueError("no prices in requested period")
    signals = strategy_signals(data, strategy, fast=fast, slow=slow)
    evaluation = data.loc[start_date:] if start_date is not None else data
    if strategy == "sma":
        from prediction.indicators import sma
        ready = sma(data.adj_close, slow).dropna().index
        evaluation = evaluation.loc[evaluation.index.intersection(ready)]
    if len(evaluation) < 3:
        raise ValueError("backtest needs at least three trading days after warm-up")
    signals = signals.reindex(evaluation.index)
    if strategy == "buy_hold":
        signals.iloc[0] = "buy"
    benchmark = adjusted_open(validate_prices(benchmark_prices)) if benchmark_prices is not None else None
    return run_backtest(evaluation, signals, initial_cash=initial_cash,
                        commission_bps=commission_bps, slippage_bps=slippage_bps,
                        benchmark_open=benchmark)
