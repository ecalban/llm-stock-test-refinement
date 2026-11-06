import numpy as np
import pandas as pd
import pytest

from prediction.evaluation import evaluate_predictions
from prediction.features import build_class_target, build_features
from prediction.model import CLASSES, SignalModel
from prediction.pipeline import run_prediction, walk_forward_predict
from prediction.reporting import analysis_report


def test_class_targets_and_inclusive_neutral_boundary(prices):
    data = prices.iloc[:6].copy()
    for column in ("open", "high", "low", "close", "adj_close"):
        data[column] = [50, 100, 101, 100, 90, 100]
    labels = build_class_target(data, neutral_band=0.01)
    assert labels.iloc[:4].tolist() == ["hold", "hold", "sell", "buy"]
    assert labels.iloc[-2:].isna().all()


@pytest.mark.parametrize("band", [-0.1, 1, np.nan, np.inf])
def test_invalid_neutral_band(prices, band):
    with pytest.raises(ValueError):
        build_class_target(prices, band)


@pytest.mark.parametrize("classes", [["buy"], ["hold"], ["sell"], ["buy", "sell"], ["buy", "hold", "sell"]])
def test_probability_schema_and_missing_training_classes(prices, classes):
    features = build_features(prices).dropna().iloc[:60]
    target = pd.Series([classes[i % len(classes)] for i in range(len(features))], index=features.index)
    model = SignalModel().fit(features, target)
    result = model.predict_proba(features.iloc[-10:])
    assert list(result.columns) == list(CLASSES)
    assert result.index.equals(features.iloc[-10:].index)
    assert np.allclose(result.sum(axis=1), 1)
    assert result.ge(0).all().all() and result.le(1).all().all()
    for missing in set(CLASSES) - set(classes):
        assert result[missing].eq(0).all()
    if len(classes) == 1:
        assert result[classes[0]].eq(1).all()


def test_signal_model_requires_fit_and_exact_schema(prices):
    x = build_features(prices).dropna()
    model = SignalModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(x)
    model.fit(x, pd.Series("hold", index=x.index))
    with pytest.raises(ValueError):
        model.predict_proba(x[x.columns[::-1]])


@pytest.mark.parametrize("kind", ["unknown", "missing", "misaligned"])
def test_signal_model_rejects_invalid_target(prices, kind):
    x = build_features(prices).dropna()
    target = pd.Series("hold", index=x.index, dtype=object)
    if kind == "unknown":
        target.iloc[0] = "short"
    elif kind == "missing":
        target.iloc[0] = None
    else:
        target = target.iloc[::-1]
    with pytest.raises(ValueError):
        SignalModel().fit(x, target)


def test_refitting_clears_constant_class(prices):
    x = build_features(prices).dropna().iloc[:-2]
    model = SignalModel().fit(x, pd.Series("hold", index=x.index))
    model.fit(x, build_class_target(prices).loc[x.index])
    assert model.predict_proba(x).buy.gt(0).all()


def test_evaluation_excludes_unknown_outcomes_and_counts_confusion():
    predictions = pd.DataFrame({
        "predicted_class": ["buy", "hold", "sell", "buy"],
        "actual_class": ["buy", "sell", "sell", None],
    })
    report = evaluate_predictions(predictions)
    assert report["samples"] == 3 and report["pending"] == 1
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["majority_accuracy"] == pytest.approx(2 / 3)
    assert report["macro_f1"] == pytest.approx(5 / 9)
    assert report["confusion_matrix"] == [[1, 0, 0], [0, 0, 0], [0, 1, 1]]
    assert report["per_class"]["sell"]["support"] == 2


def test_no_known_outcomes_has_null_accuracy():
    report = evaluate_predictions(pd.DataFrame({"predicted_class": ["hold"], "actual_class": [None]}))
    assert report["accuracy"] is None
    assert report["samples"] == 0 and report["pending"] == 1


def test_end_date_hides_later_labels_and_preserves_probabilities(prices):
    end = prices.index[100]
    a = walk_forward_predict(prices, train_size=40, end=end)
    b = walk_forward_predict(prices.loc[:end], train_size=40)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    assert a.predictions.index[-1] == end
    assert a.predictions.actual_class.iloc[-2:].isna().all()
    assert a.predictions.correct.iloc[-2:].isna().all()
    assert a.predictions.label_available_date.iloc[-2:].isna().all()
    assert pd.isna(a.predictions.execution_date.iloc[-1])


def test_short_view_keeps_earlier_training_history(prices):
    result = run_prediction(prices, train_size=40, start=prices.index[135])
    assert result.predictions.index.equals(prices.index[135:])
    assert result.folds.train_samples.iloc[0] == 113
    assert result.classification_metrics["samples"] == 23
    assert result.classification_metrics["pending"] == 2


def test_flat_data_is_genuinely_hold_class(prices):
    prices[["open", "high", "low", "close", "adj_close"]] = 100.0
    result = run_prediction(prices, train_size=40)
    assert result.predictions.prob_hold.eq(1).all()
    assert result.predictions.action.eq("hold").all()
    assert result.backtest.metrics["closed_trade_count"] == 0
    assert result.backtest.metrics["win_rate"] is None
    assert result.classification_metrics["accuracy"] == 1


def test_json_report_has_nullable_outcomes_and_real_indicator_series(prices):
    result = run_prediction(prices, train_size=40)
    report = analysis_report(prices, result)
    assert report["latest_prediction"]["actual_class"] is None
    assert report["latest_prediction"]["correct"] is None
    assert report["latest_prediction"]["execution_date"] is None
    assert sum(report["latest_prediction"]["probabilities"].values()) == pytest.approx(1)
    assert len(report["series"]) == len(result.predictions)
    assert "rsi" in report["series"][-1] and "macd_histogram" in report["series"][-1]
    import json
    json.dumps(report, allow_nan=False)
