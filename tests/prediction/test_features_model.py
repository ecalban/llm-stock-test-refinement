import numpy as np
import pandas as pd
import pytest

from prediction.features import build_features, build_target
from prediction.model import DirectionModel


def test_features_warmup_and_causality(prices):
    features = build_features(prices)
    assert features.shape == (len(prices), 9)
    assert features.iloc[:20].isna().any(axis=1).all()
    assert np.isfinite(features.iloc[20:].to_numpy()).all()
    pd.testing.assert_frame_equal(features.iloc[:75], build_features(prices.iloc[:75]))


def test_flat_prices_and_zero_volume_have_finite_features(prices):
    prices[["open", "high", "low", "close", "adj_close"]] = 100.0
    prices["volume"] = 0
    features = build_features(prices).iloc[20:]
    assert np.isfinite(features.to_numpy()).all()
    assert features.rsi_14.eq(0.5).all()
    assert features.band_position.eq(0.5).all()
    assert features.volume_ratio.eq(0).all()


def test_target_matches_next_open_execution_and_keeps_unknowns(prices):
    data = prices.iloc[:5].copy()
    data["open"] = [100, 110, 105, 120, 120]
    data["close"] = data["open"]
    data["adj_close"] = data["close"]
    data["high"] = data["open"]
    data["low"] = data["open"]
    target = build_target(data)
    assert target.iloc[:3].tolist() == [0, 1, 0]
    assert target.iloc[-2:].isna().all()


def test_model_fit_predict_and_determinism(prices):
    x = build_features(prices).dropna().iloc[:-2]
    y = build_target(prices).loc[x.index]
    a = DirectionModel().fit(x.iloc[:80], y.iloc[:80]).predict_proba(x.iloc[80:])
    b = DirectionModel().fit(x.iloc[:80], y.iloc[:80]).predict_proba(x.iloc[80:])
    assert a.between(0, 1).all()
    assert a.index.equals(x.iloc[80:].index)
    pd.testing.assert_series_equal(a, b)


@pytest.mark.parametrize("label", [0, 1])
def test_model_single_class_then_refit(prices, label):
    x = build_features(prices).dropna().iloc[:-2]
    model = DirectionModel().fit(x, pd.Series(label, index=x.index))
    assert model.predict_proba(x).eq(float(label)).all()
    model.fit(x, build_target(prices).loc[x.index])
    assert model.predict_proba(x).between(0, 1, inclusive="neither").all()


def test_model_requires_fit(prices):
    with pytest.raises(RuntimeError):
        DirectionModel().predict_proba(build_features(prices).dropna())


def test_model_rejects_column_reordering(prices):
    x = build_features(prices).dropna().iloc[:-2]
    model = DirectionModel().fit(x, build_target(prices).loc[x.index])
    with pytest.raises(ValueError, match="columns"):
        model.predict_proba(x[x.columns[::-1]])


@pytest.mark.parametrize("kind", ["unaligned", "unknown_label", "invalid_label", "nan_features", "empty"])
def test_model_rejects_invalid_training_data(prices, kind):
    x = build_features(prices).dropna().iloc[:-2]
    y = build_target(prices).loc[x.index]
    if kind == "unaligned":
        y = y.iloc[::-1]
    elif kind == "unknown_label":
        y.iloc[0] = np.nan
    elif kind == "invalid_label":
        y.iloc[0] = 2
    elif kind == "nan_features":
        x.iloc[0, 0] = np.nan
    else:
        x, y = x.iloc[:0], y.iloc[:0]
    with pytest.raises(ValueError):
        DirectionModel().fit(x, y)
