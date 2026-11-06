"""A deterministic, standardized logistic-regression baseline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ("buy", "hold", "sell")


def _validate_features(features: pd.DataFrame) -> None:
    if features.empty or features.columns.has_duplicates:
        raise ValueError("features must be non-empty with unique columns")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("features must be finite; remove warm-up rows first")


class DirectionModel:
    """Estimate P(next tradable open-to-open return > 0).

    Scaling is fitted only on training rows. A single-class training set gives
    that class's constant probability instead of failing. Probabilities are
    baseline classifier outputs, not calibrated investment confidence.
    """

    name = "logistic_direction"

    def __init__(self) -> None:
        self._columns: list[str] | None = None
        self._constant: float | None = None
        self._pipeline = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "DirectionModel":
        _validate_features(features)
        if not features.index.equals(target.index):
            raise ValueError("features and target indexes must match exactly")
        if not target.isin([0, 1]).all():
            raise ValueError("target must contain only 0 and 1")
        pipeline = None
        constant = None
        if target.nunique() == 1:
            constant = float(target.iloc[0])
        else:
            pipeline = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=0)
            )
            pipeline.fit(features, target)
        self._columns = list(features.columns)
        self._constant = constant
        self._pipeline = pipeline
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Return the positive-class probability, retaining the input index."""
        if self._columns is None:
            raise RuntimeError("fit the model before predicting")
        _validate_features(features)
        if list(features.columns) != self._columns:
            raise ValueError("prediction columns must match training columns in order")
        if self._constant is not None:
            values = np.full(len(features), self._constant)
        else:
            values = self._pipeline.predict_proba(features)[:, 1]
        return pd.Series(values, index=features.index, name="probability")


class SignalModel:
    """Three-class logistic model with a stable buy/hold/sell probability schema.

    Absent training classes get probability zero. A single-class fold produces
    a one-hot distribution. Scaling is fitted exclusively on training rows.
    ``DirectionModel`` remains available for older binary experiments.
    """

    name = "logistic_signal"

    def __init__(self) -> None:
        self._columns: list[str] | None = None
        self._constant: str | None = None
        self._pipeline = None

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "SignalModel":
        _validate_features(features)
        if not features.index.equals(target.index):
            raise ValueError("features and target indexes must match exactly")
        if not target.isin(CLASSES).all():
            raise ValueError("target must contain only buy, hold and sell")
        pipeline, constant = None, None
        if target.nunique() == 1:
            constant = str(target.iloc[0])
        else:
            pipeline = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=0)
            )
            pipeline.fit(features, target)
        self._columns = list(features.columns)
        self._constant = constant
        self._pipeline = pipeline
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if self._columns is None:
            raise RuntimeError("fit the model before predicting")
        _validate_features(features)
        if list(features.columns) != self._columns:
            raise ValueError("prediction columns must match training columns in order")
        result = pd.DataFrame(0.0, index=features.index, columns=list(CLASSES))
        if self._constant is not None:
            result[self._constant] = 1.0
        else:
            values = self._pipeline.predict_proba(features)
            result.loc[:, self._pipeline.classes_] = values
        return result
