"""Evaluate only observable out-of-sample classes; unknown outcomes stay null."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from prediction.model import CLASSES


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    """Fixed-order confusion matrix, accuracy, macro F1 and per-class metrics."""
    for column in ("predicted_class", "actual_class"):
        if column not in predictions:
            raise ValueError(f"missing column: {column}")
    if not predictions["predicted_class"].isin(CLASSES).all():
        raise ValueError("unknown predicted class")
    known = predictions.dropna(subset=["actual_class"])
    if not known["actual_class"].isin(CLASSES).all():
        raise ValueError("unknown actual class")
    result = {"samples": len(known), "pending": len(predictions) - len(known),
              "accuracy": None, "macro_f1": None, "majority_accuracy": None,
              "classes": list(CLASSES), "confusion_matrix": [[0] * 3 for _ in CLASSES],
              "per_class": {}}
    if known.empty:
        return result
    actual, predicted = known["actual_class"], known["predicted_class"]
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=list(CLASSES), zero_division=0
    )
    result.update({
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(np.mean(f1)),
        "majority_accuracy": float(actual.value_counts().max() / len(actual)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(CLASSES)).tolist(),
        "per_class": {label: {"precision": float(p), "recall": float(r),
                               "f1": float(f), "support": int(s)}
                      for label, p, r, f, s in zip(CLASSES, precision, recall, f1, support)},
    })
    return result
