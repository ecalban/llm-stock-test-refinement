import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from prediction.pipeline import run_prediction, walk_forward_predict

ROOT = Path(__file__).resolve().parents[2]


def test_walk_forward_purges_labels_and_covers_final_partial_fold(prices):
    result = walk_forward_predict(prices, train_size=40, test_size=17)
    # 20 warm-up rows, 40 training rows and two purged rows.
    assert result.predictions.index.equals(prices.index[62:])
    folds = result.folds
    assert (folds.last_label_date < folds.test_start).all()
    assert folds.train_samples.iloc[0] == 40
    assert folds.train_samples.diff().dropna().eq(17).all()
    assert folds.test_samples.iloc[-1] == 13
    assert result.predictions.probability.between(0, 1).all()


def test_future_prices_cannot_change_previous_predictions(prices):
    prefix = walk_forward_predict(prices.iloc[:105], train_size=40, test_size=17)
    changed = prices.copy()
    columns = ["open", "high", "low", "close", "adj_close"]
    changed.loc[changed.index[105:], columns] *= 3
    changed.loc[changed.index[105:], "volume"] *= 5
    full = walk_forward_predict(changed, train_size=40, test_size=17)
    columns = ["prob_buy", "prob_hold", "prob_sell", "predicted_class", "probability", "fold", "action"]
    pd.testing.assert_frame_equal(prefix.predictions[columns],
                                  full.predictions.loc[prefix.predictions.index, columns])
    assert prefix.predictions.actual_class.iloc[-2:].isna().all()


def test_training_scaler_excludes_purged_and_test_rows(prices, monkeypatch):
    from prediction.model import SignalModel

    fitted_indexes = []
    original_fit = SignalModel.fit

    def spy(self, features, target):
        fitted_indexes.append(features.index.copy())
        return original_fit(self, features, target)

    monkeypatch.setattr(SignalModel, "fit", spy)
    result = walk_forward_predict(prices, train_size=40, test_size=17)
    assert len(fitted_indexes) == len(result.folds)
    for indexes, fold in zip(fitted_indexes, result.folds.itertuples()):
        assert len(indexes) == fold.train_samples
        assert indexes[-1] == fold.train_end
        # t+2 label is observable strictly before first test close.
        assert prices.index[prices.index.get_loc(indexes[-1]) + 2] < fold.test_start


def test_pipeline_evaluates_only_out_of_sample_rows(prices):
    prices["symbol"] = "THYAO.IS"
    result = run_prediction(prices, train_size=40, test_size=17)
    assert result.backtest.daily.index.equals(result.predictions.index)
    assert result.predictions.action.isin(["buy", "sell", "hold"]).all()
    assert result.predictions.symbol.eq("THYAO.IS").all()
    assert result.predictions.model_name.eq("logistic_signal").all()
    assert result.backtest.daily.position.iloc[0] == 0
    assert result.backtest.daily.position.iloc[-1] == 0
    assert all(np.isfinite(value) for value in result.backtest.metrics.values() if value is not None)


@pytest.mark.parametrize("kwargs", [{"train_size": 0}, {"train_size": True}, {"test_size": -1}, {"test_size": 1.5}])
def test_walk_forward_rejects_invalid_windows(prices, kwargs):
    with pytest.raises(ValueError):
        walk_forward_predict(prices, **kwargs)


def test_walk_forward_rejects_insufficient_history(prices):
    with pytest.raises(ValueError, match="not enough data"):
        walk_forward_predict(prices, train_size=252)


def test_default_pipeline_accepts_documented_minimum_history(prices):
    data = pd.concat([prices, prices.iloc[:117]], ignore_index=True)
    data.index = pd.bdate_range("2023-01-02", periods=277, name="date")
    result = run_prediction(data)
    assert len(result.predictions) == 3
    assert result.folds.train_samples.tolist() == [252]


def test_cli_generates_reviewable_outputs(prices, tmp_path):
    source = tmp_path / "prices.csv"
    output = tmp_path / "results"
    prices.to_csv(source)
    result = subprocess.run([
        sys.executable, "-m", "prediction", str(source), "--symbol", "THYAO",
        "--train-size", "40", "--test-size", "17", "--output-dir", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert set(path.name for path in output.iterdir()) == {
        "signals.csv", "folds.csv", "backtest.csv", "metrics.json",
        "trades.csv", "orders.csv", "classification.json", "config.json", "report.json",
    }
    metrics = json.loads((output / "metrics.json").read_text())
    assert "total_return" in metrics
    signals = pd.read_csv(output / "signals.csv")
    assert signals.symbol.eq("THYAO.IS").all()
    assert len(signals) == len(prices) - 62


def test_cli_invalid_input_gives_actionable_error(tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "prediction", str(tmp_path / "missing.csv"),
    ], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
