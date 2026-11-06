"""JSON-safe reports shared by CLI and HTTP clients. Percentages remain fractions."""

from datetime import datetime, timezone
import json
import hashlib
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from prediction.data import validate_prices
from prediction.indicators import macd, rsi
from prediction.pipeline import PredictionResult


def records(frame: pd.DataFrame, *, index: bool = False) -> list[dict]:
    """Convert nullable pandas values to JSON null and timestamps to ISO 8601."""
    data = frame.reset_index() if index else frame
    return json.loads(data.to_json(orient="records", date_format="iso"))


def backtest_report(backtest, config: dict) -> dict:
    return {
        "config": config, "metrics": backtest.metrics,
        "daily": records(backtest.daily, index=True),
        "trades": records(backtest.trades), "orders": records(backtest.orders),
    }


def analysis_report(prices: pd.DataFrame, result: PredictionResult) -> dict:
    """Display indicators use full pre-period history; forecast outcomes stay null."""
    data = validate_prices(prices).loc[:result.config["evaluation_end"]]
    lines = macd(data.adj_close)
    series = data[["open", "high", "low", "close", "adj_close", "volume"]].copy()
    series["rsi"] = rsi(data.adj_close)
    series["macd"] = lines.macd
    series["macd_signal"] = lines.signal
    series["macd_histogram"] = lines.histogram
    series = series.loc[result.config["evaluation_start"]:]
    latest = records(result.predictions.iloc[[-1]], index=True)[0]
    latest["probabilities"] = {c: latest.pop(f"prob_{c}") for c in ("buy", "hold", "sell")}
    last = data.iloc[-1]
    modules = ("data.py", "indicators.py", "features.py", "model.py", "pipeline.py",
               "strategies.py", "backtest.py", "evaluation.py")
    source_hash = hashlib.sha256(b"".join(
        (Path(__file__).parent / name).read_bytes() for name in modules)).hexdigest()
    return {
        "schema_version": 2,
        "symbol": str(data.symbol.iloc[0]) if "symbol" in data else None,
        "as_of": data.index[-1].date().isoformat(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash,
        "runtime_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "config": result.config, "classification_metrics": result.classification_metrics,
        "quote": {"close": float(last.close),
                  "daily_change": float(last.adj_close / data.adj_close.iloc[-2] - 1),
                  "price_basis": "raw_close_change_adjusted"},
        "latest_prediction": latest,
        "series": records(series, index=True),
        "signals": records(result.predictions, index=True),
        "folds": records(result.folds),
        "backtest": backtest_report(result.backtest, result.config),
    }


def export_prediction(prices: pd.DataFrame, result: PredictionResult, destination: Path) -> dict:
    """Export inspectable ledgers, classification report and a complete UI payload."""
    report = analysis_report(prices, result)
    # Serialize first so invalid metadata cannot leave a misleading partial report.
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    destination.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(destination / "signals.csv", index_label="date")
    result.folds.to_csv(destination / "folds.csv", index=False)
    result.backtest.daily.to_csv(destination / "backtest.csv", index_label="date")
    result.backtest.trades.to_csv(destination / "trades.csv", index=False)
    result.backtest.orders.to_csv(destination / "orders.csv", index=False)
    for filename, value in (("metrics.json", result.backtest.metrics),
                             ("classification.json", result.classification_metrics),
                             ("config.json", result.config)):
        (destination / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (destination / "report.json").write_text(payload + "\n", encoding="utf-8")
    return report
