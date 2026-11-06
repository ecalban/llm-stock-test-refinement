import json
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from app.api.prediction import get_prediction_service
from app.main import app
from prediction.service import PredictionService


@pytest.fixture
def api(prices, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    prices.to_csv(data / "THYAO.csv")
    (prices.assign(open=prices.open * 10, high=prices.high * 10,
                   low=prices.low * 10, close=prices.close * 10,
                   adj_close=prices.adj_close * 10)).to_csv(data / "XU100.csv")
    service = PredictionService(data, tmp_path / "reports")
    app.dependency_overrides[get_prediction_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client, service
    finally:
        app.dependency_overrides.pop(get_prediction_service, None)


def test_capabilities_and_stock_catalog(api):
    client, _ = api
    caps = client.get("/api/capabilities").json()
    assert caps["classes"] == ["buy", "hold", "sell"]
    assert caps["models"][0]["available"] is True
    assert caps["models"][1]["available"] is False
    stocks = client.get("/api/stocks").json()["items"]
    assert {item["symbol"] for item in stocks} == {"THYAO.IS", "XU100.IS"}
    assert next(s for s in stocks if s["symbol"] == "XU100.IS")["kind"] == "index"


def test_prices_filter_without_training_and_reject_inverted_dates(api):
    client, _ = api
    result = client.get("/api/stocks/THYAO/prices", params={"start": "2024-07-01", "end": "2024-07-05"})
    assert result.status_code == 200
    assert len(result.json()["items"]) == 5
    assert client.get("/api/stocks/THYAO/prices?start=2025-01-01&end=2024-01-01").status_code == 422


def test_analysis_persists_and_exports_actual_csv(api):
    client, service = api
    response = client.post("/api/analyses", json={"symbol": "THYAO", "train_size": 40, "benchmark": True})
    assert response.status_code == 201, response.text
    report = response.json()
    identifier = report["id"]
    assert report["latest_prediction"]["actual_class"] is None
    assert report["latest_prediction"]["correct"] is None
    assert report["classification_metrics"]["samples"] == 96
    assert report["backtest"]["metrics"]["bist100_return"] is not None
    assert report["sources"][0]["sha256"]
    assert report["source_hash"] and report["runtime_versions"]["scikit-learn"]
    assert client.get(f"/api/analyses/{identifier}").json() == report
    restored = PredictionService(service.data_dir, service.runs_dir)
    assert restored.read(identifier)["id"] == identifier
    page = client.get(f"/api/analyses/{identifier}/signals?offset=2&limit=3").json()
    assert len(page["items"]) == 3 and page["total"] == 98
    download = client.get(f"/api/analyses/{identifier}/signals?format=csv")
    assert download.status_code == 200 and "text/csv" in download.headers["content-type"]
    assert "attachment" in download.headers["content-disposition"]
    assert "prob_hold" in download.text and "actual_class" in download.text


def test_analysis_short_view_preserves_training_history(api):
    client, _ = api
    response = client.post("/api/analyses", json={
        "symbol": "THYAO", "train_size": 40, "start": "2024-07-01", "end": "2024-07-31"})
    assert response.status_code == 201, response.text
    report = response.json()
    assert report["config"]["evaluation_start"] == "2024-07-01"
    assert report["as_of"] == "2024-07-31"
    assert len(report["series"]) == 23
    assert report["folds"][0]["train_samples"] > 40
    assert report["classification_metrics"]["pending"] == 2


@pytest.mark.parametrize("strategy", ["ml", "buy_hold", "sma"])
def test_all_backtest_strategies_persist_real_ledgers(api, strategy):
    client, _ = api
    response = client.post("/api/backtests", json={
        "symbol": "THYAO", "strategy": strategy, "train_size": 40,
        "start": "2024-06-03", "benchmark": True,
    })
    assert response.status_code == 201, response.text
    report = response.json()
    assert report["config"]["strategy"] == strategy
    assert report["metrics"]["bist100_return"] is not None
    assert report["metrics"]["order_count"] == len(report["orders"])
    assert report["metrics"]["closed_trade_count"] == len(report["trades"])
    assert client.get(f"/api/backtests/{report['id']}").json() == report
    if strategy == "buy_hold":
        assert report["metrics"]["closed_trade_count"] == 1
        assert report["metrics"]["total_return"] == pytest.approx(report["metrics"]["buy_hold_return"])
    download = client.get(f"/api/backtests/{report['id']}/trades?format=csv")
    assert download.status_code == 200
    assert client.get(f"/api/backtests/{report['id']}/trades?limit=2").json()["total"] == len(report["trades"])


def test_dashboard_is_computed_from_saved_analyses_not_demo_counts(api):
    client, _ = api
    before = client.get("/api/dashboard").json()
    assert before["available_stock_count"] == 1
    assert before["analysed_stock_count"] == 0 and before["pooled_accuracy"] is None
    client.post("/api/analyses", json={"symbol": "THYAO", "train_size": 40})
    client.post("/api/analyses", json={"symbol": "THYAO", "train_size": 40})
    after = client.get("/api/dashboard").json()
    assert after["analysed_stock_count"] == 1  # latest per symbol, not number of runs
    assert len(after["latest_signals"]) == 1
    assert after["pooled_accuracy"] is not None
    assert after["mutation_score"] is None and after["last_test_status"] is None


@pytest.mark.parametrize("payload", [
    {"model": "lightgbm"}, {"train_size": 0}, {"neutral_band": -1},
    {"strategy": "short"}, {"start": "2025-01-01", "end": "2024-01-01"},
    {"commission_bps": 9000, "slippage_bps": 1000},
    {"strategy": "sma", "fast": 20, "slow": 10}, {"unexpected": True},
])
def test_api_rejects_unsupported_and_invalid_requests(api, payload):
    client, _ = api
    response = client.post("/api/backtests", json={"symbol": "THYAO", **payload})
    assert response.status_code == 422


def test_missing_data_report_kind_and_unknown_run_are_explicit(api):
    client, _ = api
    assert client.post("/api/analyses", json={"symbol": "ASELS"}).status_code == 404
    assert client.get("/api/analyses/not-a-uuid").status_code == 404
    assert client.get("/api/analyses/00000000-0000-0000-0000-000000000000").status_code == 404
    report = client.post("/api/analyses", json={"symbol": "THYAO", "train_size": 40}).json()
    assert client.get(f"/api/backtests/{report['id']}").status_code == 404
    assert client.post("/api/analyses", json={"symbol": "../THYAO"}).status_code == 422


def test_insufficient_training_and_missing_benchmark_fail_without_saving(api):
    client, service = api
    assert client.post("/api/analyses", json={"symbol": "THYAO"}).status_code == 422
    (service.data_dir / "XU100.csv").unlink()
    assert client.post("/api/analyses", json={"symbol": "THYAO", "train_size": 40,
                                            "benchmark": True}).status_code == 404
    assert not list(service.runs_dir.glob("*.json"))
