from datetime import datetime
import io
import json

import pandas as pd
import pytest

from scripts.download_prices import download_prices


def response(prices):
    return {"chart": {"error": None, "result": [{
        "meta": {"symbol": "THYAO.IS", "currency": "TRY", "exchangeTimezoneName": "Europe/Istanbul"},
        "timestamp": [int(day.timestamp()) for day in prices.index],
        "indicators": {"quote": [{column: prices[column].tolist()
                                   for column in ("open", "high", "low", "close", "volume")}],
                       "adjclose": [{"adjclose": prices.adj_close.tolist()}]},
    }]}}


def test_downloader_validates_data_and_writes_provenance(prices, tmp_path, monkeypatch):
    payload = response(prices)
    payload["chart"]["result"][0]["indicators"]["quote"][0]["open"][5] = None
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    metadata = download_prices("thyao", output_dir=tmp_path)
    assert metadata["rows"] == len(prices) - 1
    assert metadata["excluded_incomplete_rows"] == 1
    assert "THYAO.IS" in calls[0]
    assert (tmp_path / "THYAO.source.json").is_file()
    assert len(pd.read_csv(tmp_path / "THYAO.csv")) == len(prices) - 1


def test_download_failure_does_not_overwrite_existing_data(tmp_path, monkeypatch):
    destination = tmp_path / "THYAO.csv"
    destination.write_text("existing data")
    def fail(*args, **kwargs):
        raise OSError("provider unavailable")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(OSError):
        download_prices("THYAO", output_dir=tmp_path)
    assert destination.read_text() == "existing data"


def test_invalid_download_options_do_not_make_network_calls(tmp_path):
    with pytest.raises(ValueError):
        download_prices("THYAO", period="bad", output_dir=tmp_path)
    with pytest.raises(ValueError):
        download_prices("../THYAO", output_dir=tmp_path)
