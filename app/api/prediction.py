"""Synchronous local prediction APIs. Long-running LLM experiment APIs are separate."""

from datetime import date
import io
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator
import pandas as pd

from app.config import get_settings
from prediction.reporting import records
from prediction.service import PredictionService

router = APIRouter(prefix="/api", tags=["prediction"])


def get_prediction_service():
    settings = get_settings()
    return PredictionService(settings.market_data_dir, settings.prediction_runs_dir)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    symbol: str
    model: Literal["logistic_signal"] = "logistic_signal"
    start: date | None = None
    end: date | None = None
    train_size: int = Field(default=252, ge=1, le=100_000)
    test_size: int = Field(default=21, ge=1, le=10_000)
    neutral_band: float = Field(default=0.003, ge=0, lt=1)
    initial_cash: float = Field(default=100_000, gt=0)
    commission_bps: float = Field(default=10, ge=0, lt=10_000)
    slippage_bps: float = Field(default=5, ge=0, lt=10_000)
    benchmark: bool = False

    @model_validator(mode="after")
    def valid_period_and_costs(self):
        if self.start and self.end and self.start > self.end:
            raise ValueError("start must be on or before end")
        if self.commission_bps + self.slippage_bps >= 10_000:
            raise ValueError("total costs must be less than 10000 bps")
        return self


class BacktestRequest(AnalysisRequest):
    strategy: Literal["ml", "buy_hold", "sma"] = "ml"
    fast: int = Field(default=5, ge=1)
    slow: int = Field(default=20, ge=2)

    @model_validator(mode="after")
    def valid_sma(self):
        if self.strategy == "sma" and self.fast >= self.slow:
            raise ValueError("fast must be smaller than slow")
        return self


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/capabilities")
def capabilities():
    return {"models": [{"id": "logistic_signal", "name": "Lojistik Regresyon", "available": True},
                        {"id": "lightgbm", "name": "LightGBM", "available": False},
                        {"id": "lstm", "name": "LSTM (opsiyonel)", "available": False}],
            "strategies": ["ml", "buy_hold", "sma"], "classes": ["buy", "hold", "sell"],
            "position_modes": ["long_cash"], "data_frequency": "daily",
            "execution": "next_open", "label_horizon": "next_open_to_following_open"}


@router.get("/stocks")
def stocks(service=Depends(get_prediction_service)):
    return {"items": _call(service.stocks)}


@router.get("/stocks/{symbol}/prices")
def prices(symbol: str, start: date | None = None, end: date | None = None,
           service=Depends(get_prediction_service)):
    if start and end and start > end:
        raise HTTPException(422, "start must be on or before end")
    data = _call(service.prices, symbol)
    selected = data.loc[str(start) if start else None:str(end) if end else None]
    return {"symbol": symbol, "items": records(selected, index=True)}


@router.post("/analyses", status_code=201)
def analyse(request: AnalysisRequest, service=Depends(get_prediction_service)):
    return _call(service.analyse, **request.model_dump(exclude={"model"}))


@router.get("/analyses/{identifier}")
def analysis(identifier: str, service=Depends(get_prediction_service)):
    return _call(service.read, identifier, kind="analysis")


def _table(report, table, format, offset, limit):
    rows = report[table]
    if format == "csv":
        buffer = io.StringIO()
        pd.DataFrame(rows).to_csv(buffer, index=False)
        return Response(buffer.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{table}.csv"'})
    return {"total": len(rows), "offset": offset, "items": rows[offset:offset + limit]}


@router.get("/analyses/{identifier}/signals")
def signals(identifier: str, format: Literal["json", "csv"] = "json",
            offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=5000),
            service=Depends(get_prediction_service)):
    report = _call(service.read, identifier, kind="analysis")
    return _table(report, "signals", format, offset, limit)


@router.post("/backtests", status_code=201)
def backtest(request: BacktestRequest, service=Depends(get_prediction_service)):
    options = request.model_dump(exclude={"model"})
    if request.strategy == "ml":
        options.pop("fast")
        options.pop("slow")
    return _call(service.backtest, **options)


@router.get("/backtests/{identifier}")
def backtest_result(identifier: str, service=Depends(get_prediction_service)):
    return _call(service.read, identifier, kind="backtest")


@router.get("/backtests/{identifier}/trades")
def trades(identifier: str, format: Literal["json", "csv"] = "json",
           offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=5000),
           service=Depends(get_prediction_service)):
    report = _call(service.read, identifier, kind="backtest")
    return _table(report, "trades", format, offset, limit)


@router.get("/dashboard")
def dashboard(service=Depends(get_prediction_service)):
    return _call(service.dashboard)
