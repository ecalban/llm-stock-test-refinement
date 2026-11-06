"""Local mutation job API and measured results, shared with the research UI."""

import csv
from functools import lru_cache
from io import StringIO
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from testgen.experiments import ExperimentService

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Literal["indicators", "signals", "data", "backtest"] = "indicators"
    provider: Literal["reference", "openai", "ollama"] = "reference"
    feedback: Literal["score", "line", "code"] = "score"
    rounds: int = Field(default=1, ge=1, le=5)
    mutant_limit: int = Field(default=8, ge=1, le=24)


@lru_cache
def get_experiments():
    return ExperimentService(get_settings())


def _summary(run):
    rounds = [{key: value for key, value in entry.items() if key not in ("mutants", "validation")}
              for entry in run["rounds"]]
    return {**run, "rounds": rounds}


@router.get("/options")
def options(service=Depends(get_experiments)):
    return service.options()


@router.get("")
def history(service=Depends(get_experiments)):
    return {"items": [_summary(run) for run in service.list()]}


@router.post("", status_code=202)
def start(request: ExperimentRequest, service=Depends(get_experiments)):
    try:
        return service.start(request.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def comparison_rows(runs):
    rows = []
    for run in runs:
        if run["status"] != "completed" or run["config"]["provider"] == "reference":
            continue
        initial, final = run["rounds"][0], run["rounds"][-1]
        rows.append({
            "id": run["id"], "target": run["config"]["target"], "model": run["model"],
            "provider": run["config"]["provider"], "feedback": run["config"]["feedback"],
            "catalog": run["catalog"], "source_hash": run["source_hash"],
            "initial_score": initial["score"], "final_score": final["score"],
            "delta": final["score"] - initial["score"], "rounds": len(run["rounds"]),
            "planned_rounds": run["config"]["rounds"], "mutants": run["selected_mutants"],
            "killed": final["killed"], "survived": final["survived"],
            "timeouts": final["timeout"], "errors": final["errors"],
            "generated": sum(r["generated"] for r in run["rounds"]),
            "valid": sum(r["valid"] for r in run["rounds"]),
            "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in run["rounds"]),
            "duration_seconds": run["duration_seconds"],
        })
    return rows


@router.get("/comparisons")
def comparisons(catalog: str | None = None, rounds: int | None = None,
                format: Literal["json", "csv"] = "json",
                service=Depends(get_experiments)):
    rows = comparison_rows(service.list())
    if catalog is not None:
        rows = [row for row in rows if row["catalog"] == catalog]
    if rounds is not None:
        rows = [row for row in rows if row["planned_rounds"] == rounds]
    if format == "json":
        return {"items": rows}
    output = StringIO()
    fields = list(rows[0]) if rows else ["id", "target", "model", "provider", "feedback", "catalog",
                                        "initial_score", "final_score", "delta"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return Response(output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="llm-experiments.csv"'})


@router.get("/{identifier}")
def detail(identifier: UUID, service=Depends(get_experiments)):
    try:
        return service.get(identifier)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Deney bulunamadı.") from exc
