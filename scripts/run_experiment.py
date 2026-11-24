"""Run the feedback-loop experiment on one module and append one CSV row per round.

Usage (from the repository root):
    python -m scripts.run_experiment prediction/signals.py --formats none line --repeats 1 --max-rounds 3
    python -m scripts.run_experiment prediction/indicators.py --provider ollama

Every run gets its own folder under tests/generated/, so earlier runs are kept.
Results go to results/experiments.csv (one row per round), which is committed:
the tables and charts in the report are built from it.

tests/test_run_experiment.py describes the helpers. Fill in the TODOs in this order:
module_name -> rows_for -> append_csv. main() is given.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from testgen.llm_client import create_client
from testgen.loop import LoopResult, run_loop

CSV_COLUMNS = [
    "run_id", "started_at", "target", "provider", "model", "fmt", "repeat",
    "round", "tests_generated", "tests_valid", "mutants_total", "mutants_survived",
    "score", "prompt_tokens", "completion_tokens", "llm_seconds", "error",
]


def module_name(target: Path) -> str:
    return ".".join(target.with_suffix("").parts)


def rows_for(
    run_id: str, started_at: str, target: Path, provider: str, model: str,
    repeat: int, result: LoopResult,
) -> list[dict]:
    common = {
        "run_id": run_id,
        "started_at": started_at,
        "target": target.as_posix(),
        "provider": provider,
        "model": model,
        "fmt": result.fmt,
        "repeat": repeat,
    }

    # A run that broke before round 0 finished: one row, all numbers 0
    if not result.rounds:
        return [{
            **common,
            "round": -1, "tests_generated": 0, "tests_valid": 0,
            "mutants_total": 0, "mutants_survived": 0, "score": 0,
            "prompt_tokens": 0, "completion_tokens": 0, "llm_seconds": 0,
            "error": result.error,
        }]

    rows = []
    for r in result.rounds:
        rows.append({
            **common,
            "round": r.round,
            "tests_generated": r.tests_generated,
            "tests_valid": r.tests_valid,
            "mutants_total": r.mutants_total,
            "mutants_survived": r.mutants_survived,
            "score": round(r.score, 4),
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "llm_seconds": round(r.duration_seconds, 2),
            "error": result.error,
        })
    return rows


def append_csv(path: Path, rows: list[dict]) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """Given. Parses the arguments and runs every (format, repeat) combination."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path, help="module under test, e.g. prediction/signals.py")
    parser.add_argument("--formats", nargs="+", default=["none", "score", "line", "full"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--provider", default="openai", choices=["openai", "ollama"])
    parser.add_argument("--out", type=Path, default=Path("results/experiments.csv"))
    args = parser.parse_args(argv)

    client = create_client(args.provider)
    model = getattr(client, "model", args.provider)

    for repeat in range(1, args.repeats + 1):
        for fmt in args.formats:
            run_id = uuid4().hex[:8]
            started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            dest = Path("tests/generated") / args.target.stem / f"{fmt}_r{repeat}_{run_id}"
            print(f"[{run_id}] {args.target} fmt={fmt} repeat={repeat} ...", flush=True)

            result = run_loop(
                client, args.target, module_name(args.target), dest,
                fmt=fmt, max_rounds=args.max_rounds,
            )
            rows = rows_for(run_id, started_at, args.target, args.provider, model, repeat, result)
            append_csv(args.out, rows)

            scores = " -> ".join(f"{r.score:.2f}" for r in result.rounds) or "-"
            print(f"[{run_id}] scores: {scores}  {result.error[:80]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
