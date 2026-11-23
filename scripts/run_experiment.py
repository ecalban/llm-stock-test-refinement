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
    """Turn a file path into its import path: prediction/signals.py -> prediction.signals.

    Hint: target.with_suffix("") drops ".py"; target.parts gives the folder names.
    Use as_posix() or parts so it also works with Windows backslashes.
    """
    # TODO(Yusuf)
    raise NotImplementedError


def rows_for(
    run_id: str, started_at: str, target: Path, provider: str, model: str,
    repeat: int, result: LoopResult,
) -> list[dict]:
    """One dict per round, with exactly the keys in CSV_COLUMNS.

    - target is written as a posix path ("prediction/signals.py")
    - score is rounded to 4 decimals, llm_seconds to 2
    - error is result.error on EVERY row of the run (so a broken run is easy to filter)
    - a run that failed before round 0 finished (no rounds) still gets ONE row:
      round = -1, every number 0, error = result.error
    """
    # TODO(Yusuf)
    raise NotImplementedError


def append_csv(path: Path, rows: list[dict]) -> None:
    """Append rows to the CSV at `path`; create the folder and write the header if the
    file does not exist yet. Never write the header twice.

    Hint: csv.DictWriter(f, fieldnames=CSV_COLUMNS); open with newline="" so Windows
    does not add blank lines; check path.exists() BEFORE opening the file.
    """
    # TODO(Yusuf)
    raise NotImplementedError


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
