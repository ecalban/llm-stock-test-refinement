"""Specification for the helpers in scripts/run_experiment.py. Make these pass."""

import csv
from pathlib import Path, PureWindowsPath

from scripts.run_experiment import CSV_COLUMNS, append_csv, module_name, rows_for
from testgen.loop import LoopResult, RoundResult

# --- module_name ------------------------------------------------------------------------


def test_module_name_from_posix_path():
    assert module_name(Path("prediction/signals.py")) == "prediction.signals"


def test_module_name_from_windows_path():
    assert module_name(PureWindowsPath(r"prediction\indicators.py")) == "prediction.indicators"


# --- rows_for ---------------------------------------------------------------------------


def round_(n: int, score: float) -> RoundResult:
    return RoundResult(n, 5, 4 + n, 36, 10 - n, score, 900, 400, 3.14159)


def make_rows(result: LoopResult) -> list[dict]:
    return rows_for("abc123", "2026-11-23T10:00:00+00:00", Path("prediction/signals.py"),
                    "openai", "gpt-4o-mini", 2, result)


def test_one_row_per_round_with_all_columns():
    result = LoopResult("line", Path("t.py"), [round_(0, 0.7222222), round_(1, 0.75)])
    rows = make_rows(result)
    assert len(rows) == 2
    assert all(list(row) == CSV_COLUMNS for row in rows)
    first = rows[0]
    assert first["run_id"] == "abc123"
    assert first["target"] == "prediction/signals.py"
    assert first["fmt"] == "line"
    assert first["repeat"] == 2
    assert first["round"] == 0
    assert first["score"] == 0.7222
    assert first["llm_seconds"] == 3.14
    assert first["mutants_survived"] == 10
    assert first["error"] == ""
    assert rows[1]["round"] == 1


def test_error_is_repeated_on_every_row():
    result = LoopResult("full", Path("t.py"), [round_(0, 0.5), round_(1, 0.6)], error="unparsable reply")
    assert [row["error"] for row in make_rows(result)] == ["unparsable reply"] * 2


def test_run_without_rounds_still_gets_one_row():
    result = LoopResult("none", Path("t.py"), [], error="SyntaxError in generated file")
    rows = make_rows(result)
    assert len(rows) == 1
    assert rows[0]["round"] == -1
    assert rows[0]["score"] == 0
    assert rows[0]["tests_valid"] == 0
    assert rows[0]["error"] == "SyntaxError in generated file"


# --- append_csv -------------------------------------------------------------------------


def test_append_creates_folder_and_header(tmp_path: Path):
    out = tmp_path / "results" / "experiments.csv"
    rows = make_rows(LoopResult("line", Path("t.py"), [round_(0, 0.5)]))
    append_csv(out, rows)
    with out.open(newline="") as f:
        read = list(csv.DictReader(f))
    assert list(read[0]) == CSV_COLUMNS
    assert read[0]["score"] == "0.5"


def test_append_twice_keeps_one_header(tmp_path: Path):
    out = tmp_path / "experiments.csv"
    rows = make_rows(LoopResult("line", Path("t.py"), [round_(0, 0.5), round_(1, 0.6)]))
    append_csv(out, rows)
    append_csv(out, rows)
    lines = out.read_text().splitlines()
    assert len(lines) == 5  # 1 header + 4 rows
    assert lines.count(",".join(CSV_COLUMNS)) == 1
