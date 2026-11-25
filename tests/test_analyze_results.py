"""Specification for the helpers in scripts/analyze_results.py. Make these pass."""

import pandas as pd
import pytest

from scripts.analyze_results import clean, curves, per_run, summary

T = "prediction/signals.py"


def row(run_id, fmt, rnd, score, error=""):
    return {"run_id": run_id, "target": T, "fmt": fmt, "round": rnd, "score": score,
            "prompt_tokens": 100, "completion_tokens": 50, "error": error}


@pytest.fixture
def df():
    return pd.DataFrame([
        row("c6855345", "none", 0, 0.72),  # old pilot row: no "-" in run_id
        row("aaaa-none", "none", 0, 0.5),
        row("aaaa-score", "score", 0, 0.5), row("aaaa-score", "score", 1, 0.5),
        row("aaaa-line", "line", 0, 0.5), row("aaaa-line", "line", 1, 0.7), row("aaaa-line", "line", 2, 0.8),
        row("aaaa-full", "full", 0, 0.5), row("aaaa-full", "full", 1, 0.6),
        row("bbbb-none", "none", 0, 0.6),
        row("bbbb-line", "line", 1, 0.8), row("bbbb-line", "line", 0, 0.6),  # unsorted on purpose
        row("cccc-full", "full", -1, 0.0, error="SyntaxError"),  # broke before round 0
    ])


# --- clean -------------------------------------------------------------------------------


def test_clean_drops_pilot_and_broken_rows(df):
    out = clean(df)
    assert "c6855345" not in set(out["run_id"])
    assert "cccc-full" not in set(out["run_id"])
    assert len(out) == 11


def test_clean_adds_start_id_without_touching_input(df):
    out = clean(df)
    assert set(out["start_id"]) == {"aaaa", "bbbb"}
    assert "start_id" not in df.columns


# --- per_run -----------------------------------------------------------------------------


def runs_by_id(df):
    return per_run(clean(df)).set_index("run_id")


def test_per_run_has_one_row_per_run(df):
    runs = per_run(clean(df))
    assert sorted(runs["run_id"]) == ["aaaa-full", "aaaa-line", "aaaa-none", "aaaa-score",
                                      "bbbb-line", "bbbb-none"]
    for column in ["target", "fmt", "start_id", "start_score", "final_score", "gain", "rounds", "tokens"]:
        assert column in runs.columns


def test_per_run_start_final_gain_rounds(df):
    line = runs_by_id(df).loc["aaaa-line"]
    assert line["start_score"] == 0.5
    assert line["final_score"] == 0.8
    assert line["gain"] == pytest.approx(0.3)
    assert line["rounds"] == 2


def test_per_run_respects_round_order_not_row_order(df):
    line = runs_by_id(df).loc["bbbb-line"]
    assert (line["start_score"], line["final_score"]) == (0.6, 0.8)


def test_per_run_counts_shared_round_zero_tokens_only_once(df):
    runs = runs_by_id(df)
    assert runs.loc["aaaa-none", "tokens"] == 150  # round 0 is paid by "none"
    assert runs.loc["aaaa-line", "tokens"] == 300  # rounds 1 and 2 only
    assert runs.loc["aaaa-full", "tokens"] == 150


# --- summary -----------------------------------------------------------------------------


def test_summary_rows_in_format_order(df):
    table = summary(per_run(clean(df)))
    assert list(table["fmt"]) == ["none", "score", "line", "full"]
    assert list(table.columns) == ["target", "fmt", "n", "start_mean", "final_mean", "gain_mean",
                                   "gain_std", "rounds_mean", "tokens_mean"]


def test_summary_values(df):
    table = summary(per_run(clean(df))).set_index("fmt")
    line = table.loc["line"]
    assert line["n"] == 2
    assert line["start_mean"] == 0.55
    assert line["final_mean"] == 0.8
    assert line["gain_mean"] == 0.25
    assert line["gain_std"] == 0.071
    assert line["rounds_mean"] == 1.5
    assert line["tokens_mean"] == 225
    assert table.loc["full", "gain_std"] == 0.0  # one run only


# --- curves ------------------------------------------------------------------------------


def test_curves_forward_fills_runs_that_stopped_early(df):
    lines = curves(clean(df), max_round=2)
    assert list(lines.index) == [0, 1, 2]
    assert lines["line"].tolist() == pytest.approx([0.55, 0.75, 0.8])
    assert lines["none"].tolist() == pytest.approx([0.55, 0.55, 0.55])
    assert lines["full"].tolist() == pytest.approx([0.5, 0.6, 0.6])
