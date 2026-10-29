"""Specification for testgen/runner.py. Make these pass."""

from pathlib import Path

from testgen.runner import RunResult, TestOutcome, parse_junit, remove_failing_tests, run_pytest

# --- valid_ratio ------------------------------------------------------------


def test_valid_ratio_counts_passing_share():
    result = RunResult(outcomes=[
        TestOutcome("test_a", True),
        TestOutcome("test_b", False, "boom"),
        TestOutcome("test_c", True),
        TestOutcome("test_d", False, "bang"),
    ])
    assert result.valid_ratio == 0.5
    assert result.passed == ["test_a", "test_c"]
    assert result.failed == ["test_b", "test_d"]


def test_valid_ratio_is_zero_without_outcomes():
    assert RunResult().valid_ratio == 0.0


# --- parse_junit -------------------------------------------------------------

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" skipped="1" tests="4">
  <testcase classname="test_x" name="test_ok" time="0.001"/>
  <testcase classname="test_x" name="test_wrong_value" time="0.001">
    <failure message="assert 2.25 == 2.0">long traceback here</failure>
  </testcase>
  <testcase classname="test_x" name="test_broken_fixture" time="0.001">
    <error message="fixture 'db' not found">trace</error>
  </testcase>
  <testcase classname="test_x" name="test_skipped" time="0.0">
    <skipped message="not now"/>
  </testcase>
</testsuite></testsuites>"""


def test_parse_junit_reads_every_outcome():
    outcomes = parse_junit(JUNIT)
    assert outcomes == [
        TestOutcome("test_ok", True, ""),
        TestOutcome("test_wrong_value", False, "assert 2.25 == 2.0"),
        TestOutcome("test_broken_fixture", False, "fixture 'db' not found"),
    ]


# --- remove_failing_tests ------------------------------------------------------

CODE = '''import pytest
from pkg import add

def helper():
    return 1

def test_good():
    assert add(1, 2) == 3

def test_bad():
    assert add(1, 2) == 4
'''


def test_remove_failing_keeps_everything_else():
    cleaned = remove_failing_tests(CODE, ["test_bad"])
    assert "def test_bad" not in cleaned
    assert "def test_good" in cleaned
    assert "def helper" in cleaned
    assert "import pytest" in cleaned
    compile(cleaned, "<cleaned>", "exec")  # still valid Python


def test_remove_nothing_when_no_failures():
    cleaned = remove_failing_tests(CODE, [])
    assert "def test_bad" in cleaned and "def test_good" in cleaned


# --- run_pytest (really runs pytest in a subprocess) ---------------------------


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "test_sample.py"
    path.write_text(body)
    return path


def test_run_pytest_reports_pass_and_fail(tmp_path: Path):
    path = write(tmp_path, "def test_ok():\n    assert 1 + 1 == 2\n\n"
                           "def test_wrong():\n    assert 1 + 1 == 3\n")
    result = run_pytest(path)
    assert result.collection_error == ""
    assert result.passed == ["test_ok"]
    assert result.failed == ["test_wrong"]
    assert "assert" in result.outcomes[1].message


def test_run_pytest_reports_syntax_error_as_collection_error(tmp_path: Path):
    path = write(tmp_path, "def test_broken(:\n    pass\n")
    result = run_pytest(path)
    assert result.outcomes == []
    assert result.collection_error != ""
    assert result.valid_ratio == 0.0


def test_run_pytest_reports_bad_import_as_collection_error(tmp_path: Path):
    path = write(tmp_path, "from no_such_module import x\n\ndef test_a():\n    assert x\n")
    result = run_pytest(path)
    assert result.outcomes == []
    assert "no_such_module" in result.collection_error


def test_run_pytest_times_out(tmp_path: Path):
    path = write(tmp_path, "import time\n\ndef test_slow():\n    time.sleep(10)\n")
    result = run_pytest(path, timeout_seconds=1)
    assert result.collection_error == "timeout"
