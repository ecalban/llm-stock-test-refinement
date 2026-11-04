"""Specification for testgen/mutation.py. Make these pass."""

from pathlib import Path

from testgen.mutation import MutantInfo, MutationReport, parse_mutmut_junit, run_mutmut

# --- MutationReport -------------------------------------------------------------


def make(mutant_id: int, status: str) -> MutantInfo:
    return MutantInfo(mutant_id, "calc.py", 2, status)


def test_report_groups_mutants_by_status():
    report = MutationReport(mutants=[
        make(1, "killed"),
        make(2, "survived"),
        make(3, "killed"),
        make(4, "timeout"),
    ])
    assert [m.mutant_id for m in report.killed] == [1, 3]
    assert [m.mutant_id for m in report.survived] == [2]


def test_score_counts_timeouts_as_detected():
    # A mutant that makes the tests hang was still noticed by them.
    report = MutationReport(mutants=[
        make(1, "killed"),
        make(2, "survived"),
        make(3, "timeout"),
        make(4, "survived"),
    ])
    assert report.score == 0.5


def test_score_is_zero_without_mutants():
    assert MutationReport().score == 0.0


# --- parse_mutmut_junit ------------------------------------------------------------

# Shape of `mutmut junitxml` output (mutmut 2.5.1), shortened.
JUNIT = """<?xml version="1.0" ?>
<testsuites disabled="0" errors="1" failures="1" tests="3" time="0.0">
  <testsuite disabled="0" errors="1" failures="1" name="mutmut" skipped="0" tests="3" time="0">
    <testcase name="Mutant #1" file="calc.py" line="2">
      <system-out>    return age &gt;= 18</system-out>
    </testcase>
    <testcase name="Mutant #2" file="calc.py" line="2">
      <failure type="failure" message="bad_survived">--- calc.py
+++ calc.py
@@ -1,2 +1,2 @@
 def is_adult(age):
-    return age &gt;= 18
+    return age &gt; 18
</failure>
      <system-out>    return age &gt;= 18</system-out>
    </testcase>
    <testcase name="Mutant #12" file="calc.py" line="5">
      <error type="timeout" message="bad_timeout">diff here</error>
      <system-out>    while True:</system-out>
    </testcase>
  </testsuite>
</testsuites>"""


def test_parse_reads_id_location_and_status():
    mutants = parse_mutmut_junit(JUNIT)
    assert [(m.mutant_id, m.file_path, m.line, m.status) for m in mutants] == [
        (1, "calc.py", 2, "killed"),
        (2, "calc.py", 2, "survived"),
        (12, "calc.py", 5, "timeout"),
    ]


def test_parse_keeps_diff_and_original_line():
    killed, survived, _ = parse_mutmut_junit(JUNIT)
    assert killed.diff == ""
    assert "+    return age > 18" in survived.diff
    assert survived.source_line.strip() == "return age >= 18"


# --- run_mutmut (really runs mutmut in a subprocess; a few seconds each) -----------

TARGET = "def is_adult(age):\n    return age >= 18\n"


def project(tmp_path: Path, tests: str) -> Path:
    """Create a tiny project: calc.py + test_calc.py in tmp_path."""
    (tmp_path / "calc.py").write_text(TARGET)
    (tmp_path / "test_calc.py").write_text("from calc import is_adult\n\n" + tests)
    return tmp_path


def test_run_mutmut_strong_tests_kill_everything(tmp_path: Path):
    root = project(tmp_path, "def test_boundary():\n"
                             "    assert is_adult(18) is True\n"
                             "    assert is_adult(17) is False\n")
    report = run_mutmut(Path("calc.py"), Path("test_calc.py"), cwd=root)
    assert report.error == ""
    assert len(report.mutants) > 0
    assert report.survived == []
    assert report.score == 1.0


def test_run_mutmut_weak_tests_leave_survivors(tmp_path: Path):
    root = project(tmp_path, "def test_far_from_boundary():\n"
                             "    assert is_adult(30) is True\n")
    report = run_mutmut(Path("calc.py"), Path("test_calc.py"), cwd=root)
    assert report.error == ""
    assert len(report.survived) > 0
    assert all(m.line == 2 for m in report.survived)
    assert report.score < 1.0


def test_run_mutmut_starts_fresh_each_time(tmp_path: Path):
    # mutmut caches results in .mutmut-cache; an old cache must not leak into a new run.
    root = project(tmp_path, "def test_far_from_boundary():\n"
                             "    assert is_adult(30) is True\n")
    weak = run_mutmut(Path("calc.py"), Path("test_calc.py"), cwd=root)
    (root / "test_calc.py").write_text("from calc import is_adult\n\n"
                                       "def test_boundary():\n"
                                       "    assert is_adult(18) is True\n"
                                       "    assert is_adult(17) is False\n")
    strong = run_mutmut(Path("calc.py"), Path("test_calc.py"), cwd=root)
    assert weak.score < 1.0
    assert strong.score == 1.0


def test_run_mutmut_reports_failing_baseline_as_error(tmp_path: Path):
    # If the tests already fail on the original code, mutation testing is meaningless.
    root = project(tmp_path, "def test_wrong():\n"
                             "    assert is_adult(10) is True\n")
    report = run_mutmut(Path("calc.py"), Path("test_calc.py"), cwd=root)
    assert report.mutants == []
    assert report.error != ""
