"""Step 2 of the test layer: run generated tests and keep only the valid ones.

A generated test is "valid" when it passes on the ORIGINAL code. Tests that fail
on the original code are oracle errors (the LLM guessed a wrong expected value);
mutmut cannot use them, so they are removed before mutation testing.

Flow:
    test file --run_pytest--> RunResult (which test functions passed / failed and why)
    test code + failing names --remove_failing_tests--> cleaned test code

tests/test_runner.py describes what each function must do. Fill in the TODOs
from easiest to hardest: valid_ratio -> parse_junit -> remove_failing_tests -> run_pytest.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TestOutcome:
    """Result of one test function."""

    __test__ = False  # keep pytest from collecting this class

    name: str  # function name, e.g. "test_sma_basic"
    passed: bool
    message: str = ""  # failure/error message, empty when passed


@dataclass
class RunResult:
    """Result of running one test file."""

    outcomes: list[TestOutcome] = field(default_factory=list)
    collection_error: str = ""  # set when the file could not even be imported (syntax error, bad import)

    @property
    def passed(self) -> list[str]:
        return [o.name for o in self.outcomes if o.passed]

    @property
    def failed(self) -> list[str]:
        return [o.name for o in self.outcomes if not o.passed]

    @property
    def valid_ratio(self) -> float:
        """Share of test functions that pass on the original code (0.0 - 1.0).

        Return 0.0 when there are no outcomes (nothing collected).
        """
        if not self.outcomes:
            return 0.0
        else:
            return len(self.passed) / len(self.outcomes)


def parse_junit(xml_text: str) -> list[TestOutcome]:
    """Turn pytest's JUnit XML report into a list of TestOutcome.

    In the XML every test is a <testcase name="..."> element.
    - No child element                  -> passed
    - A <failure message="..."> child   -> failed, message = that attribute
    - An <error message="..."> child    -> failed, message = that attribute
    - A <skipped> child                 -> ignore the test (do not include it)

    Hint: ET.fromstring(xml_text).iter("testcase") visits every testcase;
    case.find("failure") returns the child element or None; element.get("message", "").
    """
    outcomes = []
    for case in ET.fromstring(xml_text).iter("testcase"):
        name = case.get("name", "")

        #Skipped tests are neither valid nor invalid: leave them out
        if case.find("skipped") is not None:
            continue

        #A failed test has a <failure> child, a broken one an <error> child
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")

        if problem is None:
            outcomes.append(TestOutcome(name=name,passed=True,message=""))
        else:
            outcomes.append(TestOutcome(name=name, passed=False, message=problem.get("message", "")))


    return outcomes


def remove_failing_tests(code: str, failing: list[str]) -> str:
    """Return `code` without the top-level test functions whose names are in `failing`.

    Everything else (imports, helpers, passing tests) stays.
    Hint: tree = ast.parse(code); keep the nodes of tree.body that are NOT
    (an ast.FunctionDef whose .name is in failing); return ast.unparse(tree) + "\\n".
    Note: ast.unparse drops comments; that is fine for generated tests.
    """
    tree = ast.parse(code)

    kept = []
    for node in tree.body:
        # Drop the node only if it is a function AND its name is `failing`
        if isinstance(node, ast.FunctionDef) and node.name in failing:
            continue
        kept.append(node)

    tree.body = kept
    return ast.unparse(tree) + "\n"


def run_pytest(test_path: Path, timeout_seconds: float = 120.0) -> RunResult:
    """Run pytest on one file in a separate process and return the per-test results.

    Steps:
    1. Create a temporary folder for the JUnit report (tempfile.TemporaryDirectory()).
    2. Run: [sys.executable, "-m", "pytest", str(test_path), "-q",
             "-p", "no:cacheprovider", f"--junitxml={report_path}"]
       with subprocess.run(..., capture_output=True, text=True, timeout=timeout_seconds).
    3. If the report file does not exist, or pytest's return code is 2, 3 or 4
       (interrupted / internal error / usage error), the file could not be collected:
       return RunResult(collection_error=<last 2000 chars of stdout + stderr>).
    4. Otherwise parse the report with parse_junit and return RunResult(outcomes=...).
       If parse_junit finds no testcases but the run output mentions "error",
       treat it as a collection error too.
    5. If the process times out (subprocess.TimeoutExpired), return
       RunResult(collection_error="timeout").
    """
    # TODO(Yusuf)
    raise NotImplementedError
