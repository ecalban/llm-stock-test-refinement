"""Step 3 of the test layer: measure how good the valid tests are with mutation testing.

mutmut makes small changes ("mutants") to the code under test, e.g. `>=` -> `>`,
and runs the tests against each one:
- the tests fail      -> mutant KILLED   (good: the tests noticed the bug)
- the tests pass      -> mutant SURVIVED (bad: this bug would go unnoticed)
- the tests hang      -> mutant TIMEOUT  (counted as detected)

The survived mutants are exactly what the feedback loop shows the LLM next.

Flow:
    target + tests --run_mutmut--> `mutmut run` --> `mutmut junitxml` --parse_mutmut_junit--> MutationReport

tests/test_mutation.py describes what each function must do. Fill in the TODOs
in this order: killed/survived/score -> parse_mutmut_junit -> run_mutmut.
"""

from __future__ import annotations

import subprocess
import sys
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from unittest import result


@dataclass(frozen=True)
class MutantInfo:
    """One mutant and what happened to it."""

    mutant_id: int  # mutmut's number, e.g. 12 for "Mutant #12"
    file_path: str  # file that was mutated
    line: int  # 1-based line number of the change
    status: str  # "killed" | "survived" | "timeout"
    diff: str = ""  # unified diff of the change (empty for killed mutants)
    source_line: str = ""  # the original line of code


@dataclass
class MutationReport:
    """Result of one mutmut run."""

    mutants: list[MutantInfo] = field(default_factory=list)
    error: str = ""  # set when mutmut could not run (e.g. tests fail on the original code)

    @property
    def killed(self) -> list[MutantInfo]:
        result = []
        for mutant in self.mutants:
            if mutant.status == "killed":
                result.append(mutant)
        return result

    @property
    def survived(self) -> list[MutantInfo]:
        result = []
        for mutant in self.mutants:
            if mutant.status == "survived":
                result.append(mutant)
        return result

    @property
    def score(self) -> float:
        """Mutation score = detected / total, where detected = killed + timeout.

        Return 0.0 when there are no mutants.
        (Equivalent mutants are handled later, in the analysis, not here.)
        """
        if not self.mutants:
            return 0.0

        score = (len(self.mutants) - len(self.survived)) / len(self.mutants)
        return score


def parse_mutmut_junit(xml_text: str) -> list[MutantInfo]:
    """Turn the output of `mutmut junitxml` into a list of MutantInfo.

    Every mutant is a <testcase name="Mutant #12" file="..." line="..."> element:
    - no <failure>/<error> child  -> "killed"
    - a <failure> child           -> "survived", diff = the element's text
    - an <error> child            -> "timeout",  diff = the element's text
    The <system-out> child holds the original line of code.

    Hints:
    - Same pattern as parse_junit in runner.py.
    - The id is the part of the name after "#": int(name.split("#")[1]).
    - element.text is the text inside a tag; it can be None, so use `element.text or ""`.
    """
    mutants = []
    for case in ET.fromstring(xml_text).iter("testcase"):
        mutant_id = int(case.get("name").split("#")[1])

        failure = case.find("failure")
        error = case.find("error")
        if failure is not None:
            status = "survived"
            diff = failure.text or ""
        elif error is not None:
            status = "timeout"
            diff = error.text or ""
        else:
            status = "killed"
            diff = ""


        out = case.find("system-out")
        source_line = out.text or "" if out is not None else ""

        mutants.append(MutantInfo(
            mutant_id=mutant_id,
            file_path=case.get("file"),
            line=int(case.get("line")),
            diff=diff,
            status=status,
            source_line=source_line,
        ))
    return mutants


def run_mutmut(
    target: Path, tests: Path, cwd: Path = Path("."), timeout_seconds: float = 900.0
) -> MutationReport:
    """Run mutmut on `target` using only the tests in `tests` and return the report.

    `target` and `tests` are relative to `cwd` (the project folder mutmut runs in).
    This function is given; read it, it uses the same ideas as run_pytest.
    """
    # Old results would mix with the new run: always start from a clean cache.
    cache = cwd / ".mutmut-cache"
    if cache.exists():
        cache.unlink()

    # The command mutmut runs for every mutant. -x stops at the first failure (faster).
    runner = f'"{sys.executable}" -m pytest -x -q -p no:cacheprovider {tests.as_posix()}'
    run_command = [
        sys.executable, "-m", "mutmut", "run",
        "--paths-to-mutate", target.as_posix(),
        "--runner", runner,
        "--tests-dir", tests.parent.as_posix(),  # mutmut refuses to start without a tests folder
        "--no-progress",
        "--CI",  # exit code 0 = finished (even with survivors), 1 = fatal error
        "--no-progress",
        "--simple-output",  # plain text instead of emojis (Windows consoles choke on them)
    ]
    # Force UTF-8 so code with non-ASCII characters does not crash on Windows
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        run = subprocess.run(
            run_command, cwd=cwd, capture_output=True, text=True, timeout=timeout_seconds,
            env=env, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return MutationReport(error="timeout")

    if run.returncode != 0:
        # Most common cause: the tests already fail on the original code.
        return MutationReport(error=(run.stdout + run.stderr)[-2000:])

    report = subprocess.run(
        [sys.executable, "-m", "mutmut", "junitxml"],
        cwd=cwd, capture_output=True, text=True, timeout=120,
        env=env, encoding="utf-8", errors="replace",
    )
    return MutationReport(mutants=parse_mutmut_junit(report.stdout))
