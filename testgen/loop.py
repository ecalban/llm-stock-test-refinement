"""Step 5 of the test layer: the feedback loop that ties everything together.

    round 0:  LLM writes tests (no feedback) -> keep valid ones -> mutmut -> score
    round k:  surviving mutants -> feedback prompt (score / line / full) -> LLM writes
              ADDITIONAL tests -> merge -> keep valid ones -> mutmut -> score
    stop:     every mutant killed, the score did not improve, or max_rounds reached

max_rounds=0 gives the no-feedback baseline (round 0 only).

Fair comparison of formats: round 0 is noisy (the same prompt can score 0.50 one
time and 0.72 the next). So an experiment runs round 0 ONCE with start_round() and
then continues every format from that same starting file with continue_loop().
Any difference between formats then comes from the feedback alone.

`mutate` and `run_tests` are parameters so tests/test_loop.py can replace the slow
real tools (mutmut, pytest subprocess) with fast fakes. In real runs leave the
defaults.

tests/test_loop.py describes what each function must do. Fill in the TODOs in this
order: count_tests -> should_stop -> keep_valid -> run_loop.
"""

from __future__ import annotations

import ast
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from unittest import result

from sqlalchemy.util import counter

from testgen.feedback import FORMATS, build_feedback_prompt, merge_tests
from testgen.generator import SYSTEM_PROMPT, extract_code, generate_tests
from testgen.llm_client import LLMClient, LLMResult
from testgen.mutation import MutationReport, run_mutmut
from testgen.runner import RunResult, remove_failing_tests, run_pytest


@dataclass(frozen=True)
class RoundResult:
    """What happened in one round. Becomes one row in the results table."""

    round: int
    tests_generated: int  # test functions in the LLM reply
    tests_valid: int  # test functions in the file after cleaning (all rounds so far)
    mutants_total: int
    mutants_survived: int
    score: float
    prompt_tokens: int
    completion_tokens: int
    duration_seconds: float  # LLM time only


@dataclass
class LoopResult:
    fmt: str  # "none" for the baseline, otherwise one of FORMATS
    test_path: Path
    rounds: list[RoundResult] = field(default_factory=list)
    error: str = ""  # set when the loop had to stop because of a broken file or mutmut error

    @property
    def final_score(self) -> float:
        return self.rounds[-1].score if self.rounds else 0.0


def count_tests(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0

    counter = 0
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            counter += 1
    return counter


def should_stop(scores: list[float], max_rounds: int) -> bool:
    if scores[-1] >= 1.0:
        return True

    if len(scores) >= 2 and scores[-1] <= scores[-2]:
        return True

    if len(scores) - 1 >= max_rounds:
        return True

    return False

def keep_valid(
    test_path: Path, run_tests: Callable[[Path], RunResult] = run_pytest
) -> tuple[str, str]:

    results = run_tests(test_path)

    if results.collection_error:
        return "", results.collection_error

    cleaned = remove_failing_tests(test_path.read_text(), results.failed)

    test_path.write_text(cleaned)
    return cleaned, ""


def _record(round_no: int, generated: int, code: str, report: MutationReport, llm: LLMResult) -> RoundResult:
    """Build a RoundResult. Given."""
    return RoundResult(
        round=round_no,
        tests_generated=generated,
        tests_valid=count_tests(code),
        mutants_total=len(report.mutants),
        mutants_survived=len(report.survived),
        score=report.score,
        prompt_tokens=llm.prompt_tokens,
        completion_tokens=llm.completion_tokens,
        duration_seconds=llm.duration_seconds,
    )


@dataclass
class Start:
    """Round 0, shared by every format of one experiment repeat."""

    test_path: Path  # the cleaned round-0 test file (never modified afterwards)
    code: str = ""
    report: MutationReport = field(default_factory=MutationReport)
    record: RoundResult | None = None
    error: str = ""


def start_round(
    client: LLMClient,
    source_path: Path,
    module_import: str,
    dest_dir: Path,
    mutate: Callable[[Path, Path], MutationReport] = run_mutmut,
    run_tests: Callable[[Path], RunResult] = run_pytest,
) -> Start:
    """Round 0: no feedback. Given (this is the old beginning of run_loop)."""
    first = generate_tests(client, source_path, module_import, dest_dir)
    code, error = keep_valid(first.test_path, run_tests)
    if error:
        return Start(first.test_path, error=error)
    report = mutate(source_path, first.test_path)
    if report.error:
        return Start(first.test_path, error=report.error)
    record = _record(0, count_tests(first.test_code), code, report, first.llm)
    return Start(first.test_path, code, report, record)


def continue_loop(
    client: LLMClient,
    source_path: Path,
    module_import: str,
    start: Start,
    dest_dir: Path,
    fmt: str = "none",
    max_rounds: int = 5,
    mutate: Callable[[Path, Path], MutationReport] = run_mutmut,
    run_tests: Callable[[Path], RunResult] = run_pytest,
) -> LoopResult:
    """Continue from a shared round 0 with feedback format `fmt`.

    1. fmt check: "none" -> max_rounds = 0; unknown fmt -> ValueError (as before)
    2. result = LoopResult(fmt=fmt, test_path=dest_dir / start.test_path.name)
    3. if start.error: copy it into result.error and return result (no rounds)
    4. copy the start file to result.test_path, so every format works on ITS OWN copy
       and the start file stays untouched:
           dest_dir.mkdir(parents=True, exist_ok=True)
           if result.test_path != start.test_path: shutil.copyfile(start.test_path, result.test_path)
    5. result.rounds.append(start.record); code, report = start.code, start.report
    The feedback while-loop below is your code from run_loop, moved here unchanged.
    """
    source = source_path.read_text()

    if fmt == "none":
        max_rounds = 0
    elif fmt not in FORMATS:
        raise ValueError(fmt)


    result = LoopResult(fmt=fmt, test_path=dest_dir / start.test_path.name)

    if start.error:
        result.error = start.error
        return result

    dest_dir.mkdir(parents=True, exist_ok=True)
    if result.test_path != start.test_path:
        shutil.copyfile(start.test_path, result.test_path)

    result.rounds.append(start.record)
    code, report = start.code, start.report

    while not should_stop([r.score for r in result.rounds], max_rounds):
        prompt = build_feedback_prompt(source, module_import, code, report, fmt)
        llm = client.generate(prompt, system=SYSTEM_PROMPT)
        new_tests = extract_code(llm.text)

        # 4. merge; a broken reply ends the loop
        try:
            merged = merge_tests(code, new_tests)
        except SyntaxError:
            result.error = "unparsable reply"
            break

        # 5. save the merged file
        result.test_path.write_text(merged)

        # 6. drop tests that fail on the original code
        code, error = keep_valid(result.test_path, run_tests)
        if error:
            result.error = error
            break

        # 7. measure
        report = mutate(source_path, result.test_path)
        if report.error:
            result.error = report.error
            break

        # 8. remember this round
        result.rounds.append(_record(len(result.rounds), count_tests(new_tests), code, report, llm))

    return result


def run_loop(
    client: LLMClient,
    source_path: Path,
    module_import: str,
    dest_dir: Path,
    fmt: str = "none",
    max_rounds: int = 5,
    mutate: Callable[[Path, Path], MutationReport] = run_mutmut,
    run_tests: Callable[[Path], RunResult] = run_pytest,
) -> LoopResult:
    """One format from a fresh round 0. Given: start_round + continue_loop."""
    if fmt != "none" and fmt not in FORMATS:
        raise ValueError(fmt)  # fail before spending an LLM call
    start = start_round(client, source_path, module_import, dest_dir, mutate, run_tests)
    return continue_loop(client, source_path, module_import, start, dest_dir,
                         fmt, max_rounds, mutate, run_tests)
