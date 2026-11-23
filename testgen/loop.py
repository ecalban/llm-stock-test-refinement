"""Step 5 of the test layer: the feedback loop that ties everything together.

    round 0:  LLM writes tests (no feedback) -> keep valid ones -> mutmut -> score
    round k:  surviving mutants -> feedback prompt (score / line / full) -> LLM writes
              ADDITIONAL tests -> merge -> keep valid ones -> mutmut -> score
    stop:     every mutant killed, the score did not improve, or max_rounds reached

max_rounds=0 gives the no-feedback baseline (round 0 only).

`mutate` and `run_tests` are parameters so tests/test_loop.py can replace the slow
real tools (mutmut, pytest subprocess) with fast fakes. In real runs leave the
defaults.

tests/test_loop.py describes what each function must do. Fill in the TODOs in this
order: count_tests -> should_stop -> keep_valid -> run_loop.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

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
    if scores[-1] >= 0.75:
        return True

    if len(scores) >= 2 and scores[-1] <= scores[-2]:
        return True

    if len(scores) - 1 >= max_rounds:
        return True

    return False

def keep_valid(
    test_path: Path, run_tests: Callable[[Path], RunResult] = run_pytest
) -> tuple[str, str]:
    """Run the test file, remove the tests that fail on the ORIGINAL code, rewrite the file.

    Returns (cleaned_code, error). error is the collection error when the file cannot
    even be imported; in that case the file is left unchanged and cleaned_code is "".

    Steps: result = run_tests(test_path) -> if result.collection_error: return "", that
    -> cleaned = remove_failing_tests(<file text>, result.failed) -> write it -> return.
    """
    # TODO(Yusuf)
    raise NotImplementedError


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
    """Run round 0 and then feedback rounds until should_stop says stop."""
    if fmt == "none":
        max_rounds = 0  # the baseline never gets feedback
    elif fmt not in FORMATS:
        raise ValueError(fmt)

    source = source_path.read_text()

    # --- round 0: no feedback -------------------------------------------------------
    first = generate_tests(client, source_path, module_import, dest_dir)
    result = LoopResult(fmt=fmt, test_path=first.test_path)
    code, error = keep_valid(first.test_path, run_tests)
    if error:
        result.error = error
        return result
    report = mutate(source_path, first.test_path)
    if report.error:
        result.error = report.error
        return result
    result.rounds.append(_record(0, count_tests(first.test_code), code, report, first.llm))

    # --- feedback rounds ------------------------------------------------------------
    while not should_stop([r.score for r in result.rounds], max_rounds):
        # TODO(Yusuf): one feedback round, 8 steps:
        # 1. prompt = build_feedback_prompt(source, module_import, code, report, fmt)
        # 2. llm = client.generate(prompt, system=SYSTEM_PROMPT)
        # 3. new_tests = extract_code(llm.text)
        # 4. merged = merge_tests(code, new_tests)
        #    (if the LLM returned broken Python, merge_tests raises SyntaxError:
        #     catch it, set result.error = "unparsable reply" and break)
        # 5. write merged to result.test_path
        # 6. code, error = keep_valid(result.test_path, run_tests)
        #    (on error: set result.error and break)
        # 7. report = mutate(source_path, result.test_path)
        #    (if report.error: set result.error and break)
        # 8. append _record(len(result.rounds), count_tests(new_tests), code, report, llm)
        raise NotImplementedError

    return result
