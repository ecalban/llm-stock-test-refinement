"""Specification for testgen/loop.py. Make these pass.

The loop tests use a fake LLM and a fake mutation tool, so they need no API key
and run in a few seconds. pytest itself really runs on the generated files.
"""

import ast
from pathlib import Path

import pytest

from testgen.llm_client import LLMResult
from testgen.loop import count_tests, keep_valid, run_loop, should_stop
from testgen.mutation import MutantInfo, MutationReport

# --- count_tests -------------------------------------------------------------------


def test_count_tests_counts_only_test_functions():
    code = "import pytest\n\ndef helper():\n    pass\n\ndef test_a():\n    pass\n\ndef test_b():\n    pass\n"
    assert count_tests(code) == 2


def test_count_tests_is_zero_for_broken_code():
    assert count_tests("def test_a(:\n    pass\n") == 0


# --- should_stop -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("scores", "max_rounds", "stop"),
    [
        ([0.5], 5, False),  # only round 0 so far: go on
        ([0.5], 0, True),  # baseline: no feedback rounds at all
        ([1.0], 5, True),  # everything killed already
        ([0.5, 0.6], 5, False),  # improved: go on
        ([0.5, 0.5], 5, True),  # no improvement
        ([0.5, 0.4], 5, True),  # got worse
        ([0.1, 0.2, 0.3], 2, True),  # two feedback rounds done, max is 2
        ([0.1, 0.2, 0.3], 3, False),
        ([0.5, 0.75], 5, False),  # improved but not finished: go on
    ],
)
def test_should_stop(scores, max_rounds, stop):
    assert should_stop(scores, max_rounds) is stop


# --- keep_valid --------------------------------------------------------------------


def test_keep_valid_removes_failing_tests_and_rewrites_file(tmp_path: Path):
    path = tmp_path / "test_x.py"
    path.write_text("def test_ok():\n    assert 1 == 1\n\ndef test_bad():\n    assert 1 == 2\n")
    code, error = keep_valid(path)
    assert error == ""
    assert "test_ok" in code and "test_bad" not in code
    assert path.read_text() == code


def test_keep_valid_reports_broken_file(tmp_path: Path):
    path = tmp_path / "test_x.py"
    path.write_text("def test_broken(:\n    pass\n")
    code, error = keep_valid(path)
    assert code == ""
    assert error != ""
    assert path.read_text() == "def test_broken(:\n    pass\n"  # unchanged


# --- run_loop (fake LLM + fake mutmut, real pytest) ---------------------------------


def block(body: str) -> str:
    return "```python\n" + body + "```"


class FakeClient:
    """Returns the given replies in order and remembers every prompt."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResult:
        self.prompts.append(prompt)
        return LLMResult(self.replies.pop(0), "fake", 100, 50, 0.1)


def fake_mutate(target: Path, tests: Path) -> MutationReport:
    """4 mutants; each test function in the file kills one of them."""
    killed = min(count_tests(tests.read_text()), 4)
    return MutationReport(mutants=[
        MutantInfo(i, "calc.py", i, "killed" if i <= killed else "survived",
                   f"-    line {i}\n+    mutated {i}", f"    line {i}")
        for i in range(1, 5)
    ])


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "calc.py"
    path.write_text("def is_adult(age):\n    return age >= 18\n")
    return path


ROUND0 = block("def test_a():\n    assert 1 == 1\n\ndef test_wrong():\n    assert 1 == 2\n")
ROUND1 = block("def test_b():\n    assert 2 == 2\n\ndef test_c():\n    assert 3 == 3\n")
ROUND2 = block("def test_a():\n    assert 4 == 4\n")  # name clash with round 0 -> renamed


def test_baseline_runs_round_zero_only(tmp_path: Path, source: Path):
    client = FakeClient(ROUND0)
    result = run_loop(client, source, "calc", tmp_path / "gen", fmt="none", mutate=fake_mutate)
    assert result.error == ""
    assert len(result.rounds) == 1
    first = result.rounds[0]
    assert (first.round, first.tests_generated, first.tests_valid) == (0, 2, 1)
    assert (first.mutants_total, first.mutants_survived, first.score) == (4, 3, 0.25)
    assert (first.prompt_tokens, first.completion_tokens) == (100, 50)
    assert len(client.prompts) == 1


def test_loop_runs_until_every_mutant_is_killed(tmp_path: Path, source: Path):
    client = FakeClient(ROUND0, ROUND1, ROUND2)
    result = run_loop(client, source, "calc", tmp_path / "gen", fmt="line", mutate=fake_mutate)
    assert result.error == ""
    assert [r.score for r in result.rounds] == [0.25, 0.75, 1.0]
    assert [r.tests_valid for r in result.rounds] == [1, 3, 4]
    assert result.final_score == 1.0
    tree = ast.parse(result.test_path.read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert names == ["test_a", "test_b", "test_c", "test_a_2"]


def test_feedback_prompt_uses_the_chosen_format(tmp_path: Path, source: Path):
    client = FakeClient(ROUND0, ROUND1, ROUND2)
    run_loop(client, source, "calc", tmp_path / "gen", fmt="line", mutate=fake_mutate)
    second = client.prompts[1]
    assert "Current mutation score: 0.25" in second
    assert "Line 2: `line 2`" in second
    assert "def test_a" in second  # the current tests are in the prompt
    assert "test_wrong" not in second  # the failing test was removed before


def test_loop_stops_when_the_score_does_not_improve(tmp_path: Path, source: Path):
    useless = block("def test_x():\n    assert 1 == 5\n")  # fails on the original code
    client = FakeClient(ROUND0, useless, ROUND1)
    result = run_loop(client, source, "calc", tmp_path / "gen", fmt="full", mutate=fake_mutate)
    assert [r.score for r in result.rounds] == [0.25, 0.25]
    assert result.rounds[1].tests_generated == 1
    assert result.rounds[1].tests_valid == 1
    assert len(client.prompts) == 2  # ROUND1 was never requested


def test_loop_respects_max_rounds(tmp_path: Path, source: Path):
    client = FakeClient(ROUND0, ROUND1, ROUND2)
    result = run_loop(client, source, "calc", tmp_path / "gen", fmt="score", max_rounds=1, mutate=fake_mutate)
    assert len(result.rounds) == 2


def test_loop_stops_on_an_unparsable_reply(tmp_path: Path, source: Path):
    client = FakeClient(ROUND0, block("def test_broken(:\n    pass\n"))
    result = run_loop(client, source, "calc", tmp_path / "gen", fmt="line", mutate=fake_mutate)
    assert result.error != ""
    assert len(result.rounds) == 1
    assert "test_a" in result.test_path.read_text()  # the good file was not overwritten


def test_unknown_format_is_rejected(tmp_path: Path, source: Path):
    with pytest.raises(ValueError):
        run_loop(FakeClient(ROUND0), source, "calc", tmp_path / "gen", fmt="everything", mutate=fake_mutate)


# --- shared round 0: start_round + continue_loop ------------------------------------

from testgen.loop import continue_loop, start_round  # noqa: E402


def test_formats_continue_from_the_same_start(tmp_path: Path, source: Path):
    client = FakeClient(ROUND0, ROUND1, ROUND1)
    start = start_round(client, source, "calc", tmp_path / "start", mutate=fake_mutate)
    start_text = start.test_path.read_text()

    line = continue_loop(client, source, "calc", start, tmp_path / "line", fmt="line",
                         max_rounds=1, mutate=fake_mutate)
    full = continue_loop(client, source, "calc", start, tmp_path / "full", fmt="full",
                         max_rounds=1, mutate=fake_mutate)

    assert line.rounds[0] == full.rounds[0] == start.record  # identical round 0
    assert [r.score for r in line.rounds] == [0.25, 0.75]
    assert [r.score for r in full.rounds] == [0.25, 0.75]
    assert line.test_path == tmp_path / "line" / "test_calc.py"
    assert full.test_path == tmp_path / "full" / "test_calc.py"
    assert start.test_path.read_text() == start_text  # the start file was not touched
    assert len(client.prompts) == 3  # round 0 once + one feedback round per format


def test_continue_with_none_makes_no_llm_call(tmp_path: Path, source: Path):
    client = FakeClient(ROUND0)
    start = start_round(client, source, "calc", tmp_path / "start", mutate=fake_mutate)
    result = continue_loop(client, source, "calc", start, tmp_path / "none", fmt="none",
                           mutate=fake_mutate)
    assert result.rounds == [start.record]
    assert result.test_path.read_text() == start.test_path.read_text()
    assert len(client.prompts) == 1


def test_start_error_is_passed_on(tmp_path: Path, source: Path):
    client = FakeClient(block("def test_broken(:\n    pass\n"))
    start = start_round(client, source, "calc", tmp_path / "start", mutate=fake_mutate)
    assert start.error != "" and start.record is None
    result = continue_loop(client, source, "calc", start, tmp_path / "line", fmt="line",
                           mutate=fake_mutate)
    assert result.error == start.error
    assert result.rounds == []
    assert len(client.prompts) == 1


def test_continue_rejects_unknown_format(tmp_path: Path, source: Path):
    start = start_round(FakeClient(ROUND0), source, "calc", tmp_path / "start", mutate=fake_mutate)
    with pytest.raises(ValueError):
        continue_loop(FakeClient(), source, "calc", start, tmp_path / "x", fmt="everything",
                      mutate=fake_mutate)
