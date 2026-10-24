"""Specification for testgen/generator.py. Make these pass."""

from pathlib import Path

from testgen.generator import (
    SYSTEM_PROMPT,
    build_prompt,
    extract_code,
    generate_tests,
    write_test_file,
)
from testgen.llm_client import LLMResult

SOURCE = "def add(a, b):\n    return a + b\n"


# --- build_prompt ---------------------------------------------------------


def test_prompt_contains_source_in_code_block():
    prompt = build_prompt(SOURCE, "pkg.math_utils")
    assert "```python\n" + SOURCE in prompt


def test_prompt_contains_import_path():
    prompt = build_prompt(SOURCE, "pkg.math_utils")
    assert "pkg.math_utils" in prompt


def test_prompt_asks_for_pytest_and_edge_cases():
    prompt = build_prompt(SOURCE, "pkg.math_utils").lower()
    assert "pytest" in prompt
    assert "edge case" in prompt


# --- extract_code ---------------------------------------------------------


def test_extract_python_block():
    reply = "Here are tests:\n```python\ndef test_a():\n    assert 1\n```\nThanks!"
    assert extract_code(reply) == "def test_a():\n    assert 1\n"


def test_extract_takes_first_python_block():
    reply = "```python\nA = 1\n```\n```python\nB = 2\n```"
    assert extract_code(reply) == "A = 1\n"


def test_extract_plain_block_when_no_python_tag():
    reply = "```\ndef test_b():\n    pass\n```"
    assert extract_code(reply) == "def test_b():\n    pass\n"


def test_extract_whole_reply_when_no_block():
    assert extract_code("  def test_c():\n    pass  \n\n") == "def test_c():\n    pass\n"


# --- write_test_file ------------------------------------------------------


def test_write_creates_dir_and_prefixes_name(tmp_path: Path):
    path = write_test_file("x = 1\n", tmp_path / "generated", "indicators")
    assert path == tmp_path / "generated" / "test_indicators.py"
    assert path.read_text() == "x = 1\n"


def test_write_does_not_double_prefix(tmp_path: Path):
    path = write_test_file("x = 1\n", tmp_path, "test_indicators")
    assert path.name == "test_indicators.py"


# --- generate_tests (uses a fake LLM, no network) -------------------------


class FakeClient:
    model = "fake"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, *, system: str | None = None) -> LLMResult:
        self.calls.append((prompt, system))
        return LLMResult(self.reply, "fake", 10, 5, 0.01)


def test_generate_tests_end_to_end(tmp_path: Path):
    source = tmp_path / "math_utils.py"
    source.write_text(SOURCE)
    client = FakeClient("```python\nfrom pkg.math_utils import add\n\ndef test_add():\n    assert add(1, 2) == 3\n```")

    result = generate_tests(client, source, "pkg.math_utils", tmp_path / "generated")

    assert result.test_path == tmp_path / "generated" / "test_math_utils.py"
    assert "assert add(1, 2) == 3" in result.test_path.read_text()
    assert result.test_code == result.test_path.read_text()
    assert result.llm.prompt_tokens == 10
    prompt, system = client.calls[0]
    assert SOURCE in prompt
    assert system == SYSTEM_PROMPT
