"""Step 1 of the test layer: ask an LLM to write pytest tests for a module.

Flow:
    source code --build_prompt--> prompt --client.generate--> reply
    reply --extract_code--> test code --write_test_file--> tests/generated/test_x.py

Each function below has a TODO. tests/test_generator.py describes exactly what
each one must do; run `pytest tests/test_generator.py` until it is green.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from testgen.llm_client import LLMClient, LLMResult

SYSTEM_PROMPT = (
    "You are a senior Python engineer who writes thorough pytest unit tests. "
    "Reply with a single ```python code block and nothing else."
)


@dataclass(frozen=True)
class GenerationResult:
    test_path: Path
    test_code: str
    llm: LLMResult


def build_prompt(source_code: str, module_import: str) -> str:
    """Build the user prompt for the first (feedback-free) generation round.

    It must contain:
    - the full source code inside a python code block,
    - the import path the tests must use (e.g. ``testgen.sample_target.indicators``),
    - the instructions: pytest style, check exact return values and edge cases,
      do not modify the code under test.
    """
    fence = "`" * 3  # three backticks; kept in a variable so the string stays readable

    instructions = (
        "Write pytest unit tests for the Python module below.\n"
        f"Import the functions from `{module_import}`.\n"
        "Check exact return values and edge cases "
        "(empty input, invalid arguments, boundary values).\n"
        "Do not modify the code under test.\n"
    )

    code_block = f"{fence}python\n{source_code}{fence}\n"

    return instructions + "\n" + code_block


def extract_code(reply: str) -> str:
    """..."""  # keep the existing docstring
    fence = "`" * 3

    # 1) A ```python block: take the inside of the first one
    match = re.search(rf"{fence}python\n(.*?){fence}", reply, re.DOTALL)

    # 2) Otherwise a plain ``` block
    if match is None:
        match = re.search(rf"{fence}\n(.*?){fence}", reply, re.DOTALL)

    # 3) Otherwise assume the whole reply is code
    if match is None:
        code = reply
    else:
        code = match.group(1)

    # 4) Normalise whitespace: no leading/trailing blank lines, exactly one final newline
    return code.strip() + "\n"


def write_test_file(code: str, dest_dir: Path, name: str) -> Path:
    """Write `code` to dest_dir/test_<name>.py (create dest_dir if missing) and return the path.

    If `name` already starts with "test_", do not add the prefix twice.
    """
    # Create the folder (and any missing parents); do nothing if it already exists
    dest_dir.mkdir(parents=True, exist_ok=True)

    # pytest only collects files named test_*.py, so make sure the prefix is there once
    if name.startswith("test_"):
        filename = name + ".py"
    else:
        filename = "test_" + name + ".py"

    path = dest_dir / filename  # Path's "/" joins folder and file name
    path.write_text(code)
    return path


def generate_tests(
    client: LLMClient, source_path: Path, module_import: str, dest_dir: Path
) -> GenerationResult:
    """Read the source file, call the LLM once, save the extracted tests.

    The test file name is the source file's stem (indicators.py -> test_indicators.py).
    Pass SYSTEM_PROMPT as the system message.
    """
    # TODO(Yusuf): combine the three functions above with client.generate(...)
    raise NotImplementedError
