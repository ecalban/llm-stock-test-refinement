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
    """...""" #keep the existing docstring
    source_code = source_path.read_text() # 1. read the module under test
    prompt = build_prompt(source_code, module_import) # 2. build the prompt
    llm_result = client.generate(prompt, system=SYSTEM_PROMPT) # 3. ask the LLM
    test_code = extract_code(llm_result.text) # 4. keep only the code
    test_path = write_test_file(test_code, dest_dir, source_path.stem) # 5. save it

    return GenerationResult(test_path, test_code, llm_result)

    raise NotImplementedError
