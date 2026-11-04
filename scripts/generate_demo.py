"""Generate tests for the practice target with a real LLM and print a short report.

Usage:
    python -m scripts.generate_demo            # OpenAI (needs LLM_API_KEY in .env)
    python -m scripts.generate_demo ollama     # local model
"""

import sys
from pathlib import Path

from testgen.generator import generate_tests
from testgen.llm_client import create_client
from testgen.runner import remove_failing_tests, run_pytest
from testgen.mutation import run_mutmut


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"

    result = generate_tests(
        create_client(provider),
        Path("testgen/sample_target/indicators.py"),
        "testgen.sample_target.indicators",
        Path("tests/generated"),
    )

    print(result.test_code)
    print("-" * 60)
    print(f"Saved to: {result.test_path}")
    print(f"Model: {result.llm.model}")
    print(f"Tokens: prompt={result.llm.prompt_tokens}, completion={result.llm.completion_tokens}")
    print(f"Time: {result.llm.duration_seconds:.1f}s")

    first = run_pytest(result.test_path)
    if first.collection_error:
        print("Could not collect the generated file:")
        print(first.collection_error)
        return

    print(f"Generated tests: {len(first.outcomes)}")
    print(f"Valid (passing): {len(first.passed)} ratio={first.valid_ratio:.2f}")
    for outcome in first.outcomes:
        if not outcome.passed:
            print(f" removed {outcome.name}: {outcome.message[:80]}")

    cleaned = remove_failing_tests(result.test_code, first.failed)
    result.test_path.write_text(cleaned)

    second = run_pytest(result.test_path)
    print(f"After cleaning: {len(second.passed)} passed, {len(second.failed)} failed")


    print("Running mutmut (this can take 1-2 minutes)...")
    third = run_mutmut(Path("testgen/sample_target/indicators.py"), result.test_path)
    if third.error:
        print("mutmut failed:")
        print(third.error)
        return

    print(f"Mutants: {len(third.mutants)} killed: {len(third.killed)} "
          f"survived: {len(third.survived)} score={third.score:.2f}")

    #Show a few survivors: these are what the feedback loop will send to the LLM
    for m in third.survived[:5]:
        print(f"  line {m.line}: {m.source_line.strip()}")

if __name__ == "__main__":
    main()