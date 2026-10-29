"""Generate tests for the practice target with a real LLM and print a short report.

Usage:
    python -m scripts.generate_demo            # OpenAI (needs LLM_API_KEY in .env)
    python -m scripts.generate_demo ollama     # local model
"""

import sys
from pathlib import Path

from testgen.generator import generate_tests
from testgen.llm_client import create_client


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


if __name__ == "__main__":
    main()