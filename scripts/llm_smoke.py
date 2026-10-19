"""Send one short prompt to a provider and print the reply, tokens and time.

Usage:
    python -m scripts.llm_smoke openai
    python -m scripts.llm_smoke ollama
"""

import sys

from testgen.llm_client import create_client

PROMPT = """Write one pytest test for this function. Reply with code only.

def sma(values, window):
    return sum(values[-window:]) / window
"""


def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    client = create_client(provider)
    result = client.generate(PROMPT)
    print(result.text)
    print(
        f"\n[{result.model}] prompt={result.prompt_tokens} "
        f"completion={result.completion_tokens} time={result.duration_seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
