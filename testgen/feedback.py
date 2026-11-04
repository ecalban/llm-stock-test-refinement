"""Step 4 of the test layer: turn surviving mutants into feedback for the LLM.

This is the core of the project. After a round, some mutants survive. We show
them to the LLM and ask for MORE tests that would catch them. The research
question is how much detail the feedback needs, so there are three formats:

    "score"  only the mutation score            ("score 0.52, 40 of 84 mutants survived")
    "line"   the line that was changed           ("line 23: alpha = 2 / (window + 1)")
    "full"   the full diff of each mutant        (MuTAP-style)

Flow of one feedback round:
    report --describe_survivor/build_feedback_prompt--> prompt --LLM--> new tests
    existing tests + new tests --merge_tests--> one test file for the next round

tests/test_feedback.py describes what each function must do. Fill in the TODOs
in this order: describe_survivor -> build_feedback_prompt -> merge_tests.
"""

from __future__ import annotations

import ast

from testgen.mutation import MutantInfo, MutationReport

FORMATS = ("score", "line", "full")


def describe_survivor(mutant: MutantInfo, fmt: str) -> str:
    """Describe ONE surviving mutant for the prompt.

    - "line": f"Line {line}: `{original line, stripped}`"
    - "full": the mutant's diff, stripped of leading/trailing whitespace
    - "score" (or anything else): ValueError, because that format shows no mutants

    Example for "line":  Line 23: `alpha = 2 / (window + 1)`
    """
    # TODO(Yusuf)
    raise NotImplementedError


def build_feedback_prompt(
    source_code: str,
    module_import: str,
    current_tests: str,
    report: MutationReport,
    fmt: str,
    max_mutants: int = 10,
) -> str:
    """Build the prompt for a feedback round.

    It must contain:
    1. the source code in a ```python block (as in build_prompt),
    2. the current tests in a second ```python block,
    3. the score line, for EVERY format:
           f"Current mutation score: {report.score:.2f} "
           f"({len(report.survived)} of {len(report.mutants)} mutants survived)."
    4. for "line" and "full" only: the first `max_mutants` survivors, one
       describe_survivor(...) per survivor, each on its own line(s),
    5. the instructions: write ADDITIONAL pytest tests that would fail on these
       mutants, do not repeat the existing tests, import from `module_import`,
       one behaviour per test.

    An unknown fmt (not in FORMATS) raises ValueError.
    Tip: build a list of strings (`parts`) and return "\\n".join(parts).
    """
    # TODO(Yusuf)
    raise NotImplementedError


def merge_tests(existing: str, new: str) -> str:
    """Combine the current test file with the new tests from the LLM.

    Rules:
    - everything in `existing` stays, in order;
    - imports from `new` are added only if the same import line is not already there
      (compare with ast.unparse(node));
    - test functions from `new` are added at the end; if a function name already
      exists, rename the new one by adding "_2" (or "_3", ... until the name is free);
    - other top-level statements in `new` (helpers, constants) are added at the end too.
    Return ast.unparse(tree) + "\\n".

    Hints:
    - parse both: old = ast.parse(existing); extra = ast.parse(new)
    - isinstance(node, (ast.Import, ast.ImportFrom)) finds imports
    - put new imports right after the last existing import:
          old.body.insert(position, node)
    - a set of names makes the "is this name taken?" check easy
    """
    # TODO(Yusuf)
    raise NotImplementedError
