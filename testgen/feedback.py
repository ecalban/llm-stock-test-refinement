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
    if fmt == "line":
        return f"Line {mutant.line}: `{mutant.source_line.strip()}`"
    if fmt == "full":
        return mutant.diff.strip()
    raise ValueError(fmt)


def build_feedback_prompt(
    source_code: str,
    module_import: str,
    current_tests: str,
    report: MutationReport,
    fmt: str,
    max_mutants: int = 10,
) -> str:
    #Unknown format -> error (FORMATS = ("score", "line", "full"))
    if fmt not in FORMATS:
        raise ValueError(fmt)

    fence = "`" * 3
    parts = []

    # 1. the source code in a python block
    parts.append("Below is a Python module and its current pytest tests.")
    parts.append(f"{fence}python\n{source_code}{fence}")

    # 2. the current tests in a second python block
    parts.append("Current tests:")
    parts.append(f"{fence}python\n{current_tests}{fence}")

    # 3. the score line (every format)
    parts.append(
        f"Current mutation score: {report.score:.2f} "
        f"({len(report.survived)} of {len(report.mutants)} mutants survived)."
    )

    # 4. the survivors, only for "line" and "full"
    if fmt != "score":
        parts.append("These changes to the code were NOT detected by the tests:")
        for mutant in report.survived[:max_mutants]:
            parts.append(describe_survivor(mutant, fmt))

    # 5. the instructions
    parts.append(
        "Write additional pytest tests that would fail on these changes.\n"
        f"Import the functions from `{module_import}`.\n"
        "Do not repeat the existing tests.\n"
        "Every test function checks only one behaviour (1-2 asserts)."
    )

    return "\n".join(parts)



def merge_tests(existing: str, new: str) -> str:
    old = ast.parse(existing)
    extra = ast.parse(new)

    # What is already in the existing file
    seen_imports = set()
    used_names = set()
    insert_at = 0
    for i, node in enumerate(old.body):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seen_imports.add(ast.unparse(node))
            insert_at = i + 1
        elif isinstance(node, ast.FunctionDef):
            used_names.add(node.name)

    for node in extra.body:
        # 1. Imports: add only the ones we do not have yet
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            if text not in seen_imports:
                old.body.insert(insert_at, node)
                insert_at += 1
                seen_imports.add(text)
            continue

        # 2. Test functions: rename on a name clash
        if isinstance(node, ast.FunctionDef):
            name = node.name
            k = 2
            while name in used_names:
                name = f"{node.name}_{k}"
                k += 1
            node.name = name
            used_names.add(name)

        # 3. Everything else (and the renamed function) goes at the end
        old.body.append(node)

    return ast.unparse(old) + "\n"

