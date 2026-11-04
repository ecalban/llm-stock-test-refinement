"""Specification for testgen/feedback.py. Make these pass."""

import pytest

from testgen.feedback import FORMATS, build_feedback_prompt, describe_survivor, merge_tests
from testgen.mutation import MutantInfo, MutationReport

DIFF = """--- calc.py
+++ calc.py
@@ -1,2 +1,2 @@
 def is_adult(age):
-    return age >= 18
+    return age > 18
"""

SURVIVOR = MutantInfo(2, "calc.py", 2, "survived", DIFF, "    return age >= 18")

# --- describe_survivor ---------------------------------------------------------------


def test_describe_line_format():
    assert describe_survivor(SURVIVOR, "line") == "Line 2: `return age >= 18`"


def test_describe_full_format_is_the_diff():
    text = describe_survivor(SURVIVOR, "full")
    assert text == DIFF.strip()
    assert "+    return age > 18" in text


def test_describe_rejects_score_format():
    with pytest.raises(ValueError):
        describe_survivor(SURVIVOR, "score")


# --- build_feedback_prompt -------------------------------------------------------------

SOURCE = "def is_adult(age):\n    return age >= 18\n"
TESTS = "from calc import is_adult\n\ndef test_far():\n    assert is_adult(30)\n"
REPORT = MutationReport(mutants=[
    MutantInfo(1, "calc.py", 2, "killed"),
    SURVIVOR,
    MutantInfo(3, "calc.py", 2, "survived", "--- calc.py\n+    return age >= 19\n", "    return age >= 18"),
    MutantInfo(4, "calc.py", 2, "killed"),
])


def prompt(fmt: str, **kwargs) -> str:
    return build_feedback_prompt(SOURCE, "calc", TESTS, REPORT, fmt, **kwargs)


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_has_code_tests_and_score(fmt):
    text = prompt(fmt)
    assert "```python\n" + SOURCE in text
    assert TESTS in text
    assert "Current mutation score: 0.50 (2 of 4 mutants survived)." in text
    assert "calc" in text
    assert "additional" in text.lower()


def test_score_format_shows_no_mutants():
    text = prompt("score")
    assert "Line 2:" not in text
    assert "+    return age > 18" not in text


def test_line_format_lists_each_survivor():
    assert prompt("line").count("Line 2: `return age >= 18`") == 2


def test_full_format_includes_diffs():
    text = prompt("full")
    assert "+    return age > 18" in text
    assert "+    return age >= 19" in text


def test_max_mutants_limits_the_list():
    text = prompt("full", max_mutants=1)
    assert "+    return age > 18" in text
    assert "+    return age >= 19" not in text


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError):
        prompt("everything")


# --- merge_tests -----------------------------------------------------------------------

EXISTING = """import pytest
from calc import is_adult

def test_far():
    assert is_adult(30)
"""


def test_merge_adds_new_tests_at_the_end():
    new = "from calc import is_adult\n\ndef test_boundary():\n    assert is_adult(18)\n"
    merged = merge_tests(EXISTING, new)
    assert merged.index("def test_far") < merged.index("def test_boundary")
    compile(merged, "<merged>", "exec")


def test_merge_does_not_duplicate_imports():
    new = "import pytest\nfrom calc import is_adult\n\ndef test_boundary():\n    assert is_adult(18)\n"
    merged = merge_tests(EXISTING, new)
    assert merged.count("import pytest") == 1
    assert merged.count("from calc import is_adult") == 1


def test_merge_adds_missing_imports_before_the_tests():
    new = "import math\n\ndef test_nan():\n    assert not is_adult(math.nan)\n"
    merged = merge_tests(EXISTING, new)
    assert "import math" in merged
    assert merged.index("import math") < merged.index("def test_far")


def test_merge_renames_clashing_test_names():
    new = "def test_far():\n    assert is_adult(99)\n\ndef test_far_2():\n    assert is_adult(50)\n"
    merged = merge_tests(EXISTING, new)
    assert "def test_far():" in merged
    assert "def test_far_2():" in merged  # new test_far renamed
    assert "def test_far_2_2():" in merged  # new test_far_2 clashes with the renamed one
    assert "is_adult(30)" in merged and "is_adult(99)" in merged and "is_adult(50)" in merged

def test_merge_puts_every_new_import_before_the_tests():
    existing = EXISTING + "\ndef test_near():\n    assert is_adult(19)\n"
    new = "import math\nimport os\n\ndef test_x():\n    assert True\n"
    merged = merge_tests(existing, new)
    assert merged.index("import os") < merged.index("def test_far")