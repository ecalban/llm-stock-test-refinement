"""Trusted pytest driver for declarative LLM cases. Copied only into run workspaces."""

import json
from pathlib import Path

import pandas as pd
import pytest

from prediction import indicators, signals

FUNCTIONS = {
    name: getattr(module, name)
    for module, names in ((indicators, ("sma", "ema", "rsi", "macd", "bollinger_bands")),
                          (signals, ("generate_signals", "target_positions")))
    for name in names
}
CASES = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))


def assert_value(actual, expected):
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for item, wanted in zip(actual, expected, strict=True):
            assert_value(item, wanted)
    elif type(expected) in (int, float):
        assert actual == pytest.approx(expected, rel=1e-7, abs=1e-9)
    else:
        assert actual == expected


@pytest.mark.parametrize("case", CASES, ids=[f"case-{i}" for i in range(len(CASES))])
def test_generated_case(case):
    function = FUNCTIONS[case["function"]]
    args = [pd.Series(case["values"]), *case["args"]]
    if case["raises"]:
        with pytest.raises(ValueError):
            function(*args, **case["kwargs"])
        return
    result = function(*args, **case["kwargs"])
    actual = json.loads(result.to_json(orient="values", double_precision=15))
    assert_value(actual, case["expected"])
