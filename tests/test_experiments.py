import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.experiments import get_experiments
from app.config import Settings
from app.main import app
from testgen.experiments import ExperimentService, ROOT, _prompt, mutation_catalog, parse_cases, run_experiment
from testgen.llm_client import LLMResult


def test_llm_cases_are_data_and_have_a_module_specific_allowlist():
    good = {"function": "sma", "values": [1, 3], "args": [2], "expected": [None, 2]}
    data = [
        good,
        {"function": "__import__", "values": ["os"], "expected": None},
        {**good, "args": ["os.system('echo unsafe')"]},
        {**good, "kwargs": {"unexpected": 1}},
        {**good, "expected": {"python": "exec('test')"}},
    ]
    cases, generated = parse_cases("```json\n" + json.dumps(data) + "\n```", "indicators")
    assert generated == 5
    assert [row["function"] for row in cases] == ["sma"]
    assert not parse_cases(json.dumps([good]), "signals")[0]


def test_feedback_only_exposes_the_selected_level():
    rounds = [{"score": .5, "mutants": [
        {"status": "survived", "line": 12, "original": "ORIGINAL", "replacement": "REPLACEMENT",
         "source": "FULL SURVIVING SOURCE"},
        {"status": "killed", "line": 20, "original": "KILLED", "replacement": "KILLED",
         "source": "FULL KILLED SOURCE"},
    ]}]
    score, line, code = [_prompt("baseline", "indicators", rounds, level)
                         for level in ("score", "line", "code")]
    assert all("0.5000" in prompt for prompt in (score, line, code))
    assert "REPLACEMENT" not in score
    assert "REPLACEMENT" in line and "FULL SURVIVING SOURCE" not in line
    assert "FULL SURVIVING SOURCE" in code and "FULL KILLED SOURCE" not in code


def test_llm_cases_run_on_original_and_mutated_code_without_modifying_source():
    class Client:
        def generate(self, prompt, *, system):
            rows = [
                {"function": "sma", "values": [1, 3], "args": [2], "expected": [None, 2]},
                {"function": "sma", "values": [1, 3], "args": [2], "expected": [None, 100]},
                {"function": "rsi", "values": [4, 4, 4], "args": [2], "expected": [None, None, 50]},
            ]
            return LLMResult(json.dumps(rows), "fixture-model", 12, 9, 0.01)

    original = (ROOT / "prediction" / "indicators.py").read_text()
    config = dict(target="indicators", provider="openai", feedback="line", rounds=2, mutant_limit=3)
    report = run_experiment(config, lambda _: None, Client())
    initial = report["rounds"][0]
    assert initial["generated"] == 3
    assert initial["valid"] == initial["suite_size"] == 2
    assert initial["resolved_model"] == "fixture-model"
    assert [test["passed"] for test in initial["validation"]] == [True, False, True]
    assert sum(initial[k] for k in ("killed", "survived", "timeout", "errors")) == 3
    assert all(mutant["diff"] and mutant["line"] > 0 for mutant in initial["mutants"])
    assert all(
        [m["id"] for m in step["mutants"]] == [m["id"] for m in initial["mutants"]]
        for step in report["rounds"]
    )
    assert (ROOT / "prediction" / "indicators.py").read_text() == original


def test_reference_job_is_saved_and_never_counted_as_llm_result(tmp_path):
    service = ExperimentService(Settings(experiment_runs_dir=tmp_path))
    app.dependency_overrides[get_experiments] = lambda: service
    try:
        with TestClient(app) as client:
            response = client.post("/api/experiments", json={
                "provider": "reference", "target": "signals", "rounds": 3, "mutant_limit": 2,
            })
            assert response.status_code == 202
            identifier = response.json()["id"]
            service.pool.shutdown(wait=True)
            report = client.get(f"/api/experiments/{identifier}").json()
            assert report["status"] == "completed", report.get("error")
            assert len(report["rounds"]) == report["config"]["rounds"] == 1
            assert len(report["rounds"][0]["mutants"]) == 2
            history = client.get("/api/experiments").json()["items"]
            assert history[0]["id"] == identifier and "mutants" not in history[0]["rounds"][0]
            assert client.get("/api/experiments/comparisons").json() == {"items": []}
            assert client.post("/api/experiments", json={"target": "../../tmp"}).status_code == 422
            assert client.get("/api/experiments/00000000-0000-4000-8000-000000000000").status_code == 404
            assert (tmp_path / f"{identifier}.json").is_file()
    finally:
        app.dependency_overrides.pop(get_experiments, None)


def test_mutant_catalog_is_deterministic_and_source_spanning():
    path = Path("prediction/indicators.py")
    source = (ROOT / path).read_text()
    first, pool = mutation_catalog(source, path.as_posix(), 8)
    second, _ = mutation_catalog(source, path.as_posix(), 8)
    assert first == second
    assert pool >= len(first) == 8
    assert len({m.id for m in first}) == 8
    assert first[0].line < first[-1].line
