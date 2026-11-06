# Experiment runs

The Test Center calls `POST /api/experiments`, then polls the run by its ID.
Progress, validation outcomes, exact mutant diffs and test failures are saved
under `EXPERIMENT_RUNS_DIR` (default `data/experiments`). One job runs at a time.
Start the local API with one worker. A restart leaves completed reports available;
interrupted jobs appear as failed and can be started again.

| Source | Target modules | Prerequisite |
| --- | --- | --- |
| Repo reference tests | indicators, signals, data, backtest | Python dependencies |
| OpenAI-compatible LLM | indicators, signals | `LLM_API_KEY`, `LLM_API_MODEL`, optional `LLM_API_BASE_URL` |
| Ollama | indicators, signals | Running Ollama with the model in `OLLAMA_MODEL` |

Set provider credentials in the backend's `.env`, then restart the API.
The browser receives provider names and availability, never the API key. Only
start an external provider job when you intend to send the selected prediction
module, and feedback on that module, to that provider.

LLMs return bounded JSON test cases with a function name, numeric inputs and
expected outputs. The server validates the schema and a fixed function allowlist;
a trusted pytest harness evaluates the cases in a temporary copy of the
prediction package. Arbitrary model-generated Python is not executed.
Only tests that pass on the original code enter the accumulating suite. Later
rounds receive the previous score and, depending on the selected method, a
bounded set of changed lines or full surviving mutant sources. A job stops at
its round budget, when every selected mutant is caught, or when the score stops
improving.

mutmut 2.5.1 generates the variants. The runner deduplicates identical syntax
trees, then selects a deterministic sample across the file (8, 16 or 24 mutants
in the UI). It reports the full candidate pool, the selected count and a
catalog identifier, and uses the same selected variants each round. Pytest
failures on mutated code are counted as caught; timeouts and runner/collection
errors are reported separately and do not increase the score. The score is
caught divided by selected, not an estimate of full-module coverage. Semantic
equivalence is not inferred.

The comparison page lists completed **LLM** runs only and groups by target,
mutant catalog and configured round budget. For a fair experiment, run each
provider and feedback method with those settings held fixed. Repeat runs and
record their model versions before drawing conclusions. Reference runs have
their own status/history and never populate the LLM comparisons.

Example reference run:

```bash
curl -sS http://127.0.0.1:8000/api/experiments \
  -H 'Content-Type: application/json' \
  -d '{"target":"indicators","provider":"reference","mutant_limit":8}'
```

Open the Test Center to follow the run. API consumers can call
`GET /api/experiments/{id}` or list runs with `GET /api/experiments`.
Use `GET /api/experiments/comparisons?format=csv` to download completed LLM rows.
