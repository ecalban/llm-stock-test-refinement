# llm-stock-test-refinement

Graduation project (İnönü University, Computer Engineering).

A two-layer system:

1. **Prediction layer** – produces buy / sell / hold signals for Borsa Istanbul (BIST) stocks from daily prices and technical indicators, and evaluates them with a walk-forward backtest that includes transaction costs.
2. **Test layer** – asks an LLM to write unit tests for the prediction layer's own code, measures their quality with mutation testing (mutmut), and feeds the surviving mutants back to the LLM until the mutation score stops improving. It compares three feedback formats (score only / changed line / full mutant code) and closed vs. open-source models.

## Project layout

```
app/          FastAPI service, settings, database schema
prediction/   data, indicators, model, signals, backtest   (owner: Eren)
testgen/      LLM client, test runner, mutation loop        (owner: Yusuf)
scripts/      one-off commands (e.g. init_db)
tests/        pytest suite
frontend/     BIST AI Lab React interface (from the Figma Make export)
```

## Getting started

### Local BIST AI Lab interface

Start the API from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.download_prices THYAO ASELS TUPRS BIMAS AKBNK XU100
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal from the repository root (Node.js 20.19+ or 22.12+, pnpm):

```bash
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend dev --port 5173
```

Open [BIST AI Lab](http://127.0.0.1:5173/) or the [API docs](http://127.0.0.1:8000/docs).
Vite proxies `/api` to port 8000. Set `BIST_API_ORIGIN` before starting Vite if
the API uses a different address. Prediction and experiment views use local
files and do not require PostgreSQL. They require downloaded CSVs for market
analysis; skip the download command when these files are already present.

The dashboard, price and indicator charts, probability cards, strategy curves
and CSV downloads use real API responses. With **Repo referans testleri**, Test
Center runs the existing tests against actual mutmut variants without an LLM.
See [experiment details](testgen/README.md) for model setup, feedback and scoring.

### With Docker

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec api python -m scripts.init_db
curl localhost:8000/health/db   # {"status":"ok"}
```

API docs: http://localhost:8000/docs

### Without Docker

Requires Python 3.11+ and a running PostgreSQL.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # set DATABASE_URL
python -m scripts.init_db
uvicorn app.main:app --reload
```

### LLM providers

Set `LLM_API_KEY` (and optionally `LLM_API_MODEL`) in `.env` for the closed model,
and run Ollama locally for open models (`ollama pull llama3.1:8b`). Quick check:

```bash
python -m scripts.llm_smoke openai
python -m scripts.llm_smoke ollama
```

### Tests

```bash
pytest
```

The suite uses in-memory SQLite, so it does not need PostgreSQL.

```bash
pnpm --dir frontend exec tsc --noEmit
pnpm --dir frontend build
```

### Prediction layer

Run a three-class buy/hold/sell baseline and next-open, long/cash backtest
from a local daily OHLCV CSV (no database or LLM required):

```bash
python -m prediction data/THYAO.csv --symbol THYAO.IS \
  --train-size 252 --test-size 21 --output-dir data/results/THYAO
```

Download completed daily data first if needed:

```bash
python -m scripts.download_prices THYAO ASELS TUPRS BIMAS AKBNK XU100
```

This exports class probabilities, accuracy/F1, fold boundaries, daily equity,
closed positions, orders and a complete UI JSON report. The FastAPI service also
exposes prediction endpoints under `/api` (see `/docs`); its reports persist under
`data/runs`. See
[prediction/README.md](prediction/README.md) for the CSV schema, indicator
conventions, execution timing, leakage controls and API usage.

## Workflow

- `main` is protected by habit: work on a branch (`feature/...`, `fix/...`) and open a pull request.
- The other person reviews before merge; CI must be green.
- Never commit `.env` or downloaded market data (`data/` is ignored).
