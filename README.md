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
```

## Getting started

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

### Tests

```bash
pytest
```

The suite uses in-memory SQLite, so it does not need PostgreSQL.

## Workflow

- `main` is protected by habit: work on a branch (`feature/...`, `fix/...`) and open a pull request.
- The other person reviews before merge; CI must be green.
- Never commit `.env` or downloaded market data (`data/` is ignored).
