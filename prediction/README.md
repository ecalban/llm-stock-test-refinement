# Prediction layer

A reproducible daily BIST research baseline with **three-class buy/hold/sell
probabilities**, expanding-window prediction, classification evaluation and
long/cash backtests. Local CSVs, the CLI and HTTP API share the same engine.

## Run locally

Requires Python 3.11+ and the root requirements. From the repository root:

```bash
source .venv/bin/activate
python -m scripts.download_prices THYAO ASELS TUPRS BIMAS AKBNK XU100
python -m prediction data/THYAO.csv --symbol THYAO.IS
```

The downloader saves completed daily observations and provider metadata under
ignored `data/`. It excludes today's exchange session, rejects invalid bars
and reports omitted incomplete observations. A network failure preserves the
existing CSV. It uses Yahoo Finance's chart endpoint; provider availability
and historical-data completeness are external dependencies.

To evaluate a specific period with a separate BIST 100 series:

```bash
python -m prediction data/THYAO.csv --symbol THYAO.IS \
  --start 2026-01-01 --end 2026-11-04 \
  --neutral-band 0.003 --benchmark-csv data/XU100.csv \
  --output-dir data/results/THYAO-2026
```

An omitted end uses the last local observation. An explicit end also truncates
outcome availability: later source rows are not used to mark forecasts correct.
The requested start retains earlier source rows for training. A one-month chart
does not imply training on only one month.

Defaults need at least **277 valid bars**: 20 warm-up, 252 usable training,
two purged rows, and three evaluation rows. A shorter training window is useful
for small synthetic fixtures, not selected automatically to improve a result.

## Files produced

| File | Contents |
| --- | --- |
| `signals.csv` | Three class probabilities, selected class/probability, action, fold, actual class, correctness and outcome dates |
| `folds.csv` | Training/test boundaries and label availability |
| `backtest.csv` | Positions, interval returns, costs, equity, buy-hold, drawdown and valuation dates |
| `trades.csv` | Closed long positions, quantity, entry/exit, fees, P&L and duration |
| `orders.csv` | Actual entry/exit orders, capital and separate commission/slippage |
| `metrics.json` | Financial metrics, order/position counts, win rate and recovery |
| `classification.json` | Accuracy, macro F1, confusion matrix, class metrics and pending count |
| `config.json` | Period, class band, timing, training and fee assumptions |
| `report.json` | JSON payload with real price/indicator series, predictions, ledgers and metadata |

Missing outcomes and unavailable metrics are JSON null, never fabricated zeroes.
All probabilities, returns and costs-as-ratios are fractions: 0.10 means 10%.
The UI formats dates, currency and percentages.

## Model and class meaning

The default `SignalModel` is training-only StandardScaler + LogisticRegression.
It predicts a fixed `buy, hold, sell` distribution. An absent training class gets
zero probability; a single-class fold gets a one-hot distribution. The largest
probability selects the action; exact ties favor hold, then buy, then sell.
Outputs are not separately calibrated probabilities of investment success.

For a decision at close[t], the realized return is:

```text
adjusted_open[t+2] / adjusted_open[t+1] - 1
```

With `neutral_band=0.003`:

- return > +0.3%: buy;
- return < -0.3%: sell;
- inclusive interval [-0.3%, +0.3%]: hold.

This is a configurable starting label definition, not a threshold optimized on
the test data. It is recorded with each run and kept fixed across folds.
The final two labels remain unknown, including in a historical period ending
before the last available source row. Correctness is null until the outcome
is observable.

Legacy binary `DirectionModel`, `build_target` and `generate_signals` remain
available for existing experiments. The default CLI/pipeline now uses genuine
three-class probabilities. The previous buy/sell probability threshold flags
are replaced by `--neutral-band`. Binary and three-class experiment results
are different model versions and must not be conflated.

## Indicators, features and validation

CSV schema:

```csv
date,open,high,low,close,adj_close,volume
2024-01-02,100,103,99,102,51,1000000
2024-01-03,102,104,101,103,51.5,1100000
```

These two rows illustrate the schema, not a sufficient training set.
Headers are case insensitive; `Adj Close` normalizes to `adj_close`.
An optional symbol column must contain one normalized BIST ticker.
Missing dates/prices, duplicate sessions, non-finite/nonpositive prices,
negative/fractional volume and impossible OHLC bars are errors. Zero volume is
permitted. Rows are sorted; missing prices are never filled.

Adjusted close defaults to close for already-adjusted input. If unadjusted OHLC
is supplied, provide adjusted closes on the corresponding basis. Opening prices
are adjusted as `open * adj_close / close`. Historical adjusted series may
incorporate retrospective corporate-action revisions; no point-in-time archive
is provided.

Indicators preserve the pandas index:

- SMA uses a full-window warm-up; EMA starts from the first price.
- RSI uses Wilder smoothing seeded by the first window's average changes.
  Flat/gain-only/loss-only values are 50/100/0.
- MACD defaults to 12/26/9; Bollinger bands to 20 and two population standard deviations.
- Nine features: 1/5-session returns, 5/20 SMA ratio, 12/26 EMA ratio, RSI,
  normalized MACD histogram, band position, 20-session volatility and relative volume.
- Flat bands and zero volume have finite neutral features; warm-up is never backfilled.

## Walk-forward and classification evaluation

Train on expanding history, exclude the two feature rows immediately before
each test fold, fit a fresh scaler/model, then predict the next `test_size`
sessions with that fixed model. All training outcomes are known strictly before
the first decision in the fold. No shuffling, future scaling or retroactive
retraining of already-issued forecasts.

Metrics use observable out-of-sample labels only. Confusion-matrix rows are
actual classes and columns predicted classes, ordered buy/hold/sell. Macro F1
averages all three classes, including classes absent in a particular evaluation.
`majority_accuracy` is the largest class's fraction in the evaluation labels;
it describes class imbalance, not a separately trained deployable strategy.

## Trading and ledgers

Trading remains **long or cash**, never short:

- buy targets 100% long; sell closes the position; hold preserves exposure.
- Signals execute one session later at adjusted open. No leverage or repeated
  purchases from repeated buy instructions.
- Commission and slippage are per-side basis points, default 10 + 5 bps.
- Interval net return:
  `(1 - turnover * cost_rate) * (1 + position * asset_return) - 1`.
- Exit at the final observed open, including its cost. Do not open a position
  there when no subsequent opening return is observable.
- Order quantities are fractional shares; the trade ledger reconciles with
  portfolio equity. Return is compounded after entry and exit costs.
- The final two forecasts remain useful for later decisions but cannot initiate
  a fully observable open-to-open trade within this truncated dataset.

`trades` in the legacy metrics remains an order-count alias. New unambiguous
fields are `order_count` and `closed_trade_count`. `win_rate` is the fraction
of closed positions with positive net return, null without closed positions.
`total_cost` is in capital's currency, not a percentage.

`recovery_sessions` counts sessions from the worst drawdown trough to regaining
its preceding peak; null means not recovered and zero means no drawdown.
`max_underwater_sessions` includes an ongoing underwater episode.
Annualization assumes 252 sessions and Sharpe uses zero risk-free return.
Risk calculations merge final liquidation cost into the final holding interval.

The daily row at t includes the return through open[t+1]. Use
`valuation_date` to plot equity, and decision dates to plot signals.
The terminal row contains liquidation at the same final open. Aggregate duplicate
valuation dates using the last equity for an end-of-session chart.

Costs, calendar sessions and benchmark dates must be comparable. The stock's
buy-hold curve enters at the strategy's earliest executable open and exits at
the final open. Optional BIST 100 has its own prices/curve/return and the same
entry/exit cost assumption; it is not relabeled stock buy-hold.

Missing benchmark dates are rejected explicitly. In the downloaded Yahoo series,
some historical XU100 holiday observations were absent although stock bars
existed. Supply a complete source or choose a fully covered period; no forward
fill or silently shortened strategy sample is used. Without a requested index,
`bist100_return` is null.

Assumptions: fractional shares, immediate fills, no tax, lot restrictions,
cash interest, price-limit or liquidity model. A working backtest does not
establish a profitable strategy.

## Strategies

`prediction.pipeline.run_strategy_backtest` supports:

- `buy_hold`: one entry and terminal exit;
- `sma`: target long when fast SMA > slow SMA, otherwise cash; defaults 5/20.

SMA retains pre-period indicator history and excludes unavailable warm-up rows.
The ML strategy uses `run_prediction`. All strategies share the same execution,
cost and ledger engine. Compare strategies with the same explicit start/end;
their natural earliest usable dates can differ.

## HTTP API

```bash
python -m uvicorn app.main:app --reload
```

Swagger UI: http://localhost:8000/docs. Prediction requests use local CSVs and
file-backed reports; they do not query PostgreSQL or call LLMs. The separate
database health endpoint still checks PostgreSQL.

```bash
curl -X POST http://localhost:8000/api/analyses \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"THYAO","start":"2026-01-01","benchmark":true}'
```

| Method and path | Purpose |
| --- | --- |
| GET /api/capabilities | Available model/strategies and timing semantics |
| GET /api/stocks | Actual local data catalog and freshness |
| GET /api/stocks/{symbol}/prices | Daily bars with optional start/end |
| POST /api/analyses | Three-class prediction, indicators, evaluation and ML backtest |
| GET /api/analyses/{id} | Restore a saved analysis |
| GET /api/analyses/{id}/signals | Paginated history; `?format=csv` downloads all rows |
| POST /api/backtests | Strategy `ml`, `buy_hold` or `sma`, explicit period/costs |
| GET /api/backtests/{id} | Restore saved backtest |
| GET /api/backtests/{id}/trades | Paginated closed positions; `?format=csv` downloads all |
| GET /api/dashboard | Latest saved analysis per stock and pooled classification accuracy |

POSTs are synchronous and return a completed persisted report with HTTP 201.
Reports are atomically written under `PREDICTION_RUNS_DIR=data/runs` and survive
process restart. Data files come from `MARKET_DATA_DIR=data`. UUIDs identify
reports; clients cannot request arbitrary filesystem paths. Unknown symbols or
report IDs return 404; unsupported models, incomplete periods and invalid
settings return 422. All request bodies reject unknown fields.

Only logistic regression is implemented. Capabilities mark LightGBM/LSTM as
unavailable, so a frontend should disable them. LLM experiment APIs are not
implemented in this prediction change; dashboard mutation/test metrics are null.

The report includes data hashes, prediction-source hash, runtime versions,
effective evaluation dates and recorded label/trading settings. The dashboard
uses each stock's latest saved analysis; pooled accuracy is weighted by
observable label count. It is not an average of hand-entered demo figures.

## Python usage and tests

```python
from prediction.data import load_prices
from prediction.pipeline import run_prediction, run_strategy_backtest
from prediction.reporting import analysis_report

prices = load_prices("data/THYAO.csv", symbol="THYAO")
result = run_prediction(prices, start="2026-01-01")
print(result.classification_metrics)
print(result.backtest.trades)
payload = analysis_report(prices, result)

sma_result = run_strategy_backtest(prices, strategy="sma", start="2026-01-01")
```

```bash
pytest tests/prediction -q
pytest -q
```

Tests cover exact indicator/fee math, missing classes, future perturbation,
unknown outcomes, period boundaries, ledger reconciliation, independent index
benchmarks, recovery/null behavior, API persistence, actual CSV downloads,
invalid requests and an offline mocked data downloader. Production market-data
calls are not required by the test suite.
