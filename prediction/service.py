"""Local-data application service; persisted reports survive process restarts."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

from prediction.data import load_prices, normalize_symbol
from prediction.pipeline import run_prediction, run_strategy_backtest
from prediction.reporting import analysis_report, backtest_report

STOCK_NAMES = {"THYAO": "Türk Hava Yolları", "ASELS": "Aselsan", "TUPRS": "Tüpraş",
               "BIMAS": "BİM Mağazalar", "AKBNK": "Akbank", "XU100": "BIST 100"}


class PredictionService:
    def __init__(self, data_dir: Path, runs_dir: Path):
        self.data_dir, self.runs_dir = Path(data_dir), Path(runs_dir)

    def _source(self, symbol):
        ticker = normalize_symbol(symbol)
        for name in (ticker[:-3], ticker):
            path = self.data_dir / f"{name}.csv"
            if path.is_file():
                return path, ticker
        raise FileNotFoundError(f"No local CSV for {ticker}; download or import it first")

    def prices(self, symbol):
        path, ticker = self._source(symbol)
        return load_prices(path, symbol=ticker)

    def stocks(self):
        result, seen = [], set()
        for path in sorted(self.data_dir.glob("*.csv")):
            try:
                ticker = normalize_symbol(path.stem)
                if ticker in seen:
                    continue
                frame = load_prices(path, symbol=ticker)
            except ValueError:
                continue
            seen.add(ticker)
            result.append({"symbol": ticker, "name": STOCK_NAMES.get(ticker[:-3], ticker[:-3]),
                           "kind": "index" if ticker == "XU100.IS" else "stock",
                           "rows": len(frame), "first_date": frame.index[0].date().isoformat(),
                           "as_of": frame.index[-1].date().isoformat(),
                           "close": float(frame.close.iloc[-1])})
        return result

    def _provenance(self, symbol):
        path, ticker = self._source(symbol)
        return {"symbol": ticker, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _benchmark(self, enabled):
        return self.prices("XU100") if enabled else None

    def _save(self, kind, report):
        identifier = str(uuid4())
        report.update({"id": identifier, "kind": kind, "status": "completed"})
        payload = json.dumps(report, ensure_ascii=False, allow_nan=False)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        destination = self.runs_dir / f"{identifier}.json"
        temporary = destination.with_suffix(".tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return report

    def read(self, identifier, *, kind=None):
        try:
            canonical = str(UUID(identifier))
        except (ValueError, TypeError, AttributeError):
            raise FileNotFoundError("Unknown report") from None
        report = json.loads((self.runs_dir / f"{canonical}.json").read_text(encoding="utf-8"))
        if kind is not None and report["kind"] != kind:
            raise FileNotFoundError("Report has a different kind")
        return report

    def analyse(self, symbol, *, benchmark=False, **options):
        prices = self.prices(symbol)
        result = run_prediction(prices, benchmark_prices=self._benchmark(benchmark), **options)
        report = analysis_report(prices, result)
        report["sources"] = [self._provenance(symbol)]
        if benchmark:
            report["sources"].append(self._provenance("XU100"))
        return self._save("analysis", report)

    def backtest(self, symbol, *, strategy="ml", benchmark=False, **options):
        if strategy == "ml":
            # Persist a single backtest report; do not create a second hidden analysis run.
            prices = self.prices(symbol)
            result = run_prediction(prices, benchmark_prices=self._benchmark(benchmark), **options)
            report = backtest_report(result.backtest, result.config)
            report["classification_metrics"] = result.classification_metrics
        else:
            prices = self.prices(symbol)
            options = {k: v for k, v in options.items() if k not in ("train_size", "test_size", "neutral_band")}
            result = run_strategy_backtest(prices, strategy=strategy,
                                            benchmark_prices=self._benchmark(benchmark), **options)
            config = {k: str(v) if k in ("start", "end") and v is not None else v
                      for k, v in options.items()}
            config.update({"strategy": strategy, "position_mode": "long_cash", "price_basis": "adjusted",
                           "evaluation_start": result.daily.index[0].date().isoformat(),
                           "evaluation_end": result.daily.index[-1].date().isoformat()})
            report = backtest_report(result, config)
        report.update({"schema_version": 2, "symbol": normalize_symbol(symbol),
                       "computed_at": datetime.now(timezone.utc).isoformat(),
                       "sources": [self._provenance(symbol)]})
        if benchmark:
            report["sources"].append(self._provenance("XU100"))
        return self._save("backtest", report)

    def dashboard(self):
        stocks = [s for s in self.stocks() if s["kind"] == "stock"]
        latest = {}
        for path in self.runs_dir.glob("*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("kind") != "analysis":
                continue
            key = report["symbol"]
            if key not in latest or report["computed_at"] > latest[key]["computed_at"]:
                latest[key] = report
        signals = [{"symbol": symbol, "as_of": value["as_of"], "quote": value["quote"],
                    "analysis_id": value["id"],
                    "prediction": value["latest_prediction"]}
                   for symbol, value in sorted(latest.items())]
        metrics = [value["classification_metrics"] for value in latest.values()]
        samples = sum(m["samples"] for m in metrics)
        accuracy = sum(m["accuracy"] * m["samples"] for m in metrics if m["samples"]) / samples if samples else None
        return {"available_stock_count": len(stocks), "analysed_stock_count": len(latest),
                "pooled_accuracy": accuracy, "accuracy_samples": samples,
                "latest_signals": signals, "mutation_score": None, "last_test_status": None}
