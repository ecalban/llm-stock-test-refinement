"""Run ``python -m prediction --help`` for the local CSV prediction workflow."""

import argparse
import json
from pathlib import Path

from prediction.data import load_prices
from prediction.pipeline import run_prediction
from prediction.reporting import export_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward BIST long/cash baseline from a daily CSV")
    parser.add_argument("csv", type=Path, help="CSV with date, open, high, low, close, volume, optional adj_close")
    parser.add_argument("--symbol", help="BIST ticker, e.g. THYAO.IS")
    parser.add_argument("--train-size", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--neutral-band", type=float, default=0.003, help="Inclusive hold return band; 0.003 means 0.3%%")
    parser.add_argument("--start", help="First decision date (YYYY-MM-DD); older training history is retained")
    parser.add_argument("--end", help="Last observed date (YYYY-MM-DD); later outcomes are excluded")
    parser.add_argument("--benchmark-csv", type=Path, help="Optional BIST 100 OHLCV CSV covering evaluation dates")
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--commission-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prediction"))
    args = parser.parse_args()
    try:
        data = load_prices(args.csv, symbol=args.symbol)
        benchmark = load_prices(args.benchmark_csv) if args.benchmark_csv is not None else None
        result = run_prediction(
            data, train_size=args.train_size, test_size=args.test_size,
            neutral_band=args.neutral_band, start=args.start, end=args.end,
            benchmark_prices=benchmark,
            initial_cash=args.initial_cash, commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    export_prediction(data, result, args.output_dir)
    metrics = json.dumps(result.backtest.metrics, indent=2, allow_nan=False)
    print(metrics)
    print("Classification:", json.dumps(result.classification_metrics, allow_nan=False))
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
