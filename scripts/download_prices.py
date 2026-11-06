"""Download completed Yahoo daily OHLCV bars: python -m scripts.download_prices --help."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo

import pandas as pd

from prediction.data import normalize_symbol, validate_prices


def download_prices(symbol: str, *, period: str = "5y", output_dir: Path = Path("data")) -> dict:
    """Download a single stock/index, validate, then atomically replace its CSV.

    One CSV per normalized ticker. The current exchange day is excluded even
    after close, avoiding partially formed daily candles. No price filling.
    Network failures leave existing data untouched. Market data is not committed.
    """
    ticker = normalize_symbol(symbol)
    if period not in ("1y", "2y", "5y", "10y", "max"):
        raise ValueError("period must be 1y, 2y, 5y, 10y or max")
    query = urllib.parse.urlencode({"range": period, "interval": "1d"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        chart = json.load(response)["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"No Yahoo data for {ticker}: {chart.get('error')}")
    result = chart["result"][0]
    if result["meta"]["symbol"].upper() != ticker:
        raise ValueError("market-data symbol mismatch")
    tz = ZoneInfo(result["meta"]["exchangeTimezoneName"])
    today = datetime.now(tz).date()
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"].get("adjclose")
    if not adjusted:
        raise ValueError("provider did not return adjusted closes")
    frame = pd.DataFrame({
        "date": [datetime.fromtimestamp(ts, tz).date() for ts in timestamps],
        **{key: quote[key] for key in ("open", "high", "low", "close", "volume")},
        "adj_close": adjusted[0]["adjclose"], "symbol": ticker,
    })
    frame = frame.loc[frame.date < today]
    incomplete = int(frame.isna().any(axis=1).sum())
    data = validate_prices(frame.dropna())
    metadata = {"symbol": ticker, "currency": result["meta"].get("currency"),
                "source_url": url, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(data), "first_date": str(data.index[0].date()),
                "last_date": str(data.index[-1].date()), "excluded_incomplete_rows": incomplete}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{ticker[:-3]}.csv"
    temporary = destination.with_suffix(".csv.tmp")
    try:
        data.to_csv(temporary, index_label="date")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    (output_dir / f"{ticker[:-3]}.source.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Download completed daily BIST prices from Yahoo Finance")
    parser.add_argument("symbols", nargs="+", help="e.g. THYAO ASELS TUPRS BIMAS AKBNK XU100")
    parser.add_argument("--period", choices=["1y", "2y", "5y", "10y", "max"], default="5y")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        for symbol in args.symbols:
            metadata = download_prices(symbol, period=args.period, output_dir=args.output_dir)
            print(f"{metadata['symbol']}: {metadata['rows']} rows, through {metadata['last_date']}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
