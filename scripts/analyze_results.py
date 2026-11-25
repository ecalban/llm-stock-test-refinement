"""Turn results/experiments.csv into the tables and the chart for the report.

Usage (from the repository root):
    python -m scripts.analyze_results                      # all targets
    python -m scripts.analyze_results --target prediction/signals.py

Writes:
    results/summary.csv      one row per (target, fmt): mean start/final score, gain, cost
    results/curves_<module>.png   mean mutation score per round, one line per format

Only shared-start runs are used (run_id "<start>-<fmt>"); the first pilot rows
(before 24 Nov) and runs that broke before round 0 are left out.

tests/test_analyze_results.py describes the helpers. Fill in the TODOs in this
order: clean -> per_run -> summary -> curves. load() and main() are given.
pandas practice: groupby, agg, pivot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    """Given."""
    return pd.read_csv(path, keep_default_na=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only usable rows:
    - run_id contains "-" (shared-start runs; the old pilot ids have no "-")
    - round >= 0 (round -1 means the run broke before round 0)
    Return a copy with a new column `start_id` = the part of run_id before "-".

    Hints: df[df["run_id"].str.contains("-")]; df["run_id"].str.split("-").str[0]
    """
    # TODO(Yusuf)
    raise NotImplementedError


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run_id with columns:
    target, fmt, start_id, start_score (round 0), final_score (last round),
    gain (final - start), rounds (number of FEEDBACK rounds = last round number),
    tokens (sum of prompt_tokens + completion_tokens over the run's rounds,
            round 0 counted only for fmt "none" so the shared start is not paid 4 times)

    Hint: sort by round, then groupby("run_id") and take .iloc[0] / .iloc[-1] per group,
    or use .agg(start_score=("score", "first"), final_score=("score", "last"), ...).
    """
    # TODO(Yusuf)
    raise NotImplementedError


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, fmt), ordered none, score, line, full, with columns:
    n (number of runs), start_mean, final_mean, gain_mean, gain_std, rounds_mean, tokens_mean.
    Numbers rounded to 3 decimals; gain_std is 0.0 when there is only one run.

    Hint: runs.groupby(["target", "fmt"]).agg(n=("final_score", "size"), ...)
    std of a single value is NaN in pandas: .fillna(0.0).
    """
    # TODO(Yusuf)
    raise NotImplementedError


def curves(df: pd.DataFrame, max_round: int) -> pd.DataFrame:
    """Mean score per round for each fmt: rows = round 0..max_round, columns = fmt.

    A run that stopped early keeps its last score for the remaining rounds
    (it did not get worse, it just stopped asking). So per run_id: reindex its
    scores to 0..max_round and forward-fill, then average per fmt.

    Hint: df.pivot(index="round", columns="run_id", values="score")
          .reindex(range(max_round + 1)).ffill()
          then group the columns by their fmt and take the mean.
    """
    # TODO(Yusuf)
    raise NotImplementedError


ORDER = ["none", "score", "line", "full"]


def main(argv: list[str] | None = None) -> None:
    """Given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, default=Path("results/experiments.csv"))
    parser.add_argument("--target", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    df = clean(load(args.csv))
    if args.target:
        df = df[df["target"] == args.target]
    if df.empty:
        sys.exit("No shared-start rows to analyse yet. Run scripts.run_experiment first.")

    table = summary(per_run(df))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "summary.csv", index=False)
    print(table.to_string(index=False))

    import matplotlib

    matplotlib.use("Agg")  # no window, just the file
    import matplotlib.pyplot as plt

    for target, part in df.groupby("target"):
        lines = curves(part, int(part["round"].max()))
        fig, ax = plt.subplots(figsize=(6, 4))
        for fmt in [f for f in ORDER if f in lines.columns]:
            ax.plot(lines.index, lines[fmt], marker="o", label=fmt)
        ax.set(title=f"{target}: mean mutation score per round",
               xlabel="round (0 = no feedback)", ylabel="mutation score", ylim=(0, 1))
        ax.set_xticks(lines.index)
        ax.grid(alpha=0.3)
        ax.legend(title="feedback")
        name = Path(target).stem
        fig.tight_layout()
        fig.savefig(args.out_dir / f"curves_{name}.png", dpi=150)
        print(f"chart: {args.out_dir / f'curves_{name}.png'}")


if __name__ == "__main__":
    main(sys.argv[1:])
