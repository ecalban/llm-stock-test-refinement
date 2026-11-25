from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load(path: Path) -> pd.DataFrame:
    """Given."""
    return pd.read_csv(path, keep_default_na=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[(df["run_id"].str.contains("-")) & (df["round"] >= 0)].copy()

    # 1) without expand: each row is a list, .str[0] takes its first element
    out["start_id"] = out["run_id"].str.split("-").str[0]

    # 2) with expand: a table, take its column 0
    out["start_id"] = out["run_id"].str.split("-", expand=True)[0]
    return out


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    # 1. rows of each run in round order, so "first" = round 0 and "last" = final round
    df = df.sort_values(["run_id", "round"]).copy()

    # 2. token cost per row; the shared round 0 is paid only by "none"
    df["cost"] = df["prompt_tokens"] + df["completion_tokens"]
    df.loc[(df["round"] == 0) & (df["fmt"] != "none"), "cost"] = 0

    # 3. one row per run
    runs = df.groupby("run_id").agg(
        target=("target", "first"),
        fmt=("fmt", "first"),
        start_id=("start_id", "first"),
        start_score=("score", "first"),
        final_score=("score", "last"),
        rounds=("round", "max"),
        tokens=("cost", "sum"),
    ).reset_index()

    # 4. gain = final - start
    runs["gain"] = runs["final_score"] - runs["start_score"]
    return runs


def summary(runs: pd.DataFrame) -> pd.DataFrame:

    table = runs.groupby(["target", "fmt"]).agg(
        n = ("final_score", "size"),
        start_mean = ("start_score", "mean"),
        final_mean = ("final_score", "mean"),
        gain_mean = ("gain", "mean"),
        gain_std = ("gain", "std"),
        rounds_mean = ("rounds", "mean"),
        tokens_mean = ("tokens", "mean"),
    ).reset_index()

    table["gain_std"] = table["gain_std"].fillna(0.0)

    table["order"] = table["fmt"].map({fmt: i for i, fmt in enumerate(ORDER)})
    table = table.sort_values(["target", "order"]).drop(columns="order")

    return table.round(3).reset_index(drop=True)


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
