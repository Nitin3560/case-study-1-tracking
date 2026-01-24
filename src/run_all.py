# src/run_all.py

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    return ap.parse_args()


def aggregate(controller: str, seeds: List[int]) -> pd.DataFrame:
    """
    Load per-seed CSVs for a controller and concatenate them into one dataframe.
    Adds a 'seed' column for CI computations.
    """
    dfs = []
    for s in seeds:
        p = Path("outputs/csv") / f"{controller}_seed{s}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing per-run CSV: {p} (did run_one fail?)")
        df = pd.read_csv(p)
        df["seed"] = int(s)
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No CSVs found for controller={controller}")

    return pd.concat(dfs, axis=0, ignore_index=True)


def mean_ci_over_seeds(df: pd.DataFrame, col: str) -> Tuple[pd.Series, pd.Series]:
    """
    95% CI over seeds at each time t.
    NOTE: We compute statistics over all rows grouped by time. Since each seed produces
    one row per control tick, groupby('t') works.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)}")

    g = df.groupby("t")[col]
    mean = g.mean()
    std = g.std(ddof=1)
    n = g.count()
    ci = 1.96 * (std / np.sqrt(n))
    # std can be NaN if n==1 at any t; replace with 0 CI in that case
    ci = ci.fillna(0.0)
    return mean, ci


def plot_curve(df: pd.DataFrame, col: str, title: str, outpath: Path):
    mean, ci = mean_ci_over_seeds(df, col)
    t = mean.index.values

    plt.figure()
    plt.plot(t, mean.values)
    plt.fill_between(t, (mean - ci).values, (mean + ci).values, alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel(col)
    plt.title(title)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def summarize(df: pd.DataFrame, controller_name: str) -> Dict[str, Any]:
    """
    Summarize one controller across all seeds and all timesteps.
    Robust to missing optional columns.
    """
    def _mean(col: str) -> float:
        return float(df[col].mean()) if col in df.columns else float("nan")

    def _max(col: str) -> float:
        return float(df[col].max()) if col in df.columns else float("nan")

    return {
        "controller": controller_name,
        "mean_err_m": _mean("mean_err_m"),
        "max_err_m": _max("max_err_m"),
        "com_err_m": _mean("com_err_m"),                 # KEY metric for trajectory tracking
        "formation_err_m": _mean("formation_err_m"),
        "connectivity_rate": _mean("connectivity_rate"),
    }


def main():
    args = parse_args()
    seeds = args.seeds

    # Ensure output dirs exist
    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    Path("outputs/figs").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)

    # ---- Run all controllers x seeds ----
    for ctrl in ["openloop", "pid", "agentic"]:
        for s in seeds:
            subprocess.check_call(
                [
                    "python",
                    "-m",
                    "src.run_one",
                    "--config",
                    args.config,
                    "--controller",
                    ctrl,
                    "--seed",
                    str(s),
                ]
            )

    # ---- Aggregate ----
    open_df = aggregate("openloop", seeds)
    pid_df = aggregate("pid", seeds)
    ag_df = aggregate("agentic", seeds)

    # ---- Figures ----
    figs = Path("outputs/figs")

    # Track error curves (existing metric)
    plot_curve(open_df, "mean_err_m", "Open-loop mean tracking error (mean ± 95% CI)", figs / "openloop_mean_err.png")
    plot_curve(pid_df,  "mean_err_m", "PID mean tracking error (mean ± 95% CI)",        figs / "pid_mean_err.png")
    plot_curve(ag_df,   "mean_err_m", "Agentic mean tracking error (mean ± 95% CI)",    figs / "agentic_mean_err.png")

    # Centroid tracking error curves (magazine-aligned). Only plot if column exists.
    if "com_err_m" in open_df.columns and "com_err_m" in pid_df.columns and "com_err_m" in ag_df.columns:
        plot_curve(open_df, "com_err_m", "Open-loop COM tracking error (mean ± 95% CI)", figs / "openloop_com_err.png")
        plot_curve(pid_df,  "com_err_m", "PID COM tracking error (mean ± 95% CI)",       figs / "pid_com_err.png")
        plot_curve(ag_df,   "com_err_m", "Agentic COM tracking error (mean ± 95% CI)",   figs / "agentic_com_err.png")

    # ---- Summary CSV (THIS WAS MISSING IN YOUR FILE) ----
    summary_rows = [
        summarize(open_df, "openloop"),
        summarize(pid_df, "pid"),
        summarize(ag_df, "agentic"),
    ]
    summary = pd.DataFrame(summary_rows)

    summary_path = Path("outputs") / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {summary_path.resolve()}")
    print(f"Saved figures in: {figs.resolve()}")


if __name__ == "__main__":
    main()
