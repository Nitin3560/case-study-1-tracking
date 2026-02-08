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

    Assumption (true for this project):
      Each seed produces one row per control tick, so grouping by 't' yields
      one sample per seed per time.
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


def plot_curve(
    df: pd.DataFrame,
    col: str,
    title: str,
    outpath: Path,
    *,
    ylabel: str | None = None,
):
    mean, ci = mean_ci_over_seeds(df, col)
    t = mean.index.values

    plt.figure()
    plt.plot(t, mean.values)
    plt.fill_between(t, (mean - ci).values, (mean + ci).values, alpha=0.2)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel if ylabel is not None else col)
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

    # In this project:
    #   mean_err_m / max_err_m should be NOMINAL mission-reference error (after run_one fix).
    mean_nom = _mean("mean_err_m")
    max_nom = _max("max_err_m")

    # Commanded-reference error (only differs for agentic; optional)
    mean_cmd = _mean("mean_err_cmd_m")
    max_cmd = _max("max_err_cmd_m")

    out = {
        "controller": controller_name,

        # Explicit, paper-safe names
        "mean_err_nominal_m": mean_nom,
        "max_err_nominal_m": max_nom,

        "mean_err_cmd_m": mean_cmd,
        "max_err_cmd_m": max_cmd,

        "formation_err_rel": _mean("formation_err_rel"),
        "connectivity_rate": _mean("connectivity_rate"),

        # Agentic interpretability channels (optional)
        "agentic_active_rate": _mean("agentic_active"),
        "agentic_ref_shift_mean": _mean("agentic_ref_shift"),
    }

    # Backward-compatible legacy columns (do not remove unless you update downstream)
    out["mean_err_m"] = mean_nom
    out["max_err_m"] = max_nom

    return out


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

    figs = Path("outputs/figs")

    # ------------------------------------------------------------------
    # Figures: NOMINAL mission-reference tracking error (primary metric)
    # ------------------------------------------------------------------
    plot_curve(
        open_df,
        "mean_err_m",
        "Open-loop mean NOMINAL tracking error (mean ± 95% CI)",
        figs / "openloop_mean_err_nominal.png",
        ylabel="mean_err_nominal_m",
    )
    plot_curve(
        pid_df,
        "mean_err_m",
        "PID mean NOMINAL tracking error (mean ± 95% CI)",
        figs / "pid_mean_err_nominal.png",
        ylabel="mean_err_nominal_m",
    )
    plot_curve(
        ag_df,
        "mean_err_m",
        "Agentic mean NOMINAL tracking error (mean ± 95% CI)",
        figs / "agentic_mean_err_nominal.png",
        ylabel="mean_err_nominal_m",
    )

    # ------------------------------------------------------------------
    # Figures: COMMANDED-reference tracking error (optional diagnostic)
    # ------------------------------------------------------------------
    if (
        "mean_err_cmd_m" in open_df.columns
        and "mean_err_cmd_m" in pid_df.columns
        and "mean_err_cmd_m" in ag_df.columns
    ):
        plot_curve(
            open_df,
            "mean_err_cmd_m",
            "Open-loop mean COMMANDED tracking error (mean ± 95% CI)",
            figs / "openloop_mean_err_cmd.png",
            ylabel="mean_err_cmd_m",
        )
        plot_curve(
            pid_df,
            "mean_err_cmd_m",
            "PID mean COMMANDED tracking error (mean ± 95% CI)",
            figs / "pid_mean_err_cmd.png",
            ylabel="mean_err_cmd_m",
        )
        plot_curve(
            ag_df,
            "mean_err_cmd_m",
            "Agentic mean COMMANDED tracking error (mean ± 95% CI)",
            figs / "agentic_mean_err_cmd.png",
            ylabel="mean_err_cmd_m",
        )

    # ------------------------------------------------------------------
    # Figures: formation + connectivity (only if present)
    # ------------------------------------------------------------------
    if (
        "formation_err_rel" in open_df.columns
        and "formation_err_rel" in pid_df.columns
        and "formation_err_rel" in ag_df.columns
    ):
        plot_curve(
            open_df,
            "formation_err_rel",
            "Open-loop relative formation error (mean ± 95% CI)",
            figs / "openloop_formation_err_rel.png",
            ylabel="formation_err_rel",
        )
        plot_curve(
            pid_df,
            "formation_err_rel",
            "PID relative formation error (mean ± 95% CI)",
            figs / "pid_formation_err_rel.png",
            ylabel="formation_err_rel",
        )
        plot_curve(
            ag_df,
            "formation_err_rel",
            "Agentic relative formation error (mean ± 95% CI)",
            figs / "agentic_formation_err_rel.png",
            ylabel="formation_err_rel",
        )

    if (
        "connectivity_rate" in open_df.columns
        and "connectivity_rate" in pid_df.columns
        and "connectivity_rate" in ag_df.columns
    ):
        plot_curve(
            open_df,
            "connectivity_rate",
            "Open-loop connectivity rate (mean ± 95% CI)",
            figs / "openloop_connectivity_rate.png",
            ylabel="connectivity_rate",
        )
        plot_curve(
            pid_df,
            "connectivity_rate",
            "PID connectivity rate (mean ± 95% CI)",
            figs / "pid_connectivity_rate.png",
            ylabel="connectivity_rate",
        )
        plot_curve(
            ag_df,
            "connectivity_rate",
            "Agentic connectivity rate (mean ± 95% CI)",
            figs / "agentic_connectivity_rate.png",
            ylabel="connectivity_rate",
        )

    # ------------------------------------------------------------------
    # Summary CSV (explicit semantics)
    # ------------------------------------------------------------------
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