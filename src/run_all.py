# src/run_all.py

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    return ap.parse_args()


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


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
        "supervisor_active_rate": _mean("supervisor_active"),
        "supervisor_mode": _mean("supervisor_mode"),
        "ref_shift_norm": _mean("ref_shift_norm"),
        "formation_scale": _mean("formation_scale"),
        "smooth_alpha": _mean("smooth_alpha"),
        "int_hold_rate": _mean("int_hold"),

        # Network/task/failure proxies
        "throughput_ratio": _mean("throughput_ratio"),
        "latency_proxy": _mean("latency_proxy"),
        "control_overhead": _mean("control_overhead"),
        "task_reassign_rate": _mean("task_reassign"),
        "task_cost_mean": _mean("task_cost_mean"),
        "alive_ratio": _mean("alive_ratio"),
        "failed_count": _mean("failed_count"),
        "task_completion_proxy": _mean("task_completion_proxy"),
        "system_utility": _mean("system_utility"),
    }

    # Backward-compatible legacy columns (do not remove unless you update downstream)
    out["mean_err_m"] = mean_nom
    out["max_err_m"] = max_nom

    return out


def summarize_windows(df: pd.DataFrame, controller_name: str) -> pd.DataFrame:
    """
    Windowed (phase-based) summary. Uses wind_phase if present.
    Returns per-window mean over time, then averaged over seeds.
    """
    if "wind_phase" not in df.columns:
        return pd.DataFrame()

    # mean per seed within each phase, then average across seeds
    g = df.groupby(["wind_phase", "seed"])
    per_seed = g.agg(
        mean_err_m=("mean_err_m", "mean"),
        max_err_m=("max_err_m", "max"),
        formation_err_rel=("formation_err_rel", "mean"),
        connectivity_rate=("connectivity_rate", "mean"),
        task_completion_proxy=("task_completion_proxy", "mean"),
        system_utility=("system_utility", "mean"),
    ).reset_index()

    out = (
        per_seed
        .groupby("wind_phase")
        .agg(
            mean_err_m=("mean_err_m", "mean"),
            max_err_m=("max_err_m", "mean"),
            formation_err_rel=("formation_err_rel", "mean"),
            connectivity_rate=("connectivity_rate", "mean"),
            task_completion_proxy=("task_completion_proxy", "mean"),
            system_utility=("system_utility", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "controller", controller_name)
    return out


def summarize_fixed_bins(df: pd.DataFrame, controller_name: str, bins: list[float]) -> pd.DataFrame:
    """
    Fixed time-bin summary. bins is a list of bin edges in seconds.
    Returns per-bin mean over time, then averaged over seeds.
    """
    if "t" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["time_bin"] = pd.cut(df["t"], bins=bins, right=False, include_lowest=True)

    g = df.groupby(["time_bin", "seed"], observed=False)
    per_seed = g.agg(
        mean_err_m=("mean_err_m", "mean"),
        max_err_m=("max_err_m", "max"),
        formation_err_rel=("formation_err_rel", "mean"),
        connectivity_rate=("connectivity_rate", "mean"),
        task_completion_proxy=("task_completion_proxy", "mean"),
        system_utility=("system_utility", "mean"),
    ).reset_index()

    out = (
        per_seed
        .groupby("time_bin", observed=False)
        .agg(
            mean_err_m=("mean_err_m", "mean"),
            max_err_m=("max_err_m", "mean"),
            formation_err_rel=("formation_err_rel", "mean"),
            connectivity_rate=("connectivity_rate", "mean"),
            task_completion_proxy=("task_completion_proxy", "mean"),
            system_utility=("system_utility", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "controller", controller_name)
    return out


def _first_fault_t0(cfg: Dict[str, Any]) -> float | None:
    faults = (cfg.get("faults", {}) or {})
    events = faults.get("events", []) or []
    if not events:
        return None
    t0_vals = []
    for ev in events:
        try:
            t0_vals.append(float(ev.get("t0_s")))
        except Exception:
            continue
    return min(t0_vals) if t0_vals else None


def _first_fault_event(cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    faults = (cfg.get("faults", {}) or {})
    events = faults.get("events", []) or []
    if not events:
        return None
    valid = []
    for ev in events:
        try:
            t0 = float(ev.get("t0_s"))
            valid.append((t0, ev))
        except Exception:
            continue
    if not valid:
        return None
    valid.sort(key=lambda x: x[0])
    return valid[0][1]


def _pct_change(new: float, base: float) -> float:
    if np.isnan(new) or np.isnan(base) or abs(base) < 1e-9:
        return float("nan")
    return float((new - base) / base * 100.0)


def _abs_degradation_pct(new: float, base: float) -> float:
    if np.isnan(new) or np.isnan(base) or abs(base) < 1e-9:
        return float("nan")
    return float((abs(new) - abs(base)) / abs(base) * 100.0)


def _build_fault_degradation_table(fault_df: pd.DataFrame) -> pd.DataFrame:
    out = fault_df.copy()
    out["err_fault_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["mean_err_fault"], out["mean_err_pre"])
    ]
    out["err_post_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["mean_err_post"], out["mean_err_pre"])
    ]
    out["conn_fault_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["connectivity_fault"], out["connectivity_pre"])
    ]
    out["conn_post_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["connectivity_post"], out["connectivity_pre"])
    ]
    out["task_fault_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["task_completion_fault"], out["task_completion_pre"])
    ]
    out["task_post_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["task_completion_post"], out["task_completion_pre"])
    ]
    out["utility_fault_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["system_utility_fault"], out["system_utility_pre"])
    ]
    out["utility_post_vs_pre_pct"] = [
        _pct_change(n, b) for n, b in zip(out["system_utility_post"], out["system_utility_pre"])
    ]
    # Utility is often negative; these columns are sign-safe for reporting degradation.
    out["utility_fault_delta"] = [
        float(n - b) if not (np.isnan(n) or np.isnan(b)) else float("nan")
        for n, b in zip(out["system_utility_fault"], out["system_utility_pre"])
    ]
    out["utility_post_delta"] = [
        float(n - b) if not (np.isnan(n) or np.isnan(b)) else float("nan")
        for n, b in zip(out["system_utility_post"], out["system_utility_pre"])
    ]
    out["utility_fault_abs_degradation_pct"] = [
        _abs_degradation_pct(n, b) for n, b in zip(out["system_utility_fault"], out["system_utility_pre"])
    ]
    out["utility_post_abs_degradation_pct"] = [
        _abs_degradation_pct(n, b) for n, b in zip(out["system_utility_post"], out["system_utility_pre"])
    ]
    return out


def _plot_fault_timeline(
    open_df: pd.DataFrame,
    pid_df: pd.DataFrame,
    ag_df: pd.DataFrame,
    outpath: Path,
    t0: float,
    t1: float | None = None,
) -> None:
    plt.figure(figsize=(9, 6))
    ax1 = plt.subplot(2, 1, 1)
    for name, df, color in [
        ("openloop", open_df, "tab:blue"),
        ("pid", pid_df, "tab:orange"),
        ("agentic", ag_df, "tab:green"),
    ]:
        mean, _ = mean_ci_over_seeds(df, "mean_err_m")
        ax1.plot(mean.index.values, mean.values, label=name, color=color)
    ax1.axvline(float(t0), linestyle="--", color="k", linewidth=1.0)
    if t1 is not None:
        ax1.axvspan(float(t0), float(t1), color="gray", alpha=0.15)
    ax1.set_ylabel("Mean Error (m)")
    ax1.set_title("Fault Timeline: Tracking Error")
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    for name, df, color in [
        ("openloop", open_df, "tab:blue"),
        ("pid", pid_df, "tab:orange"),
        ("agentic", ag_df, "tab:green"),
    ]:
        mean, _ = mean_ci_over_seeds(df, "connectivity_rate")
        ax2.plot(mean.index.values, mean.values, label=name, color=color)
    ax2.axvline(float(t0), linestyle="--", color="k", linewidth=1.0)
    if t1 is not None:
        ax2.axvspan(float(t0), float(t1), color="gray", alpha=0.15)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Connectivity")
    ax2.set_title("Fault Timeline: Connectivity")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def _recovery_time(df: pd.DataFrame, t0: float, thr_mult: float = 1.5, hold_s: float = 1.0) -> float:
    if "t" not in df.columns or "mean_err_m" not in df.columns:
        return float("nan")
    pre = df[(df["t"] >= t0 - 5.0) & (df["t"] < t0)]["mean_err_m"]
    if pre.empty:
        return float("nan")
    thr = float(pre.mean()) * float(thr_mult)
    ts = df["t"].to_numpy()
    es = df["mean_err_m"].to_numpy()
    for i in range(len(ts)):
        if ts[i] < t0:
            continue
        t1 = ts[i] + hold_s
        j = int(np.searchsorted(ts, t1, side="right"))
        if j <= i:
            continue
        if np.all(es[i:j] <= thr):
            return float(ts[i] - t0)
    return float("nan")


def summarize_fault_windows(df: pd.DataFrame, controller_name: str, t0: float) -> pd.DataFrame:
    if "t" not in df.columns:
        return pd.DataFrame()
    bins = [t0 - 5.0, t0, t0 + 5.0, t0 + 15.0]
    labels = ["pre", "fault", "post"]
    d = df.copy()
    d["fault_window"] = pd.cut(d["t"], bins=bins, labels=labels, right=False, include_lowest=True)
    d = d[d["fault_window"].notna()]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(["fault_window", "seed"], observed=False)
    per_seed = g.agg(
        mean_err_m=("mean_err_m", "mean"),
        max_err_m=("max_err_m", "max"),
        formation_err_rel=("formation_err_rel", "mean"),
        connectivity_rate=("connectivity_rate", "mean"),
        task_completion_proxy=("task_completion_proxy", "mean"),
        system_utility=("system_utility", "mean"),
    ).reset_index()
    out = (
        per_seed
        .groupby("fault_window", observed=False)
        .agg(
            mean_err_m=("mean_err_m", "mean"),
            max_err_m=("max_err_m", "mean"),
            formation_err_rel=("formation_err_rel", "mean"),
            connectivity_rate=("connectivity_rate", "mean"),
            task_completion_proxy=("task_completion_proxy", "mean"),
            system_utility=("system_utility", "mean"),
        )
        .reset_index()
    )
    out.insert(0, "controller", controller_name)
    return out


def summarize_faults(df: pd.DataFrame, controller_name: str, t0: float) -> Dict[str, Any]:
    def _seg_mean(t_start: float, t_end: float, col: str = "mean_err_m") -> float:
        seg = df[(df["t"] >= t_start) & (df["t"] < t_end)]
        if seg.empty or col not in seg.columns:
            return float("nan")
        return float(seg[col].mean())

    pre = _seg_mean(t0 - 5.0, t0)
    fault = _seg_mean(t0, t0 + 5.0)
    post = _seg_mean(t0 + 5.0, t0 + 15.0)
    pre_conn = _seg_mean(t0 - 5.0, t0, "connectivity_rate")
    fault_conn = _seg_mean(t0, t0 + 5.0, "connectivity_rate")
    post_conn = _seg_mean(t0 + 5.0, t0 + 15.0, "connectivity_rate")
    pre_task = _seg_mean(t0 - 5.0, t0, "task_completion_proxy")
    fault_task = _seg_mean(t0, t0 + 5.0, "task_completion_proxy")
    post_task = _seg_mean(t0 + 5.0, t0 + 15.0, "task_completion_proxy")
    pre_util = _seg_mean(t0 - 5.0, t0, "system_utility")
    fault_util = _seg_mean(t0, t0 + 5.0, "system_utility")
    post_util = _seg_mean(t0 + 5.0, t0 + 15.0, "system_utility")
    peak_fault = float(df[(df["t"] >= t0) & (df["t"] < t0 + 5.0)]["max_err_m"].max())
    rec = _recovery_time(df, t0=t0, thr_mult=1.5, hold_s=1.0)
    return {
        "controller": controller_name,
        "fault_t0_s": float(t0),
        "mean_err_pre": pre,
        "mean_err_fault": fault,
        "mean_err_post": post,
        "max_err_fault": peak_fault,
        "recovery_time_s": rec,
        "connectivity_pre": pre_conn,
        "connectivity_fault": fault_conn,
        "connectivity_post": post_conn,
        "task_completion_pre": pre_task,
        "task_completion_fault": fault_task,
        "task_completion_post": post_task,
        "system_utility_pre": pre_util,
        "system_utility_fault": fault_util,
        "system_utility_post": post_util,
    }


def main():
    args = parse_args()
    seeds = args.seeds
    cfg = load_cfg(args.config)

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
    # Figures: network/task proxies (if present)
    # ------------------------------------------------------------------
    if (
        "throughput_ratio" in open_df.columns
        and "throughput_ratio" in pid_df.columns
        and "throughput_ratio" in ag_df.columns
    ):
        plot_curve(
            open_df,
            "throughput_ratio",
            "Open-loop throughput ratio (mean ± 95% CI)",
            figs / "openloop_throughput_ratio.png",
            ylabel="throughput_ratio",
        )
        plot_curve(
            pid_df,
            "throughput_ratio",
            "PID throughput ratio (mean ± 95% CI)",
            figs / "pid_throughput_ratio.png",
            ylabel="throughput_ratio",
        )
        plot_curve(
            ag_df,
            "throughput_ratio",
            "Agentic throughput ratio (mean ± 95% CI)",
            figs / "agentic_throughput_ratio.png",
            ylabel="throughput_ratio",
        )

    if (
        "latency_proxy" in open_df.columns
        and "latency_proxy" in pid_df.columns
        and "latency_proxy" in ag_df.columns
    ):
        plot_curve(
            open_df,
            "latency_proxy",
            "Open-loop latency proxy (mean ± 95% CI)",
            figs / "openloop_latency_proxy.png",
            ylabel="latency_proxy",
        )
        plot_curve(
            pid_df,
            "latency_proxy",
            "PID latency proxy (mean ± 95% CI)",
            figs / "pid_latency_proxy.png",
            ylabel="latency_proxy",
        )
        plot_curve(
            ag_df,
            "latency_proxy",
            "Agentic latency proxy (mean ± 95% CI)",
            figs / "agentic_latency_proxy.png",
            ylabel="latency_proxy",
        )

    if (
        "control_overhead" in open_df.columns
        and "control_overhead" in pid_df.columns
        and "control_overhead" in ag_df.columns
    ):
        plot_curve(
            open_df,
            "control_overhead",
            "Open-loop control overhead (mean ± 95% CI)",
            figs / "openloop_control_overhead.png",
            ylabel="control_overhead",
        )
        plot_curve(
            pid_df,
            "control_overhead",
            "PID control overhead (mean ± 95% CI)",
            figs / "pid_control_overhead.png",
            ylabel="control_overhead",
        )
        plot_curve(
            ag_df,
            "control_overhead",
            "Agentic control overhead (mean ± 95% CI)",
            figs / "agentic_control_overhead.png",
            ylabel="control_overhead",
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

    # Windowed summary (phase-based)
    win_rows = []
    for ctrl_name, dfc in [("openloop", open_df), ("pid", pid_df), ("agentic", ag_df)]:
        win = summarize_windows(dfc, ctrl_name)
        if not win.empty:
            win_rows.append(win)
    if win_rows:
        win_df = pd.concat(win_rows, ignore_index=True)
        win_path = Path("outputs") / "summary_windows.csv"
        win_df.to_csv(win_path, index=False)
        print(f"Saved: {win_path.resolve()}")

    # Fixed-bin summary (time bins)
    bins = [0.0, 20.0, 40.0, 60.0, 80.0, 1e9]
    bin_rows = []
    for ctrl_name, dfc in [("openloop", open_df), ("pid", pid_df), ("agentic", ag_df)]:
        bdf = summarize_fixed_bins(dfc, ctrl_name, bins)
        if not bdf.empty:
            bin_rows.append(bdf)
    if bin_rows:
        bin_df = pd.concat(bin_rows, ignore_index=True)
        bin_path = Path("outputs") / "summary_bins.csv"
        bin_df.to_csv(bin_path, index=False)
        print(f"Saved: {bin_path.resolve()}")

    # Fault-centered summaries (if fault events exist in config)
    fault_event = _first_fault_event(cfg)
    fault_t0 = _first_fault_t0(cfg)
    if fault_t0 is not None:
        fault_rows = [
            summarize_faults(open_df, "openloop", fault_t0),
            summarize_faults(pid_df, "pid", fault_t0),
            summarize_faults(ag_df, "agentic", fault_t0),
        ]
        fault_df = pd.DataFrame(fault_rows)
        fault_path = Path("outputs") / "summary_faults.csv"
        fault_df.to_csv(fault_path, index=False)
        print(f"Saved: {fault_path.resolve()}")

        fw_rows = []
        for ctrl_name, dfc in [("openloop", open_df), ("pid", pid_df), ("agentic", ag_df)]:
            fw = summarize_fault_windows(dfc, ctrl_name, fault_t0)
            if not fw.empty:
                fw_rows.append(fw)
        if fw_rows:
            fw_df = pd.concat(fw_rows, ignore_index=True)
            fw_path = Path("outputs") / "summary_fault_windows.csv"
            fw_df.to_csv(fw_path, index=False)
            print(f"Saved: {fw_path.resolve()}")

        degr_df = _build_fault_degradation_table(fault_df)
        degr_path = Path("outputs") / "summary_fault_degradation.csv"
        degr_df.to_csv(degr_path, index=False)
        print(f"Saved: {degr_path.resolve()}")

        t1 = None
        if fault_event is not None and fault_event.get("t1_s", None) is not None:
            try:
                t1 = float(fault_event.get("t1_s"))
            except Exception:
                t1 = None
        timeline_path = Path("outputs/figs") / "fault_timeline_error_connectivity.png"
        _plot_fault_timeline(open_df, pid_df, ag_df, timeline_path, t0=float(fault_t0), t1=t1)
        print(f"Saved: {timeline_path.resolve()}")

    print(f"Saved: {summary_path.resolve()}")
    print(f"Saved figures in: {figs.resolve()}")


if __name__ == "__main__":
    main()
