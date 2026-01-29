#!/usr/bin/env python3
import glob, re
import numpy as np
import pandas as pd

# EDIT THESE 3 GLOBS TO MATCH YOUR FILES
OPENLOOP_GLOB = "outputs/openloop/**/*.csv"
PID_GLOB      = "outputs/pid/**/*.csv"
AGENTIC_GLOB  = "outputs/agentic/**/*.csv"

T0 = 10.0  # steady-state start time (seconds). change if needed.

TIME_COL_CAND = ["t", "time", "timestamp", "sec", "seconds"]
ERR_COL_CAND  = ["tracking_error", "pos_error", "position_error", "error", "rmse", "e"]

def seed_from_path(p: str) -> int:
    m = re.search(r"seed[_\\- ]*(\\d+)", p, flags=re.IGNORECASE)
    if m: return int(m.group(1))
    nums = re.findall(r"(\\d+)", p)
    return int(nums[-1]) if nums else -1

def pick_col(cols, cands):
    lower = {c.lower(): c for c in cols}
    for k in cands:
        if k.lower() in lower: return lower[k.lower()]
    for k in cands:
        for c in cols:
            if k.lower() in c.lower(): return c
    return None

def load_stats(path: str):
    df = pd.read_csv(path)
    tcol = pick_col(df.columns, TIME_COL_CAND)
    ecol = pick_col(df.columns, ERR_COL_CAND)
    if tcol is None or ecol is None:
        raise RuntimeError(f"Column detect failed for {path}. Columns={list(df.columns)}")

    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy()
    e = pd.to_numeric(df[ecol], errors="coerce").to_numpy()
    m = np.isfinite(t) & np.isfinite(e)
    t, e = t[m], e[m]
    if t.size == 0:
        raise RuntimeError(f"No valid numeric rows in {path}")

    # sort by time just in case
    if not np.all(np.diff(t) >= 0):
        idx = np.argsort(t)
        t, e = t[idx], e[idx]

    seg = e[t >= T0]
    if seg.size == 0:
        seg = e[-max(1, int(0.1 * len(e))):]  # fallback: last 10%

    return float(np.mean(seg)), float(np.max(seg))

def main():
    ol_files = glob.glob(OPENLOOP_GLOB, recursive=True)
    pid_files = glob.glob(PID_GLOB, recursive=True)
    ag_files = glob.glob(AGENTIC_GLOB, recursive=True)

    ol_map  = {seed_from_path(p): p for p in ol_files}
    pid_map = {seed_from_path(p): p for p in pid_files}
    ag_map  = {seed_from_path(p): p for p in ag_files}

    seeds = sorted(set(ol_map) & set(pid_map) & set(ag_map))
    if not seeds:
        raise SystemExit("No common seeds found. Check your GLOB paths.")

    rows = []
    ol_means, pid_means, ag_means = [], [], []

    for s in seeds:
        ol_mean, ol_max   = load_stats(ol_map[s])
        pid_mean, pid_max = load_stats(pid_map[s])
        ag_mean, ag_max   = load_stats(ag_map[s])

        ag_vs_pid = (pid_mean - ag_mean) / max(pid_mean, 1e-9) * 100.0
        ag_vs_ol  = (ol_mean  - ag_mean) / max(ol_mean,  1e-9) * 100.0

        rows.append([
            s,
            f"{ol_mean:.2f}/{ol_max:.2f}",
            f"{pid_mean:.2f}/{pid_max:.2f}",
            f"{ag_mean:.2f}/{ag_max:.2f}",
            f"{ag_vs_pid:.2f}",
            f"{ag_vs_ol:.2f}",
        ])

        ol_means.append(ol_mean)
        pid_means.append(pid_mean)
        ag_means.append(ag_mean)

    print(f"\nSteady-state window: t >= {T0:.1f}s\n")
    print(" Seed  OpenLoop mean/max   PID mean/max     Agentic mean/max   Agentic vs PID (%)  Agentic vs OpenLoop (%)")
    for r in rows:
        print(f"{r[0]:>5}  {r[1]:>15}  {r[2]:>13}  {r[3]:>16}  {r[4]:>16}  {r[5]:>21}")

    pct_ol = (np.mean(ol_means) - np.mean(ag_means)) / max(np.mean(ol_means), 1e-9) * 100.0
    print(f"\nAgentic vs OpenLoop reduction (avg means) = {pct_ol:.2f}%")

if __name__ == "__main__":
    main()
