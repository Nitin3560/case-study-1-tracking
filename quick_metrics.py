import yaml
import numpy as np
import pandas as pd

CFG_PATH = "configs/scenario_truth.yaml"
SEEDS = range(1, 11)

T_COL = "t"
MEAN_COL = "mean_err_m"
MAX_COL = "max_err_m"


def steady_stats(path, t0):
    df = pd.read_csv(path)
    d = df[df[T_COL] >= t0]
    if len(d) == 0:
        raise ValueError(f"{path}: no rows with t >= {t0}")
    return float(d[MEAN_COL].mean()), float(d[MAX_COL].max())


# load recovery time from yaml
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f) or {}

t0_baseline = 10.0
t0_recovery = float((cfg.get("metrics", {}) or {}).get("steady_state_after_s", 25.0))


def make_table(t0):
    rows = []
    ol_means, pid_means, ag_means = [], [], []

    for s in SEEDS:
        ol_path  = f"outputs/csv/openloop_seed{s}.csv"
        pid_path = f"outputs/csv/pid_seed{s}.csv"
        ag_path  = f"outputs/csv/agentic_seed{s}.csv"

        ol_mean, ol_max   = steady_stats(ol_path, t0)
        pid_mean, pid_max = steady_stats(pid_path, t0)
        ag_mean, ag_max   = steady_stats(ag_path, t0)

        rows.append({
            "Seed": s,
            "OpenLoop mean/max": f"{ol_mean:.2f}/{ol_max:.2f}",
            "PID mean/max":      f"{pid_mean:.2f}/{pid_max:.2f}",
            "Agentic mean/max":  f"{ag_mean:.2f}/{ag_max:.2f}",
            "Agentic vs PID (%)":      100.0 * (pid_mean - ag_mean) / max(pid_mean, 1e-9),
            "Agentic vs OpenLoop (%)": 100.0 * (ol_mean  - ag_mean) / max(ol_mean,  1e-9),
        })

        ol_means.append(ol_mean)
        pid_means.append(pid_mean)
        ag_means.append(ag_mean)

    df = pd.DataFrame(rows)

    print(f"\nSteady-state window: t >= {t0:.1f}s\n")
    print(df.to_string(index=False, formatters={
        "Agentic vs PID (%)": lambda x: f"{x:8.2f}",
        "Agentic vs OpenLoop (%)": lambda x: f"{x:8.2f}",
    }))

    print("\nAverages over seeds (steady-state):")
    print(f"OpenLoop mean = {np.mean(ol_means):.2f}")
    print(f"PID      mean = {np.mean(pid_means):.2f}")
    print(f"Agentic  mean = {np.mean(ag_means):.2f}")

    pct_ol = (np.mean(ol_means) - np.mean(ag_means)) / max(np.mean(ol_means), 1e-9) * 100
    print(f"\nAgentic vs OpenLoop reduction (avg means) = {pct_ol:.2f}%")


print("=== WINDOW A: Post-transient (t >= 10s) ===")
make_table(t0_baseline)

print("\n=== WINDOW B: Recovery window (t >= metrics.steady_state_after_s) ===")
make_table(t0_recovery)
