import pandas as pd
import numpy as np
import glob

STEADY_T = 25.0   # Phase 3 start

controllers = ["openloop", "pid", "agentic"]
rows = []

for ctrl in controllers:
    files = sorted(glob.glob(f"outputs/csv/{ctrl}_seed*.csv"))

    for f in files:
        seed = int(f.split("seed")[-1].split(".")[0])
        df = pd.read_csv(f)

        # Phase 3 only
        df = df[df["t"] >= STEADY_T]

        rows.append({
            "controller": ctrl,
            "seed": seed,

            "mean_err_nominal_m": df["mean_err_nominal_m"].mean(),
            "max_err_nominal_m": df["max_err_nominal_m"].max(),

            "mean_err_cmd_m": df["mean_err_cmd_m"].mean(),
            "max_err_cmd_m": df["max_err_cmd_m"].max(),

            "formation_err_rel": df["formation_err_rel"].mean(),
            "connectivity_rate": df["connectivity_rate"].mean(),

            "agentic_active_rate": df["agentic_active"].mean(),
            "agentic_ref_shift_mean": df["agentic_ref_shift"].mean(),
        })

summary = pd.DataFrame(rows)

# ---- Aggregate across seeds ----
final = (
    summary
    .groupby("controller")
    .agg(
        mean_err_nominal_m=("mean_err_nominal_m", "mean"),
        max_err_nominal_m=("max_err_nominal_m", "mean"),

        mean_err_cmd_m=("mean_err_cmd_m", "mean"),
        max_err_cmd_m=("max_err_cmd_m", "mean"),

        formation_err_rel=("formation_err_rel", "mean"),
        connectivity_rate=("connectivity_rate", "mean"),

        agentic_active_rate=("agentic_active_rate", "mean"),
        agentic_ref_shift_mean=("agentic_ref_shift_mean", "mean"),
    )
    .reset_index()
)

# Pretty print
pd.set_option("display.float_format", "{:.6f}".format)
print(final.to_string(index=False))

# Optional: save
final.to_csv("outputs/summary_phase3.csv", index=False)
print("\nSaved: outputs/summary_phase3.csv")