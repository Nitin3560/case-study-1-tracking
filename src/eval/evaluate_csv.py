from __future__ import annotations
import os, glob
import pandas as pd
import numpy as np

ERR_COL_CANDIDATES = ["com_err", "err_xy", "err_xy_norm", "tracking_err", "pos_err"]

def pick_err_col(df: pd.DataFrame) -> str:
    for c in ERR_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No known error column found. Have: {list(df.columns)}")

def window_metrics(df: pd.DataFrame, t0: float) -> dict:
    w = df[df["t"] >= t0]
    if len(w) == 0:
        return {"mean": np.nan, "max": np.nan, "n": 0}
    err_col = pick_err_col(w)
    return {
        "mean": float(w[err_col].mean()),
        "max": float(w[err_col].max()),
        "n": int(len(w)),
        "err_col": err_col
    }

def segmented_mean_of_means(df: pd.DataFrame, t0: float, seg_s: float) -> dict:
    w = df[df["t"] >= t0].copy()
    if len(w) == 0:
        return {"seg_mean_of_means": np.nan, "segments": 0}

    err_col = pick_err_col(w)
    w["seg"] = np.floor((w["t"] - t0) / seg_s).astype(int)

    seg_means = w.groupby("seg")[err_col].mean().values
    return {
        "seg_mean_of_means": float(np.mean(seg_means)) if len(seg_means) else np.nan,
        "segments": int(len(seg_means)),
        "seg_s": float(seg_s),
        "err_col": err_col
    }

def evaluate_folder(csv_glob: str, t0: float = 10.0, seg_s: float = 2.0) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(csv_glob)):
        df = pd.read_csv(path)
        if "t" not in df.columns:
            raise ValueError(f"{path} missing 't' column")

        wm = window_metrics(df, t0=t0)
        sm = segmented_mean_of_means(df, t0=t0, seg_s=seg_s)

        rows.append({
            "file": os.path.basename(path),
            "t0": t0,
            "seg_s": seg_s,
            "win_mean": wm["mean"],
            "win_max": wm["max"],
            "seg_mean_of_means": sm["seg_mean_of_means"],
            "segments": sm["segments"],
            "err_col": wm.get("err_col", sm.get("err_col", "")),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Example:
    #   python -m src.eval.evaluate_csv -- but if you run directly:
    df = evaluate_folder("outputs/csv/*.csv", t0=10.0, seg_s=2.0)
    print(df.to_string(index=False))
