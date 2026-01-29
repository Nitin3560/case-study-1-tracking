from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from .env_factory import make_env
from .disturbances import DisturbanceModel
from .trajectories import p_ref, formation_offsets
from .metrics import connectivity_rate, formation_error_relative

from .controllers.pid_tracker import PIDTracker
from .controllers.openloop_ff import OpenLoopFeedforwardFollower


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _safe_connectivity(pos: np.ndarray, comm_range: float) -> float:
    """
    Compatibility wrapper: connectivity_rate() signature may differ across versions.
    """
    try:
        return float(connectivity_rate(pos, comm_range=comm_range))
    except TypeError:
        return float(connectivity_rate(pos, comm_range_m=comm_range, use_3d=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--controller",
        type=str,
        required=True,
        choices=["openloop", "pid", "agentic"],
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Always define to avoid NameError in non-agentic runs
    agentic = None

    # -------------------------------
    # Config + seeding
    # -------------------------------
    cfg = _load_cfg(args.config)
    cfg.setdefault("sim", {})
    cfg["sim"]["seed"] = int(args.seed)

    # -------------------------------
    # Env
    # -------------------------------
    handles = make_env(cfg)
    env = handles.env
    dt_sim = float(handles.dt_sim)
    dt_ctrl = float(handles.dt_ctrl)

    sim_cfg = cfg.get("sim", {}) or {}
    n = int(sim_cfg.get("num_drones", 1))
    duration = float(sim_cfg["duration_s"])
    steps = int(round(duration / dt_sim))

    offs = formation_offsets(cfg, n)
    dist = DisturbanceModel(cfg, seed=int(args.seed))

    drone_model = env.DRONE_MODEL

    # -------------------------------
    # Controllers
    # -------------------------------
    openloop = OpenLoopFeedforwardFollower(cfg, drone_model)
    pid = PIDTracker(cfg, drone_model)

    # Lazy import: non-agentic runs never import AgenticReplanner (avoids import-path issues)
    if args.controller == "agentic":
        from .controllers.agentic_replanner import AgenticReplanner

        agentic = AgenticReplanner(cfg, drone_model)

    # -------------------------------
    # Reset
    # -------------------------------
    obs, _ = env.reset(seed=int(args.seed))
    if hasattr(openloop, "reset"):
        openloop.reset()
    if hasattr(pid, "reset"):
        pid.reset()
    if agentic and hasattr(agentic, "reset"):
        agentic.reset()

    action = np.zeros((n, 4), dtype=float)
    ctrl_decim = max(1, int(round(dt_ctrl / dt_sim)))

    rows = []
    comm_range = float(sim_cfg.get("comm_range_m", 10.0))

    # Agentic lookahead config (safe default)
    agentic_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
    lookahead_s = float(agentic_cfg.get("lookahead_s", 0.5))

    # -------------------------------
    # Control loop
    # -------------------------------
    for k in tqdm(range(steps)):
        t = k * dt_sim
        dist.apply_wind(env, t)

        # Fast sim stepping between control ticks
        if (k % ctrl_decim) != 0:
            obs, *_ = env.step(action)
            continue

        # -------------------------------
        # Sensing (truth vs GPS measurement)
        # -------------------------------
        obs_meas = [np.array(o, dtype=float).copy() for o in obs]
        for i in range(n):
            true_pos = obs[i][0:3]
            obs_meas[i][0:3] = dist.gps_measurement(true_pos=true_pos, t=t, dt=dt_sim)

        sense_cfg = cfg.get("controllers", {}).get("sensing", {}) or {}
        mode = sense_cfg.get(args.controller, "gps").lower()
        obs_ctrl = obs_meas if mode == "gps" else obs

        # -------------------------------
        # Nominal trajectory (now + lookahead)
        # -------------------------------
        p_now, v_now = p_ref(cfg, t)
        p_ahead, v_ahead = p_ref(cfg, t + lookahead_s)

        p_des_now = np.zeros((n, 3), dtype=float)
        v_des_now = np.zeros((n, 3), dtype=float)
        p_des_ahead = np.zeros((n, 3), dtype=float)
        v_des_ahead = np.zeros((n, 3), dtype=float)

        for i in range(n):
            p_des_now[i] = p_now + offs[i]
            v_des_now[i] = v_now
            p_des_ahead[i] = p_ahead + offs[i]
            v_des_ahead[i] = v_ahead

        # What controller actually used (truth-metrics must reference this)
        p_des_used = np.zeros((n, 3), dtype=float)

        # Agentic instrumentation per control tick
        agentic_active_step = 0.0
        agentic_ref_shift_step = 0.0

        # -------------------------------
        # Control
        # -------------------------------
        if args.controller == "openloop":
            for i in range(n):
                p_des_used[i] = p_des_now[i]
                action[i] = openloop.compute_rpms(
                    obs_ctrl[i],
                    p_des_used[i],
                    v_des_now[i],
                    yaw_des=0.0,
                )

        elif args.controller == "pid":
            for i in range(n):
                p_des_used[i] = p_des_now[i]
                action[i] = pid.compute_rpms(
                    obs_ctrl[i],
                    target_pos=p_des_used[i],
                    target_vel=v_des_now[i],
                    integrate_z=False,  # keep consistent with agentic XY-only integral design
                )

        else:
            # AGENTIC: supervisor owns its own PIDTracker and applies integrator supervision internally
            assert agentic is not None

            applied = []
            shifts = []

            for i in range(n):
                (
                    rpms,
                    err,
                    sat,
                    gate_ok,
                    bias_norm,
                    target_pos_adj,
                    apply_bias_float,
                ) = agentic.compute_rpms(
                    state=obs_ctrl[i],
                    p_ref_now=p_des_now[i],
                    v_ref_now=v_des_now[i],
                    p_ref_ahead=p_des_ahead[i],
                    v_ref_ahead=v_des_ahead[i],
                    yaw_des=0.0,
                )

                action[i] = rpms
                p_des_used[i] = np.asarray(target_pos_adj, dtype=float)

                applied.append(float(apply_bias_float))
                delta_xy = (p_des_used[i] - p_des_now[i])[:2]
                shifts.append(float(np.linalg.norm(delta_xy)))

            agentic_active_step = float(np.mean(applied)) if applied else 0.0
            agentic_ref_shift_step = float(np.mean(shifts)) if shifts else 0.0

        # -------------------------------
        # Metrics (truth-based)
        # -------------------------------
        pos_true = np.stack([obs[i][0:3] for i in range(n)])
        track_errs = np.linalg.norm((pos_true - p_des_used)[:, :2], axis=1)

        rows.append(
            {
                "t": t,
                "mean_err_m": float(np.mean(track_errs)),
                "max_err_m": float(np.max(track_errs)),
                "formation_err_rel": float(formation_error_relative(pos_true, offs)),
                "connectivity_rate": _safe_connectivity(pos_true, comm_range),
                "agentic_active": float(agentic_active_step),
                "agentic_ref_shift": float(agentic_ref_shift_step),
            }
        )

        obs, *_ = env.step(action)

    os.makedirs("outputs/csv", exist_ok=True)
    out_path = f"outputs/csv/{args.controller}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
