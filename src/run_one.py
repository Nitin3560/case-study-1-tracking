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

# Controllers
from .controllers.openloop import OpenLoopController
from .controllers.pid import PIDController
from .controllers.pid_agentic import PIDAgenticController


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _safe_connectivity(pos: np.ndarray, comm_range: float) -> float:
    try:
        return float(connectivity_rate(pos, comm_range=comm_range))
    except TypeError:
        return float(connectivity_rate(pos, comm_range_m=comm_range, use_3d=False))


def _sat_to_frac(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (float, np.floating, int, np.integer)):
        return float(np.clip(float(x), 0.0, 1.0))
    try:
        return 1.0 if bool(x) else 0.0
    except Exception:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--controller", type=str, required=True, choices=["openloop", "pid", "agentic"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    cfg.setdefault("sim", {})
    cfg["sim"]["seed"] = int(args.seed)

    handles = make_env(cfg)
    env = handles.env
    dt_sim = float(handles.dt_sim)
    dt_ctrl = float(handles.dt_ctrl)
    cfg["sim"]["ctrl_hz"] = int(round(1.0 / dt_ctrl))

    sim_cfg = cfg.get("sim", {}) or {}
    n = int(sim_cfg.get("num_drones", 1))
    duration = float(sim_cfg["duration_s"])
    steps = int(round(duration / dt_sim))

    offs = formation_offsets(cfg, n)
    dist = DisturbanceModel(cfg, seed=int(args.seed))

    drone_model = env.DRONE_MODEL

    openloop = OpenLoopController(cfg, drone_model)
    pid = PIDController(cfg, drone_model)
    agentic = PIDAgenticController(cfg, drone_model) if args.controller == "agentic" else None

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

    agentic_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
    lookahead_s = float(agentic_cfg.get("lookahead_s", 0.5))

    for k in tqdm(range(steps)):
        t = k * dt_sim
        dist.apply_wind(env, t)

        if (k % ctrl_decim) != 0:
            obs, *_ = env.step(action)
            continue

        # Sensing (truth vs GPS)
        obs_meas = [np.array(o, dtype=float).copy() for o in obs]
        for i in range(n):
            true_pos = obs[i][0:3]
            obs_meas[i][0:3] = dist.gps_measurement(true_pos=true_pos, t=t, dt=dt_sim)

        sense_cfg = cfg.get("controllers", {}).get("sensing", {}) or {}
        mode = sense_cfg.get(args.controller, "gps").lower()
        obs_ctrl = obs_meas if mode == "gps" else obs

        # Nominal trajectory (now + lookahead)
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

        p_des_used = np.zeros((n, 3), dtype=float)

        agentic_active_step = 0.0
        agentic_ref_shift_step = 0.0

        sat_fracs: list[float] = []

        # -------------------------------
        # Control
        # -------------------------------
        if args.controller == "openloop":
            for i in range(n):
                p_des_used[i] = p_des_now[i]
                action[i] = openloop.compute_rpms(obs_ctrl[i], p_des_used[i], v_des_now[i], yaw_des=0.0)
            # sat_fracs stays empty -> NaN

        elif args.controller == "pid":
            for i in range(n):
                p_des_used[i] = p_des_now[i]
                action[i] = pid.compute_rpms(
                    obs_ctrl[i],
                    target_pos=p_des_used[i],
                    target_vel=v_des_now[i],
                    integrate_z=False,
                )
                # IMPORTANT: log per-drone sat_frac from PID baseline
                sat_fracs.append(_sat_to_frac(getattr(pid, "sat_frac", float("nan"))))

        else:
            assert agentic is not None
            applied = []
            shifts = []

            for i in range(n):
                (
                    rpms,
                    err,
                    sat,  # graded fraction (from pid_agentic)
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

                sat_fracs.append(_sat_to_frac(sat))

            agentic_active_step = float(np.mean(applied)) if applied else 0.0
            agentic_ref_shift_step = float(np.mean(shifts)) if shifts else 0.0

        # -------------------------------
        # Metrics (truth-based)
        # -------------------------------
        pos_true = np.stack([obs[i][0:3] for i in range(n)])

        err_nom_xy = np.linalg.norm((pos_true - p_des_now)[:, :2], axis=1)
        err_cmd_xy = np.linalg.norm((pos_true - p_des_used)[:, :2], axis=1)

        mean_err_nom = float(np.mean(err_nom_xy))
        max_err_nom = float(np.max(err_nom_xy))
        mean_err_cmd = float(np.mean(err_cmd_xy))
        max_err_cmd = float(np.max(err_cmd_xy))

        sat_frac_step = float(np.nanmean(sat_fracs)) if len(sat_fracs) else float("nan")

        rows.append(
            {
                "t": t,

                # Backward compatible columns (NOW explicitly NOMINAL)
                "mean_err_m": mean_err_nom,
                "max_err_m": max_err_nom,

                # Explicit, paper-safe columns
                "mean_err_nominal_m": mean_err_nom,
                "max_err_nominal_m": max_err_nom,
                "mean_err_cmd_m": mean_err_cmd,
                "max_err_cmd_m": max_err_cmd,

                "formation_err_rel": float(formation_error_relative(pos_true, offs)),
                "connectivity_rate": _safe_connectivity(pos_true, comm_range),

                "agentic_active": float(agentic_active_step),
                "agentic_ref_shift": float(agentic_ref_shift_step),

                "sat_frac": sat_frac_step,
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