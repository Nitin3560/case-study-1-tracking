# src/run_one.py

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
from .trajectories import p_ref, waypoint_by_time, formation_offsets
from .metrics import connectivity_rate


def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _safe_connectivity(pos: np.ndarray, comm_range: float) -> float:
    """
    Be robust to either:
      connectivity_rate(pos, comm_range=10.0)
    or:
      connectivity_rate(pos, comm_range_m=10.0, use_3d=False)
    depending on your metrics.py version.
    """
    try:
        return float(connectivity_rate(pos, comm_range=comm_range))
    except TypeError:
        return float(connectivity_rate(pos, comm_range_m=comm_range, use_3d=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--controller", type=str, required=True, choices=["openloop", "pid", "agentic"])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    cfg.setdefault("sim", {})
    cfg["sim"]["seed"] = int(args.seed)

    # ---- Env ----
    handles = make_env(cfg)
    env = handles.env
    dt_sim = float(handles.dt_sim)
    dt_ctrl = float(handles.dt_ctrl)

    n = int(cfg["sim"]["num_drones"])

    # ---- Duration/steps ----
    sim_cfg = cfg.get("sim", {}) or {}
    if "duration_s" in sim_cfg and sim_cfg["duration_s"] is not None:
        duration = float(sim_cfg["duration_s"])
    elif "steps" in sim_cfg and sim_cfg["steps"] is not None:
        duration = float(sim_cfg["steps"]) / float(sim_cfg["ctrl_hz"])
    else:
        raise KeyError("Config must specify either sim.duration_s or sim.steps")

    steps = int(round(duration / dt_sim))

    # ---- Formation offsets (n,3) ----
    offs = formation_offsets(cfg, n)

    # ---- Disturbances ----
    dist = DisturbanceModel(cfg, seed=int(args.seed))

    # ---- Controller construction ----
    from .controllers.pid_tracker import PIDTracker
    from .controllers.agentic_replanner import AgenticReplanner
    from .controllers.openloop_ff import OpenLoopFeedforwardFollower

    drone_model = env.DRONE_MODEL  # gym-pybullet-drones enum

    if args.controller == "openloop":
        controller = OpenLoopFeedforwardFollower(cfg, drone_model=drone_model)
    elif args.controller == "pid":
        controller = PIDTracker(cfg, drone_model=drone_model)
    elif args.controller == "agentic":
        controller = AgenticReplanner(cfg, drone_model=drone_model)
    else:
        raise ValueError(f"Unknown controller {args.controller}")

    # ---- Reset ----
    obs, _ = env.reset(seed=int(args.seed))
    if hasattr(controller, "reset"):
        controller.reset()

    # ---- Logs ----
    rows = []
    comm_range = float(cfg.get("sim", {}).get("comm_range_m", 10.0))

    # Action shape (n,4)
    action = np.zeros((n, 4), dtype=float)

    # ---- Main loop ----
    ctrl_decim = int(round(dt_ctrl / dt_sim))
    ctrl_decim = max(ctrl_decim, 1)

    for k in tqdm(range(steps)):
        t = k * dt_sim

        # Apply disturbances (wind force in world frame)
        dist.apply_wind(env, t)

        # Control at ctrl_hz, physics at sim_hz
        if (k % ctrl_decim) == 0:

            # --- Build measured observations for controllers (GPS drift + outage) ---
            obs_meas = [np.array(o, dtype=float).copy() for o in obs]
            for i in range(n):
                true_pos = np.array(obs[i][0:3], dtype=float)
                meas_pos = dist.gps_measurement(true_pos=true_pos, t=t, dt=dt_sim)
                obs_meas[i][0:3] = meas_pos
                sense_cfg = (cfg.get("controllers", {}) or {}).get("sensing", {}) or {}
                mode = str(sense_cfg.get(args.controller, "gps")).lower()
                obs_ctrl = obs_meas if mode == "gps" else obs  # truth if not gps

            # ----------------------------------------------------------------------

            # Reference trajectory (center)
            p_traj_now, v_traj_now = p_ref(cfg, t)

            # Lookahead for agentic
            agentic_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
            lookahead_s = float(agentic_cfg.get("lookahead_s", 0.0))
            p_traj_ahead, v_traj_ahead = p_ref(cfg, t + lookahead_s)

            # Per-drone desired references (FORMATION-CENTRIC)
            p_des_all = np.zeros((n, 3), dtype=float)
            v_des_all = np.zeros((n, 3), dtype=float)
            p_ahead_all = np.zeros((n, 3), dtype=float)
            v_ahead_all = np.zeros((n, 3), dtype=float)

            for i in range(n):
                p_des_all[i] = p_traj_now + offs[i]
                v_des_all[i] = v_traj_now
                p_ahead_all[i] = p_traj_ahead + offs[i]
                v_ahead_all[i] = v_traj_ahead

            # -------- Agentic-only centroid correction (computed from MEASURED positions) --------
            p_des_all_agentic = p_des_all.copy()
            p_ahead_all_agentic = p_ahead_all.copy()

            centroid_k = float(agentic_cfg.get("centroid_k", 0.0))  # 0.0 disables
            if centroid_k != 0.0:
                pos_now_meas = np.stack([obs_meas[i][0:3] for i in range(n)], axis=0)  # (n,3)
                p_com_meas = pos_now_meas.mean(axis=0)
                e_com = p_com_meas - p_traj_now
                e_com_corr = np.array([e_com[0], e_com[1], 0.0], dtype=float)

                p_des_all_agentic = p_des_all - centroid_k * e_com_corr
                p_ahead_all_agentic = p_ahead_all - centroid_k * e_com_corr
            # -------------------------------------------------------------------------------

            # ---- Controller actions (ALL use obs_meas) ----
            if args.controller == "openloop":
                open_cfg = (cfg.get("controllers", {}) or {}).get("openloop", {}) or {}
                waypoint_dt_s = float(open_cfg.get("waypoint_dt_s", 0.5))

                p_wp_center = waypoint_by_time(cfg, t)
                p_wp_next_center = waypoint_by_time(cfg, t + waypoint_dt_s)

                for i in range(n):
                    state_i = obs_ctrl[i]
                    action[i, :] = controller.compute_rpms(state_i, p_des_all[i], v_des_all[i])

            elif args.controller == "pid":
                for i in range(n):
                    state_i = obs_ctrl[i]
                    action[i, :] = controller.compute_rpms(state_i, p_des_all[i], v_des_all[i])

            elif args.controller == "agentic":
                for i in range(n):
                    state_i = obs_ctrl[i]
                    rpms, _err = controller.compute_rpms(
                        state_i,
                        p_des_all_agentic[i],
                        v_des_all[i],
                        p_ahead_all_agentic[i],
                        v_ahead_all[i],
                    )
                    action[i, :] = rpms

            # ---- Metrics computed at control rate on TRUE states (obs) ----
            pos_true = np.stack([obs[i][0:3] for i in range(n)], axis=0)  # (n,3)

            desired = p_des_all  # NOMINAL desired (not agentic-corrected)
            track_errs = np.linalg.norm((pos_true - desired)[:, :2], axis=1)

            com_true = pos_true.mean(axis=0)
            rel_true = pos_true - com_true
            form_errs = np.linalg.norm((rel_true - offs)[:, :2], axis=1)

            com_err_m = float(np.linalg.norm((com_true - p_traj_now)[:2]))

            rows.append(
                {
                    "t": float(t),
                    "mean_err_m": float(np.mean(track_errs)),
                    "max_err_m": float(np.max(track_errs)),
                    "formation_err_m": float(np.mean(form_errs)),
                    "com_err_m": com_err_m,
                    "connectivity_rate": _safe_connectivity(pos_true, comm_range),
                }
            )

        # Step physics at sim rate
        obs, _, _, _, _ = env.step(action)

    # ---- Save ----
    os.makedirs("outputs/csv", exist_ok=True)
    out_path = f"outputs/csv/{args.controller}_seed{int(args.seed)}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
