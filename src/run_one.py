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
from .trajectories import p_ref, formation_offsets
from .metrics import connectivity_rate, formation_error_relative

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
    import argparse
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

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

    sim_cfg = cfg.get("sim", {}) or {}
    n = int(sim_cfg.get("num_drones", 1))

    # ---- Duration/steps ----
    if sim_cfg.get("duration_s", None) is not None:
        duration = float(sim_cfg["duration_s"])
    elif sim_cfg.get("steps", None) is not None:
        duration = float(sim_cfg["steps"]) / float(sim_cfg["ctrl_hz"])
    else:
        raise KeyError("Config must specify either sim.duration_s or sim.steps")

    steps = int(round(duration / dt_sim))

    # ---- Formation offsets ----
    offs = formation_offsets(cfg, n)

    # ---- Disturbances ----
    dist = DisturbanceModel(cfg, seed=int(args.seed))

    # ---- Controllers ----
    from .controllers.pid_tracker import PIDTracker
    from .controllers.agentic_replanner import AgenticReplanner
    from .controllers.openloop_ff import OpenLoopFeedforwardFollower

    drone_model = env.DRONE_MODEL

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
    rows: list[dict] = []
    e_com_int = np.zeros(3, dtype=float)
    comm_range = float(sim_cfg.get("comm_range_m", 10.0))

    # Action shape (n,4)
    action = np.zeros((n, 4), dtype=float)

    # Control decimation
    ctrl_decim = int(round(dt_ctrl / dt_sim))
    ctrl_decim = max(ctrl_decim, 1)

    # Sensing mode (single source of truth)
    sense_cfg = (cfg.get("controllers", {}) or {}).get("sensing", {}) or {}
    mode = str(sense_cfg.get(args.controller, "gps")).lower()
    printed_sense = False

    # ---- Main loop ----
    for k in tqdm(range(steps)):
        t = k * dt_sim

        # Apply disturbances
        dist.apply_wind(env, t)

        # Physics at sim rate, control at ctrl_hz (hold last action)
        if (k % ctrl_decim) != 0:
            obs, _, _, _, _ = env.step(action)
            continue

        # ---------- Build measured observations (GPS drift/outage) ----------
        obs_meas = [np.array(o, dtype=float).copy() for o in obs]
        for i in range(n):
            true_pos = np.array(obs[i][0:3], dtype=float)
            meas_pos = dist.gps_measurement(true_pos=true_pos, t=t, dt=dt_sim)
            obs_meas[i][0:3] = meas_pos

        obs_ctrl = obs_meas if mode == "gps" else obs
        if not printed_sense:
            print("SENSING MODE:", mode)
            printed_sense = True

        # ---------- Reference trajectory ----------
        p_traj_now, v_traj_now = p_ref(cfg, t)

        agentic_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
        lookahead_s = float(agentic_cfg.get("lookahead_s", 0.0))
        p_traj_ahead, v_traj_ahead = p_ref(cfg, t + lookahead_s)

        # ---------- Per-drone desired references (nominal) ----------
        p_des_all = np.zeros((n, 3), dtype=float)
        v_des_all = np.zeros((n, 3), dtype=float)
        p_ahead_all = np.zeros((n, 3), dtype=float)
        v_ahead_all = np.zeros((n, 3), dtype=float)

        for i in range(n):
            p_des_all[i] = p_traj_now + offs[i]
            v_des_all[i] = v_traj_now
            p_ahead_all[i] = p_traj_ahead + offs[i]
            v_ahead_all[i] = v_traj_ahead

        # ---------- Optional centroid correction (GLOBAL, TRUTH-based) ----------
        # Always define these (so later code always works)
        p_des_all_agentic = p_des_all.copy()
        p_ahead_all_agentic = p_ahead_all.copy()

        centroid_k = float(agentic_cfg.get("centroid_k", 0.0))
        centroid_ki = float(agentic_cfg.get("centroid_ki", 0.0))
        centroid_int_clamp = float(agentic_cfg.get("centroid_int_clamp", 10.0))

        centroid_integrate_ok = False
        v_com = np.zeros(3, dtype=float)

        if args.controller == "agentic" and ((centroid_k != 0.0) or (centroid_ki != 0.0)):
            pos_now_true = np.stack([obs[i][0:3] for i in range(n)], axis=0)
            p_com = pos_now_true.mean(axis=0)

            e_com = p_com - p_traj_now
            e_com[2] = 0.0

            if centroid_ki > 0.0:
                int_gate_err = float(agentic_cfg.get("centroid_int_gate_err_m", 8.0))
                centroid_integrate_ok = float(np.linalg.norm(e_com[:2])) < int_gate_err

                int_gate_v = float(agentic_cfg.get("centroid_int_gate_v_mps", 2.5))
                v_now_true = np.stack([obs[i][10:13] for i in range(n)], axis=0)
                v_com = v_now_true.mean(axis=0)
                centroid_integrate_ok = centroid_integrate_ok and (float(np.linalg.norm(v_com[:2])) < int_gate_v)

                leak = float(agentic_cfg.get("centroid_int_leak", 0.995))
                e_com_int[:2] *= leak
                if centroid_integrate_ok:
                    e_com_int[:2] += e_com[:2] * dt_ctrl

                e_com_int[0] = float(np.clip(e_com_int[0], -centroid_int_clamp, centroid_int_clamp))
                e_com_int[1] = float(np.clip(e_com_int[1], -centroid_int_clamp, centroid_int_clamp))
                e_com_int[2] = 0.0

            corr = centroid_k * e_com + centroid_ki * e_com_int
            p_des_all_agentic = p_des_all - corr
            p_ahead_all_agentic = p_ahead_all - corr

        # ---------------- Choose desired references for this controller ----------------
        p_des_nom = p_des_all
        p_ahead_nom = p_ahead_all

        p_des_used = p_des_all_agentic if args.controller == "agentic" else p_des_all
        p_ahead_used = p_ahead_all_agentic if args.controller == "agentic" else p_ahead_all

        # ---------------- Controller actions ----------------
        agentic_sat_flags = None
        agentic_gate_flags = None
        agentic_shift_norms = None

        if args.controller == "openloop":
            for i in range(n):
                state_i = obs_ctrl[i]
                action[i, :] = controller.compute_rpms(
                    state=state_i,
                    p_ref=p_des_used[i],
                    v_ref=v_des_all[i],
                    yaw_des=0.0,
                )

        elif args.controller == "pid":
            for i in range(n):
                state_i = obs_ctrl[i]
                action[i, :] = controller.compute_rpms(
                    state=state_i,
                    target_pos=p_des_used[i],
                    target_vel=v_des_all[i],
                )

        elif args.controller == "agentic":
            # agentic returns (rpms, err) OR (rpms, err, sat, gate_ok, shift_norm)
            agentic_sat_flags = np.zeros(n, dtype=bool)
            agentic_gate_flags = np.zeros(n, dtype=bool)
            agentic_shift_norms = np.zeros(n, dtype=float)

            for i in range(n):
                state_i = obs_ctrl[i]
                out = controller.compute_rpms(
                    state=state_i,
                    p_ref_now=p_des_used[i],
                    v_ref_now=v_des_all[i],
                    p_ref_ahead=p_ahead_used[i],
                    v_ref_ahead=v_ahead_all[i],
                    yaw_des=0.0,
                )

                rpms = out[0]
                action[i, :] = rpms

                if len(out) >= 5:
                    agentic_sat_flags[i] = bool(out[2])
                    agentic_gate_flags[i] = bool(out[3])
                    agentic_shift_norms[i] = float(out[4])

        # ---------------- Metrics computed at control rate on TRUE states ----------------
        pos_true = np.stack([obs[i][0:3] for i in range(n)], axis=0)

        track_errs = np.linalg.norm((pos_true - p_des_used)[:, :2], axis=1)

        com_true = pos_true.mean(axis=0)
        com_des_used = p_des_used.mean(axis=0)
        com_err_m = float(np.linalg.norm((com_true - com_des_used)[:2]))

        rel_true = pos_true - com_true
        form_errs = np.linalg.norm((rel_true - offs)[:, :2], axis=1)
        formation_err_rel = formation_error_relative(pos_true, offs)

        # ---------------- Proof diagnostics: how much agentic shifts the reference ----------------
        com_des_nom = p_des_nom.mean(axis=0)
        com_shift = (com_des_used - com_des_nom)[:2]
        com_shift_norm = float(np.sqrt(com_shift[0] ** 2 + com_shift[1] ** 2))

        sat_frac = float(agentic_sat_flags.mean()) if agentic_sat_flags is not None else 0.0
        gate_frac = float(agentic_gate_flags.mean()) if agentic_gate_flags is not None else 0.0
        shift_mean = float(agentic_shift_norms.mean()) if agentic_shift_norms is not None else 0.0

        # ---------------- Log row (dict then append) ----------------
        row = {
            "t": float(t),
            "mean_err_m": float(np.mean(track_errs)),
            "max_err_m": float(np.max(track_errs)),
            "formation_err_m": float(np.mean(form_errs)),
            "formation_err_rel": float(formation_err_rel),
            "com_err_m": float(com_err_m),
            "connectivity_rate": _safe_connectivity(pos_true, comm_range),

            "com_true_x": float(com_true[0]),
            "com_true_y": float(com_true[1]),
            "com_des_nom_x": float(com_des_nom[0]),
            "com_des_nom_y": float(com_des_nom[1]),
            "com_des_used_x": float(com_des_used[0]),
            "com_des_used_y": float(com_des_used[1]),
            "com_shift_norm": float(com_shift_norm),

            "agentic_sat_frac": float(sat_frac),
            "agentic_gate_frac": float(gate_frac),
            "agentic_shift_norm_mean": float(shift_mean),

            # Optional: centroid debug (useful if enabled)
            "centroid_integrate_ok": float(1.0 if centroid_integrate_ok else 0.0),
            "centroid_e_int_x": float(e_com_int[0]),
            "centroid_e_int_y": float(e_com_int[1]),
            "centroid_v_com_x": float(v_com[0]),
            "centroid_v_com_y": float(v_com[1]),
        }
        rows.append(row)

        # Step physics
        obs, _, _, _, _ = env.step(action)

        if k < 5:
            o0 = obs[0]
            print("obs_len:", len(o0))
            print("pos [0:3]:", o0[0:3])
            print("quat[3:7]:", o0[3:7])
            print("slice [7:10]:", o0[7:10])
            print("slice [10:13]:", o0[10:13])
            print("-" * 60)

    # ---- Save ----
    os.makedirs("outputs/csv", exist_ok=True)
    out_path = f"outputs/csv/{args.controller}_seed{int(args.seed)}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
