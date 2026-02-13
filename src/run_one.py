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
from .metrics import (
    connectivity_rate,
    formation_error_relative,
    adjacency_matrix,
    throughput_ratio,
    avg_shortest_path,
    control_overhead_ratio,
)

# Controllers
from .controllers.openloop import OpenLoopController
from .controllers.pid import PIDController
from .controllers.pid_agentic import PIDAgenticController
from .controllers.agentic_supervisor import AgenticSupervisor
from .faults import FaultInjector


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


def _build_demand(n: int, cfg: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    net_cfg = (cfg.get("network", {}) or {})
    demand = np.zeros((n, n), dtype=float)
    pairs = net_cfg.get("traffic_pairs", None)
    if pairs:
        for p in pairs:
            try:
                i, j = int(p[0]), int(p[1])
                w = float(p[2]) if len(p) > 2 else 1.0
                if 0 <= i < n and 0 <= j < n and i != j:
                    demand[i, j] += w
                    demand[j, i] += w
            except Exception:
                continue
        return demand

    # random sparse traffic by default
    num_flows = int(net_cfg.get("num_flows", max(1, n)))
    w_lo, w_hi = net_cfg.get("demand_range", [0.5, 1.5])
    for _ in range(num_flows):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        w = float(rng.uniform(float(w_lo), float(w_hi)))
        demand[i, j] += w
        demand[j, i] += w
    return demand


def _greedy_assignment(cost: np.ndarray) -> list[int]:
    """
    Greedy bipartite assignment. cost is (N,N), returns assignment list of length N.
    """
    n = cost.shape[0]
    assign = [-1] * n
    taken_t = set()
    pairs = []
    for i in range(n):
        for j in range(n):
            pairs.append((cost[i, j], i, j))
    pairs.sort(key=lambda x: x[0])
    for _, i, j in pairs:
        if assign[i] != -1 or j in taken_t:
            continue
        assign[i] = j
        taken_t.add(j)
        if len(taken_t) == n:
            break
    # fill any remaining (degenerate)
    for i in range(n):
        if assign[i] == -1:
            for j in range(n):
                if j not in taken_t:
                    assign[i] = j
                    taken_t.add(j)
                    break
    return assign


def _wind_phase_id(cfg: Dict[str, Any], t: float) -> int:
    dist = cfg.get("disturbance", {}) or {}
    wind = dist.get("wind", {}) or {}
    phases = wind.get("phases", None)
    if not phases:
        return 0
    for idx, ph in enumerate(phases):
        t0 = float(ph.get("t0_s", 0.0))
        t1 = float(ph.get("t1_s", 1e9))
        if t0 <= t < t1:
            return idx
    return len(phases) - 1


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
    fault_rows = []
    comm_range = float(sim_cfg.get("comm_range_m", 10.0))

    agentic_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
    lookahead_s = float(agentic_cfg.get("lookahead_s", 0.5))
    authority_conn_gate = float(agentic_cfg.get("authority_connectivity_gate", 0.6))
    authority_conn_scale = float(agentic_cfg.get("authority_connectivity_scale", 0.6))
    authority_untrusted_scale = float(agentic_cfg.get("authority_untrusted_scale", 0.2))

    rng = np.random.default_rng(int(args.seed))
    demand = _build_demand(n, cfg, rng)

    task_cfg = (cfg.get("tasks", {}) or {})
    reassign_margin_m = float(task_cfg.get("reassign_margin_m", 0.5))
    assign_prev = list(range(n))  # mapping drone -> task index
    metrics_cfg = (cfg.get("metrics", {}) or {})
    task_completion_err_thresh_m = float(metrics_cfg.get("task_completion_err_thresh_m", 2.0))
    utility_lambda_form = float(metrics_cfg.get("utility_lambda_form", 1.0))
    utility_bonus_conn = float(metrics_cfg.get("utility_bonus_conn", 5.0))
    utility_penalty_failed = float(metrics_cfg.get("utility_penalty_failed", 5.0))

    # Agentic supervisor (slow, selective, priority modes)
    supervisor = AgenticSupervisor(cfg) if args.controller == "agentic" else None
    ref_shift = np.zeros(3, dtype=float)
    formation_scale = 1.0
    smooth_alpha = 1.0
    int_hold = False
    supervisor_active = 0.0
    supervisor_mode = 0.0
    last_sat_mean = 0.0
    fault = FaultInjector(cfg.get("faults", {}))

    for k in tqdm(range(steps)):
        t = k * dt_sim
        dist.apply_wind(env, t)
        failed_dist = dist.failed_drones(t, n)
        offline_mask = fault.offline_mask(n, t)
        sensor_corrupt_mask = fault.sensor_corrupt_mask(n, t)
        failed = {i: bool(failed_dist.get(i, False) or offline_mask[i]) for i in range(n)}
        active_mask = np.array([not failed.get(i, False) for i in range(n)], dtype=bool)
        trusted_mask = active_mask & (~sensor_corrupt_mask)
        comm_scale = fault.comm_range_scale(t)
        comm_range_eff = float(comm_range * comm_scale)

        if (k % ctrl_decim) != 0:
            obs, *_ = env.step(action)
            continue

        # Sensing (truth vs GPS)
        obs_meas = [np.array(o, dtype=float).copy() for o in obs]
        for i in range(n):
            true_pos = obs[i][0:3]
            obs_meas[i][0:3] = dist.gps_measurement(true_pos=true_pos, t=t, dt=dt_sim)
            pos_used, vel_used = fault.apply_sensing(i, obs_meas[i][0:3], obs_meas[i][10:13], t)
            obs_meas[i][0:3] = pos_used
            obs_meas[i][10:13] = vel_used

        sense_cfg = cfg.get("controllers", {}).get("sensing", {}) or {}
        mode = sense_cfg.get(args.controller, "gps").lower()
        obs_ctrl = obs_meas if mode == "gps" else obs

        # If supervisor requests integrator hold, apply to agentic PID only
        if args.controller == "agentic" and agentic is not None:
            leak_override = supervisor.integrator_leak_override() if supervisor is not None else None
            agentic.set_integrator_hold(bool(int_hold), leak_override=leak_override)
            trusted_frac = float(np.sum(trusted_mask)) / float(max(1, np.sum(active_mask)))
            authority_scale = 1.0
            if trusted_frac < 1.0:
                authority_scale *= authority_untrusted_scale
            if np.any(active_mask):
                pos_active = np.stack([obs_ctrl[j][0:3] for j in range(n) if active_mask[j]], axis=0)
                conn_active = _safe_connectivity(pos_active, comm_range_eff) if pos_active.shape[0] > 0 else 0.0
                if conn_active < authority_conn_gate:
                    authority_scale *= authority_conn_scale
            agentic.set_authority_scale(authority_scale)

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

        # Apply agentic supervisor adjustments (slow, bounded)
        if args.controller == "agentic":
            v_now_eff = smooth_alpha * v_now
            v_ahead_eff = smooth_alpha * v_ahead
            offs_scaled = offs * formation_scale
            for i in range(n):
                shift_i = ref_shift if trusted_mask[i] else np.zeros(3, dtype=float)
                p_now_eff_i = p_now + shift_i
                p_ahead_eff_i = p_now_eff_i + smooth_alpha * (p_ahead - p_now)
                p_des_now[i] = p_now_eff_i + offs_scaled[i]
                v_des_now[i] = v_now_eff
                p_des_ahead[i] = p_ahead_eff_i + offs_scaled[i]
                v_des_ahead[i] = v_ahead_eff

        # -------------------------------
        # Task allocation (agentic only)
        # -------------------------------
        assign_map = list(range(n))
        task_reassign = 0.0
        if args.controller == "agentic":
            alive = [i for i in range(n) if not failed.get(i, False)]
            if alive:
                pos_meas = np.array([obs_ctrl[i][0:3] for i in alive], dtype=float)
                # cost between alive drones and tasks (all n tasks)
                cost = np.zeros((len(alive), n), dtype=float)
                for ii, i in enumerate(alive):
                    for j in range(n):
                        cost[ii, j] = float(np.linalg.norm((pos_meas[ii] - p_des_now[j])[:2]))

                # greedy assignment to tasks (allow unused tasks)
                pairs = []
                for ii in range(cost.shape[0]):
                    for j in range(cost.shape[1]):
                        pairs.append((cost[ii, j], ii, j))
                pairs.sort(key=lambda x: x[0])
                new_assign_alive = [-1] * len(alive)
                used_tasks = set()
                for c, ii, j in pairs:
                    if new_assign_alive[ii] != -1 or j in used_tasks:
                        continue
                    new_assign_alive[ii] = j
                    used_tasks.add(j)
                    if len(used_tasks) == len(alive):
                        break
                for ii in range(len(alive)):
                    if new_assign_alive[ii] == -1:
                        for j in range(n):
                            if j not in used_tasks:
                                new_assign_alive[ii] = j
                                used_tasks.add(j)
                                break

                # hysteresis: keep previous assignment if improvement is small
                prev_cost = 0.0
                prev_ok = True
                used_prev = set()
                for ii, i in enumerate(alive):
                    t_prev = assign_prev[i] if i < len(assign_prev) else -1
                    if t_prev < 0 or t_prev >= n or t_prev in used_prev:
                        prev_ok = False
                        break
                    used_prev.add(t_prev)
                    prev_cost += cost[ii, t_prev]
                new_cost = sum(cost[ii, new_assign_alive[ii]] for ii in range(len(alive)))
                if prev_ok and prev_cost <= new_cost + reassign_margin_m:
                    # keep previous mapping
                    for ii, i in enumerate(alive):
                        assign_map[i] = assign_prev[i]
                else:
                    for ii, i in enumerate(alive):
                        assign_map[i] = new_assign_alive[ii]
                    task_reassign = 1.0

                assign_prev = assign_map.copy()

        p_des_used = np.zeros((n, 3), dtype=float)

        agentic_active_step = 0.0
        agentic_ref_shift_step = 0.0

        sat_fracs: list[float] = []

        # -------------------------------
        # Control
        # -------------------------------
        if args.controller == "openloop":
            for i in range(n):
                if failed.get(i, False):
                    action[i] = np.zeros(4, dtype=float)
                    continue
                t_idx = assign_map[i]
                p_des_used[i] = p_des_now[t_idx]
                action[i] = openloop.compute_rpms(obs_ctrl[i], p_des_used[i], v_des_now[t_idx], yaw_des=0.0)
            # sat_fracs stays empty -> NaN

        elif args.controller == "pid":
            for i in range(n):
                if failed.get(i, False):
                    action[i] = np.zeros(4, dtype=float)
                    continue
                t_idx = assign_map[i]
                p_des_used[i] = p_des_now[t_idx]
                action[i] = pid.compute_rpms(
                    obs_ctrl[i],
                    target_pos=p_des_used[i],
                    target_vel=v_des_now[t_idx],
                    integrate_z=False,
                )
                # IMPORTANT: log per-drone sat_frac from PID baseline
                sat_fracs.append(_sat_to_frac(getattr(pid, "sat_frac", float("nan"))))

        else:
            assert agentic is not None
            applied = []
            shifts = []
            if np.any(trusted_mask):
                group_pos_active = np.stack([obs_ctrl[j][0:3] for j in range(n) if trusted_mask[j]], axis=0)
            elif np.any(active_mask):
                group_pos_active = np.stack([obs_ctrl[j][0:3] for j in range(n) if active_mask[j]], axis=0)
            else:
                group_pos_active = None

            for i in range(n):
                if failed.get(i, False):
                    action[i] = np.zeros(4, dtype=float)
                    continue
                t_idx = assign_map[i]
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
                    p_ref_now=p_des_now[t_idx],
                    v_ref_now=v_des_now[t_idx],
                    p_ref_ahead=p_des_ahead[t_idx],
                    v_ref_ahead=v_des_ahead[t_idx],
                    yaw_des=0.0,
                    group_pos=group_pos_active,
                    comm_range_m=comm_range_eff,
                )

                action[i] = rpms
                p_des_used[i] = np.asarray(target_pos_adj, dtype=float)

                applied.append(float(apply_bias_float))
                delta_xy = (p_des_used[i] - p_des_now[i])[:2]
                shifts.append(float(np.linalg.norm(delta_xy)))

                sat_fracs.append(_sat_to_frac(sat))

            agentic_active_step = float(np.mean(applied)) if applied else 0.0
            agentic_ref_shift_step = float(np.mean(shifts)) if shifts else 0.0

        # Apply dropout/no-control overrides after controller output
        dropped_agents = []
        sensor_bad_agents = []
        for i in range(n):
            action[i], meta = fault.apply_control(i, action[i], t)
            if meta.get("dropout_active", False):
                dropped_agents.append(i)
            if fault.agent_has_sensor_fault(i, t):
                sensor_bad_agents.append(i)

        # -------------------------------
        # Metrics (truth-based)
        # -------------------------------
        pos_true = np.stack([obs[i][0:3] for i in range(n)])
        alive_mask = np.array([not failed.get(i, False) for i in range(n)])
        alive_idx = np.where(alive_mask)[0]

        p_nom_assigned = np.zeros_like(p_des_now)
        for i in range(n):
            t_idx = assign_map[i]
            p_nom_assigned[i] = p_des_now[t_idx]

        err_nom_xy = np.linalg.norm((pos_true - p_nom_assigned)[:, :2], axis=1)
        err_cmd_xy = np.linalg.norm((pos_true - p_des_used)[:, :2], axis=1)

        if alive_idx.size > 0:
            mean_err_nom = float(np.mean(err_nom_xy[alive_idx]))
            max_err_nom = float(np.max(err_nom_xy[alive_idx]))
            mean_err_cmd = float(np.mean(err_cmd_xy[alive_idx]))
            max_err_cmd = float(np.max(err_cmd_xy[alive_idx]))
            task_completion_proxy = float(np.mean(err_nom_xy[alive_idx] <= task_completion_err_thresh_m))
        else:
            mean_err_nom = float("nan")
            max_err_nom = float("nan")
            mean_err_cmd = float("nan")
            max_err_cmd = float("nan")
            task_completion_proxy = float("nan")

        sat_frac_step = float(np.nanmean(sat_fracs)) if len(sat_fracs) else float("nan")
        if np.isfinite(sat_frac_step):
            last_sat_mean = sat_frac_step

        pos_alive = pos_true[alive_idx] if alive_idx.size > 0 else pos_true
        offs_assigned = np.zeros_like(offs)
        for i in range(n):
            offs_assigned[i] = offs[assign_map[i]]
        if args.controller == "agentic":
            offs_assigned = offs_assigned * formation_scale
        offs_alive = offs_assigned[alive_idx] if alive_idx.size > 0 else offs_assigned

        adj = adjacency_matrix(pos_alive, comm_range_m=comm_range_eff, use_3d=False)
        demand_alive = demand[np.ix_(alive_idx, alive_idx)] if alive_idx.size > 0 else demand
        tp_ratio = throughput_ratio(adj, demand_alive) if alive_idx.size > 1 else 0.0
        lat_proxy = avg_shortest_path(adj) if alive_idx.size > 1 else 0.0
        ctrl_over = control_overhead_ratio(adj) if alive_idx.size > 1 else 0.0
        conn_rate = _safe_connectivity(pos_alive, comm_range_eff)
        form_err_rel = float(formation_error_relative(pos_alive, offs_alive))
        alive_ratio = float(np.mean(alive_mask)) if alive_mask.size > 0 else 0.0
        system_utility = (
            -mean_err_nom
            - utility_lambda_form * form_err_rel
            + utility_bonus_conn * conn_rate
            - utility_penalty_failed * (1.0 - alive_ratio)
        )

        rows.append(
            {
                "t": t,
                "wind_phase": float(_wind_phase_id(cfg, t)),

                # Backward compatible columns (NOW explicitly NOMINAL)
                "mean_err_m": mean_err_nom,
                "max_err_m": max_err_nom,

                # Explicit, paper-safe columns
                "mean_err_nominal_m": mean_err_nom,
                "max_err_nominal_m": max_err_nom,
                "mean_err_cmd_m": mean_err_cmd,
                "max_err_cmd_m": max_err_cmd,

                "formation_err_rel": form_err_rel,
                "connectivity_rate": conn_rate,

                "throughput_ratio": float(tp_ratio),
                "latency_proxy": float(lat_proxy),
                "control_overhead": float(ctrl_over),

                "task_reassign": float(task_reassign),
                "task_cost_mean": float(np.mean(err_nom_xy[alive_idx])) if alive_idx.size > 0 else float("nan"),

                "failed_count": float(np.sum(~alive_mask)),
                "alive_ratio": alive_ratio,
                "task_completion_proxy": task_completion_proxy,
                "system_utility": float(system_utility),
                "fault_comm_scale": float(comm_scale),
                "fault_dropout_count": float(len(dropped_agents)),
                "fault_sensor_count": float(len(sensor_bad_agents)),

                "agentic_active": float(agentic_active_step),
                "agentic_ref_shift": float(agentic_ref_shift_step),
                "supervisor_active": float(supervisor_active),
                "supervisor_mode": float(supervisor_mode),
                "ref_shift_norm": float(np.linalg.norm(ref_shift[:2])),
                "formation_scale": float(formation_scale),
                "smooth_alpha": float(smooth_alpha),
                "int_hold": float(1.0 if int_hold else 0.0),

                "sat_frac": sat_frac_step,
            }
        )

        obs, *_ = env.step(action)

        # -------------------------------
        # Supervisor update (slow layer)
        # -------------------------------
        if args.controller == "agentic" and supervisor is not None and supervisor.should_tick(dt_ctrl):
            pos_sup = pos_true[alive_idx] if alive_idx.size > 0 else pos_true
            offs_sup = offs_assigned[alive_idx] if alive_idx.size > 0 else offs_assigned
            state = supervisor.step(
                dt_ctrl=dt_ctrl,
                pos_true=pos_sup,
                p_ref_now=p_now,
                formation_err=float(formation_error_relative(pos_sup, offs_sup)),
                connectivity_rate=float(_safe_connectivity(pos_sup, comm_range_eff)),
                sat_mean=float(last_sat_mean),
                phase_id=int(_wind_phase_id(cfg, t)),
                comm_scale=float(comm_scale),
                active_mask=active_mask[alive_idx] if alive_idx.size > 0 else active_mask,
                trusted_mask=trusted_mask[alive_idx] if alive_idx.size > 0 else trusted_mask,
                sensor_corrupt_mask=sensor_corrupt_mask[alive_idx] if alive_idx.size > 0 else sensor_corrupt_mask,
            )
            ref_shift = state.ref_shift
            formation_scale = state.formation_scale
            smooth_alpha = state.smooth_alpha
            int_hold = bool(state.int_hold)
            supervisor_active = float(state.supervisor_active)
            supervisor_mode = float(state.supervisor_mode)

        active_faults = ",".join([str(ev.get("type", "")) for ev in fault.active_events(t)])
        fault_rows.append(
            {
                "t_s": float(t),
                "active_faults": active_faults,
                "dropout_agent_id": ";".join(str(x) for x in dropped_agents),
                "sensor_bad_agent_id": ";".join(str(x) for x in sensor_bad_agents),
                "comm_scale": float(comm_scale),
            }
        )

    os.makedirs("outputs/csv", exist_ok=True)
    out_path = f"outputs/csv/{args.controller}_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if fault.enabled:
        os.makedirs("outputs/faults", exist_ok=True)
        fault_path = f"outputs/faults/{args.controller}_seed{args.seed}_faults_timeline.csv"
        pd.DataFrame(fault_rows).to_csv(fault_path, index=False)
        print(f"Saved: {fault_path}")
    print(f"Saved: {out_path}")
    env.close()


if __name__ == "__main__":
    main()
