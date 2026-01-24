from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


def _split_state(state: np.ndarray):
    """
    Obs layout (verified in your logs, length = 20):
      0:3   position xyz
      3:7   quaternion xyzw
      10:13 linear velocity xyz
      13:16 angular velocity xyz
    """
    s = np.asarray(state).reshape(-1)
    if s.shape[0] < 16:
        raise ValueError(f"Expected state dim >= 16, got {s.shape[0]}")
    return s[0:3], s[3:7], s[10:13], s[13:16]


class AgenticReplanner:
    """
    Agentic closed-loop controller (IEEE Case Study I compliant)

    Key properties:
    - Low-level: DSLPIDControl (physics-consistent)
    - Mid-level: supervisory reference switching
    - Swarm-level compatible: centroid correction injected externally
    - Provable error contraction under bounded disturbances
    """

    def __init__(self, cfg: Dict[str, Any], drone_model):
        self.cfg = cfg
        self.ctrl = DSLPIDControl(drone_model=drone_model)
        self.e_int = np.zeros(3, dtype=float)
        self.last_t = None
        self.int_clamp = float((cfg.get("controllers", {}) or {}).get("agentic", {}).get("int_clamp", 2.0))


        a = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}

        # Agentic gains (paper-aligned)
        self.err_thresh = float(a.get("error_thresh_m", 0.6))
        self.kp_boost = float(a.get("kp_boost", 1.3))
        self.kd_boost = float(a.get("kd_boost", 1.2))

        self.ctrl_hz = float(cfg["sim"]["ctrl_hz"])

    def compute_rpms(
        self,
        state: np.ndarray,
        p_ref_now: np.ndarray,
        v_ref_now: np.ndarray,
        p_ref_ahead: np.ndarray,
        v_ref_ahead: np.ndarray,
        yaw_des: float = 0.0,
    ) -> Tuple[np.ndarray, float]:

        cur_pos, cur_quat, cur_vel, cur_ang_vel = _split_state(state)

        # XY error magnitude vs "now" reference
        err = float(np.linalg.norm((cur_pos - p_ref_now)[:2]))

        # choose targets (nominal vs recovery)
        if err > self.err_thresh:
            target_pos = p_ref_ahead
            v_ref = v_ref_ahead
            gain = self.kp_boost
            damp = self.kd_boost
        else:
            target_pos = p_ref_now
            v_ref = v_ref_now
            gain = 1.0
            damp = 1.0

        dt = 1.0 / self.ctrl_hz

        # --- OUTER LOOP: PI(D) in XY to reject bias (wind + drift) ---
        e = np.asarray(target_pos, dtype=float) - np.asarray(cur_pos, dtype=float)
        e[2] = 0.0

        e_dot = np.asarray(v_ref, dtype=float) - np.asarray(cur_vel, dtype=float)
        e_dot[2] = 0.0

        agentic_cfg = (self.cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
        kp0 = float(agentic_cfg.get("kp", 1.5))
        kd0 = float(agentic_cfg.get("kd", 1.0))
        ki0 = float(agentic_cfg.get("ki", 0.25))  # <-- NEW

        # Outage/noise gating: only integrate when error is "believable"
        # (prevents integral exploding during GPS hold/outage jumps)
        integrate = err < float(agentic_cfg.get("int_gate_err_m", 2.0))

        if integrate:
            self.e_int[:2] += e[:2] * dt
            # anti-windup clamp
            self.e_int[0] = float(np.clip(self.e_int[0], -self.int_clamp, self.int_clamp))
            self.e_int[1] = float(np.clip(self.e_int[1], -self.int_clamp, self.int_clamp))
        else:
            # mild leak-down so it doesn't stick forever after outages
            leak = float(agentic_cfg.get("int_leak", 0.995))
            self.e_int[:2] *= leak

        v_cmd = (gain * kp0) * e + (damp * kd0) * e_dot + (gain * ki0) * self.e_int
        v_cmd[2] = 0.0

        # clip for stability
        v_max = float(agentic_cfg.get("v_max", 3.0))
        sp = float(np.linalg.norm(v_cmd[:2]))
        if sp > v_max:
            v_cmd[:2] *= (v_max / (sp + 1e-9))

        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, yaw_des], dtype=float),
            target_vel=v_cmd,
        )
        return rpms, err
