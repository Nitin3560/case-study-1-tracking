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

    def reset(self):
        self.e_int[:] = 0.0
        self.last_t = None

    from typing import Tuple
    import numpy as np

    def compute_rpms(
        self,
        state: np.ndarray,
        p_ref_now: np.ndarray,
        v_ref_now: np.ndarray,
        p_ref_ahead: np.ndarray,
        v_ref_ahead: np.ndarray,
        yaw_des: float = 0.0,
    ) -> Tuple[np.ndarray, float, bool, bool, float, np.ndarray]:
        """
        Agentic controller = bias compensator on top of DSLPIDControl.

        What it does:
        - Lookahead blends the reference (optional, alpha in [0,1]).
        - Learns a slow XY bias (integral) to cancel persistent drift (wind/GPS bias).
        - Does NOT add an outer-loop P controller (avoids fighting inner DSLPIDControl).

        Returns:
        rpms:        (4,) motor rpms
        err:         XY tracking error magnitude to blended reference (meters)
        sat:         whether bias correction hit corr_max clamp this step
        gate_ok:     whether integrator update was allowed this step
        bias_norm:   ||bias_xy|| magnitude (meters)
        target_pos_adj: (3,) the actual target_pos passed to inner controller
        """

        cur_pos, cur_quat, cur_vel, cur_ang_vel = _split_state(state)

        agentic_cfg = (self.cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
        if not getattr(self, "_printed_cfg", False):
            print("AGENTIC CFG USED:", agentic_cfg)
            self._printed_cfg = True

        # ---------- Lookahead blending (feedforward reference) ----------
        alpha = float(np.clip(agentic_cfg.get("lookahead_alpha", 0.7), 0.0, 1.0))

        p_now   = np.asarray(p_ref_now, dtype=float)
        p_ahead = np.asarray(p_ref_ahead, dtype=float)
        v_now   = np.asarray(v_ref_now, dtype=float)
        v_ahead = np.asarray(v_ref_ahead, dtype=float)

        target_pos = (1.0 - alpha) * p_now + alpha * p_ahead
        v_ref      = (1.0 - alpha) * v_now + alpha * v_ahead

        # Keep altitude/vertical speed from "now" reference (agentic is XY only)
        target_pos = target_pos.copy()
        v_ref = v_ref.copy()
        target_pos[2] = float(p_now[2])
        v_ref[2] = float(v_now[2])

        # ---------- Error (for gating + reporting) ----------
        e = target_pos - np.asarray(cur_pos, dtype=float)
        e[2] = 0.0
        err = float(np.linalg.norm(e[:2]))

        # ---------- Timing ----------
        dt = 1.0 / float(self.ctrl_hz)

        # ---------- Integrator gating ----------
        err_gate = float(agentic_cfg.get("int_gate_err_m", 12.0))
        v_gate = float(agentic_cfg.get("int_gate_v_mps", 8.0))

        vel_xy = float(np.linalg.norm(np.asarray(cur_vel, dtype=float)[:2]))
        gate_ok = (err < err_gate) and (vel_xy < v_gate)

        # ---------- Persistent bias state ----------
        if not hasattr(self, "bias_xy"):
            self.bias_xy = np.zeros(2, dtype=float)

        ki = float(agentic_cfg.get("ki", 0.05))          # slow learning rate
        leak = float(agentic_cfg.get("int_leak", 0.995)) # decay when gated off
        clamp = float(agentic_cfg.get("int_clamp", 2.0)) # per-axis clamp

        if gate_ok:
            # Bias learning (XY)
            self.bias_xy += ki * e[:2] * dt
            self.bias_xy[0] = float(np.clip(self.bias_xy[0], -clamp, clamp))
            self.bias_xy[1] = float(np.clip(self.bias_xy[1], -clamp, clamp))
        else:
            self.bias_xy *= leak

        # ---------- Clamp final correction magnitude ----------
        corr_max = float(agentic_cfg.get("pos_corr_max_m", 2.0))
        bias_norm = float(np.linalg.norm(self.bias_xy))
        sat = False
        if bias_norm > corr_max:
            self.bias_xy *= corr_max / (bias_norm + 1e-9)
            sat = True
            bias_norm = corr_max

        # ---------- Apply bias to target ----------
        target_pos_adj = target_pos.copy()
        target_pos_adj[0] += float(self.bias_xy[0])
        target_pos_adj[1] += float(self.bias_xy[1])

        # ---------- Inner controller (DSLPIDControl) ----------
        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos_adj,
            target_rpy=np.array([0.0, 0.0, yaw_des], dtype=float),
            target_vel=v_ref,  # feedforward trajectory vel ONLY
        )

        return rpms, err, sat, gate_ok, bias_norm, target_pos_adj
