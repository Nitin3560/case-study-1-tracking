from __future__ import annotations
from typing import Any, Dict
import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


def _split_state(state: np.ndarray):
    s = np.asarray(state).reshape(-1)
    if s.shape[0] < 16:
        raise ValueError(f"Expected state dim >= 16, got {s.shape[0]}")
    return s[0:3], s[3:7], s[10:13], s[13:16]


class OpenLoopFeedforwardFollower:
    """
    Open-loop feedforward baseline:
    - Uses trajectory velocity feedforward
    - Removes XY position feedback by setting XY target_pos to current XY
    - Keeps altitude tracking from reference
    """

    def __init__(self, cfg: Dict[str, Any], drone_model):
        self.cfg = cfg
        self.ctrl = DSLPIDControl(drone_model=drone_model)
        self.ctrl_hz = float(cfg["sim"]["ctrl_hz"])

        ol = (cfg.get("controllers", {}) or {}).get("openloop", {}) or {}
        self.v_max = float(ol.get("v_max", 3.0))

    def compute_rpms(self, state: np.ndarray, p_ref: np.ndarray, v_ref: np.ndarray, yaw_des: float = 0.0) -> np.ndarray:
        cur_pos, cur_quat, cur_vel, cur_ang_vel = _split_state(state)

        p_ref = np.asarray(p_ref, dtype=float)
        v_ref = np.asarray(v_ref, dtype=float)

        v_cmd = np.array([v_ref[0], v_ref[1], 0.0], dtype=float)
        sp = float(np.linalg.norm(v_cmd[:2]))
        if sp > self.v_max:
            v_cmd[:2] *= (self.v_max / (sp + 1e-9))

        target_pos = np.array([cur_pos[0], cur_pos[1], p_ref[2]], dtype=float)

        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=1.0 / self.ctrl_hz,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
            target_rpy=np.array([0.0, 0.0, yaw_des], dtype=float),
            target_vel=v_cmd,
        )
        return rpms
