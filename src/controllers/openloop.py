from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class OpenLoopController:
    """
    Open-loop (feedforward) baseline.

    IMPORTANT:
    - This controller MUST NOT use measured state feedback.
    - It consumes only the reference trajectory (target_pos, target_vel, yaw_des)
      and produces motor RPMs.

    Implementation detail:
    - DSLPIDControl is inherently a closed-loop controller, but we use it here only
      as a motor-command mapping utility by feeding it a *synthetic* "current state"
      derived purely from the reference:
          cur_pos  = target_pos
          cur_vel  = target_vel
          cur_quat = identity (assume level)
          cur_ang_vel = 0
    - This prevents any real feedback correction while keeping the RPM interface.
    """

    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        self.ctrl = DSLPIDControl(drone_model=drone_model)

        self._ref_queue = []

    def reset(self):
        self._ref_queue = []

    def _delay_steps(self) -> int:
        ol_cfg = (self.cfg.get("controllers", {}) or {}).get("openloop", {}) or {}
        delay_s = float(ol_cfg.get("openloop_delay_s", 0.0))
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])
        return max(0, int(round(delay_s / max(dt, 1e-9))))

    def compute_rpms(self, state, target_pos, target_vel, yaw_des: float = 0.0):
        # dt comes from config only (not from measured timing)
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])

        # Optional open-loop command delay (more realistic baseline)
        self._ref_queue.append((np.asarray(target_pos, dtype=float), np.asarray(target_vel, dtype=float)))
        delay = self._delay_steps()
        if delay > 0 and len(self._ref_queue) > delay:
            target_pos, target_vel = self._ref_queue.pop(0)
        else:
            target_pos = np.asarray(target_pos, dtype=float)
            target_vel = np.asarray(target_vel, dtype=float)

        # --- OPEN-LOOP: no outer-loop correction; use measured state for stabilization ---
        cur_pos = np.asarray(state[0:3], dtype=float)
        cur_quat = np.asarray(state[3:7], dtype=float)
        cur_vel = np.asarray(state[10:13], dtype=float)
        cur_ang_vel = np.asarray(state[13:16], dtype=float)

        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=np.asarray(target_pos, dtype=float),
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=np.asarray(target_vel, dtype=float),
        )
        return rpms
