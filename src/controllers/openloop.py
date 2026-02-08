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

        # Fixed "no-feedback" synthetic attitude/state (does not use measurements)
        self._quat_identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self._zeros3 = np.zeros(3, dtype=float)

    def reset(self):
        pass

    def compute_rpms(self, state, target_pos, target_vel, yaw_des: float = 0.0):
        # dt comes from config only (not from measured timing)
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])

        # --- OPEN-LOOP: do not read `state` at all ---
        cur_pos = np.asarray(target_pos, dtype=float)
        cur_vel = np.asarray(target_vel, dtype=float)
        cur_quat = self._quat_identity
        cur_ang_vel = self._zeros3

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