import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


def _split_state(state: np.ndarray):
    state = np.asarray(state, dtype=float).reshape(-1)
    if state.shape[0] < 16:
        raise ValueError(f"Expected state dim >= 16, got {state.shape[0]}")
    return state[0:3], state[3:7], state[10:13], state[13:16]


class PIDTracker:
    """
    Inner stabilizer:
      - Outer-loop PID on position generates v_cmd
      - DSLPIDControl tracks (target_pos, v_cmd)

    Exposes integrator supervision hooks for agentic supervisor.
    """

    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        self.ctrl = DSLPIDControl(drone_model=drone_model)

        pid_cfg = (cfg.get("controllers", {}) or {}).get("pid", {}) or {}
        self.kp = float(pid_cfg.get("kp", 1.0))
        self.kd = float(pid_cfg.get("kd", 0.6))
        self.ki = float(pid_cfg.get("ki", 0.0))
        self.v_max = float(pid_cfg.get("v_max", 3.0))

        self._e_int = np.zeros(3, dtype=float)

    # ---------------------------
    # Integrator supervision API
    # ---------------------------
    def reset(self):
        self._e_int[:] = 0.0

    def get_integral(self) -> np.ndarray:
        return self._e_int.copy()

    def get_integral_xy(self) -> np.ndarray:
        return self._e_int[:2].copy()

    def set_integral(self, e_int_xyz: np.ndarray) -> None:
        v = np.asarray(e_int_xyz, dtype=float).reshape(3)
        self._e_int[:] = v

    def set_integral_xy(self, e_int_xy: np.ndarray) -> None:
        v = np.asarray(e_int_xy, dtype=float).reshape(2)
        self._e_int[0] = float(v[0])
        self._e_int[1] = float(v[1])

    def leak_integral(self, beta: float) -> None:
        self._e_int *= float(beta)

    def leak_integral_xy(self, beta: float) -> None:
        b = float(beta)
        self._e_int[0] *= b
        self._e_int[1] *= b

    def clamp_integral_xy(self, clamp: float) -> None:
        c = float(abs(clamp))
        self._e_int[0] = float(np.clip(self._e_int[0], -c, c))
        self._e_int[1] = float(np.clip(self._e_int[1], -c, c))

    # ---------------------------
    # Main control
    # ---------------------------
    def compute_rpms(
        self,
        state: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray | None = None,
        *,
        integrate: bool = True,
        integrate_z: bool = True,
        int_leak: float = 1.0,
        int_clamp: float | None = None,
    ) -> np.ndarray:
        cur_pos, cur_quat, cur_vel, cur_ang_vel = _split_state(state)

        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])
        target_pos = np.asarray(target_pos, dtype=float)
        v_ref = np.zeros(3, dtype=float) if target_vel is None else np.asarray(target_vel, dtype=float)

        # Errors
        e = target_pos - cur_pos
        e_dot = v_ref - cur_vel

        # Saturation check using current integrator (pre-update)
        v_cmd_raw = self.kp * e + self.ki * self._e_int + self.kd * e_dot
        saturated = float(np.linalg.norm(v_cmd_raw)) > self.v_max

        # Leak first
        if int_leak != 1.0:
            self.leak_integral(float(int_leak))

        # Integrate only if allowed and NOT saturated
        if integrate and (self.ki != 0.0) and (not saturated):
            if integrate_z:
                self._e_int += e * dt
            else:
                self._e_int[0:2] += e[0:2] * dt
                # keep Z integral untouched (or you can decay it)
                # self._e_int[2] *= 1.0

        # Optional clamp on XY only
        if int_clamp is not None:
            self.clamp_integral_xy(float(int_clamp))

        # Final command
        v_cmd = self.kp * e + self.ki * self._e_int + self.kd * e_dot

        sp = float(np.linalg.norm(v_cmd))
        if sp > self.v_max:
            v_cmd *= self.v_max / (sp + 1e-9)

        yaw_des = 0.0
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
        return rpms
