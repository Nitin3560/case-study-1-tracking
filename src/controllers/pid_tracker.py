import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

def _split_state(state: np.ndarray):
    state = np.asarray(state).reshape(-1)
    if state.shape[0] < 16:
        raise ValueError(f"Expected state dim >= 16, got {state.shape[0]}")
    return state[0:3], state[3:7], state[10:13], state[13:16]

class PIDTracker:
    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        self.ctrl = DSLPIDControl(drone_model=drone_model)

        # Optional: outer-loop gains (if you want), otherwise keep it simple
        pid_cfg = cfg.get("controllers", {}).get("pid", {})
        self.kp = float(pid_cfg.get("kp", 1.0))
        self.kd = float(pid_cfg.get("kd", 0.6))
        self.ki = float(pid_cfg.get("ki", 0.0))
        self._e_int = np.zeros(3)

    def reset(self):
        self._e_int[:] = 0.0

    def compute_rpms(self, state: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray | None = None) -> np.ndarray:
        cur_pos, cur_quat, cur_vel, cur_ang_vel = _split_state(state)

        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])
        target_pos = np.asarray(target_pos, dtype=float)
        v_ref = np.zeros(3) if target_vel is None else np.asarray(target_vel, dtype=float)

        # Outer-loop PID on position -> desired velocity (lightweight, stable)
        e = target_pos - cur_pos
        self._e_int += e * dt
        e_dot = v_ref - cur_vel

        v_cmd = self.kp * e + self.ki * self._e_int + self.kd * e_dot

        # Clip to keep it numerically stable / avoid saturating Crazyflie model
        v_max = float(self.cfg.get("controllers", {}).get("pid", {}).get("v_max", 3.0))
        sp = np.linalg.norm(v_cmd)
        if sp > v_max:
            v_cmd = v_cmd * (v_max / (sp + 1e-9))

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
