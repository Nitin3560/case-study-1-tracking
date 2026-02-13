from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from src.control_core.pid_core import PIDCore, PIDConfig


class PIDController:
    """
    Cascaded closed-loop baseline (clean architecture).

    Outer loop (PIDCore):
      - uses position error (e) and velocity error (v_err)
      - outputs dv_cmd (a velocity correction)

    Inner loop (DSLPIDControl):
      - tracks ONLY velocity reference (target_vel + dv_cmd)
      - position error is intentionally zeroed by setting target_pos = cur_pos

    Saturation (IMPORTANT for fair comparison + anti-windup correctness):
      - computes graded sat_frac in [0,1] from RPM proximity to bounds
      - feeds sat_frac into PIDCore SAME TICK via 2-pass re-step
      - exposes self.sat_frac for logging by run_one.py
    """

    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        pid_cfg = (cfg.get("controllers", {}) or {}).get("pid", {}) or {}

        self.pid = PIDCore(
            PIDConfig(
                kp=float(pid_cfg.get("kp", 1.0)),
                ki=float(pid_cfg.get("ki", 0.0)),
                kd=float(pid_cfg.get("kd", 0.6)),

                v_max=float(pid_cfg.get("v_max", 3.0)),
                axis_mask=tuple(pid_cfg.get("axis_mask", (1.0, 1.0, 0.0))),
                clamp_mode=str(pid_cfg.get("clamp_mode", "xy")),
                v_max_xy=(None if pid_cfg.get("v_max_xy", None) is None else float(pid_cfg["v_max_xy"])),
                v_max_z=(None if pid_cfg.get("v_max_z", None) is None else float(pid_cfg["v_max_z"])),

                int_gate_on_err_m=float(pid_cfg.get("int_gate_on_err_m", 2.0)),
                int_gate_off_err_m=float(pid_cfg.get("int_gate_off_err_m", 3.0)),
                int_gate_on_vel_mps=float(pid_cfg.get("int_gate_on_vel_mps", 4.0)),
                int_gate_off_vel_mps=float(pid_cfg.get("int_gate_off_vel_mps", 5.0)),
                int_leak=float(pid_cfg.get("int_leak", 0.995)),
                int_clamp=float(pid_cfg.get("int_clamp", 2.0)),

                d_mode=str(pid_cfg.get("d_mode", "vel")),
                d_blend=float(pid_cfg.get("d_blend", 0.5)),
                d_tau_s=float(pid_cfg.get("d_tau_s", 0.05)),

                int_sat_hold=bool(pid_cfg.get("int_sat_hold", True)),
                sat_eps=float(pid_cfg.get("sat_eps", 1e-6)),
                int_sat_scale=bool(pid_cfg.get("int_sat_scale", True)),
                int_sat_scale_pow=float(pid_cfg.get("int_sat_scale_pow", 1.0)),
                int_sat_leak=float(pid_cfg.get("int_sat_leak", 1.0)),
            )
        )

        self.ctrl = DSLPIDControl(drone_model=drone_model)

        # --- Saturation detection (graded) ---
        # Prefer sim.rpm_min/rpm_max; allow override in controllers.pid
        self._rpm_min = pid_cfg.get("rpm_min", cfg.get("sim", {}).get("rpm_min", None))
        self._rpm_max = pid_cfg.get("rpm_max", cfg.get("sim", {}).get("rpm_max", None))

        # band defines what "near the limit" means (fraction of rpm range).
        # sat_frac=0 if all motors are >= band away from bounds; sat_frac->1 at bounds.
        self._sat_band_frac = float(pid_cfg.get("sat_band_frac", 0.05))
        self._sat_eps_rpm = float(pid_cfg.get("rpm_eps", 1e-6))

        # Exposed for logging
        self.sat_frac: float = float("nan")

    def reset(self):
        self.pid.reset()
        self.sat_frac = float("nan")
        if hasattr(self.ctrl, "reset"):
            try:
                self.ctrl.reset()
            except TypeError:
                pass

    def _sat_frac_from_rpms(self, rpms: np.ndarray) -> float:
        """
        Graded saturation fraction in [0,1] based on proximity to rpm bounds.
        Returns 0.0 if bounds unknown.
        """
        if self._rpm_min is None or self._rpm_max is None:
            return 0.0

        rmin = float(self._rpm_min)
        rmax = float(self._rpm_max)
        if rmax <= rmin:
            return 0.0

        rpms = np.asarray(rpms, dtype=float)
        rng = rmax - rmin
        band = max(rng * float(self._sat_band_frac), 1e-6)

        margin = np.minimum(rpms - rmin, rmax - rpms)

        # sat=0 when margin>=band ; sat=1 when margin<=0
        sat_each = 1.0 - np.clip((margin - self._sat_eps_rpm) / band, 0.0, 1.0)
        return float(np.clip(np.max(sat_each), 0.0, 1.0))

    def compute_rpms(
        self,
        state,
        target_pos,
        target_vel,
        *,
        integrate_z: bool = False,
        yaw_des: float = 0.0,
    ):
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])

        cur_pos = np.asarray(state[0:3], dtype=float)
        cur_quat = np.asarray(state[3:7], dtype=float)
        cur_vel = np.asarray(state[10:13], dtype=float)
        cur_ang_vel = np.asarray(state[13:16], dtype=float)

        target_pos = np.asarray(target_pos, dtype=float)
        target_vel = np.asarray(target_vel, dtype=float)

        e = target_pos - cur_pos
        v_err = target_vel - cur_vel

        # -----------------------------
        # SAME-TICK sat_frac (2-pass)
        # -----------------------------
        dv_cmd_1 = self.pid.step(e, v_err, dt, integrate_xy=not integrate_z)
        v_cmd_1 = target_vel + np.asarray(dv_cmd_1, dtype=float)

        inner_target_pos = cur_pos  # zero inner position error

        rpms_1, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=inner_target_pos,
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=v_cmd_1,
        )

        sat_frac_1 = self._sat_frac_from_rpms(rpms_1)
        self.pid.set_saturation(sat_frac_1)

        dv_cmd = self.pid.step(e, v_err, dt, integrate_xy=not integrate_z, saturated=sat_frac_1)
        v_cmd = target_vel + np.asarray(dv_cmd, dtype=float)

        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=inner_target_pos,
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=v_cmd,
        )

        self.sat_frac = self._sat_frac_from_rpms(rpms)
        self.pid.set_saturation(self.sat_frac)

        return rpms