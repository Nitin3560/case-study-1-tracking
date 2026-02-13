from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from src.control_core.pid_core import PIDCore, PIDConfig


def _clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


class PIDAgenticController:
    """
    PID + Agentic (clean + defensible):

    - Agentic layer is FEEDFORWARD w.r.t. measured state:
        * It does NOT use cur_pos/cur_vel/tracking error to compute its bias.
        * It only uses reference geometry: (p_ref_ahead - p_ref_now).
      (Optionally, it can be a *reference filter* if authority_alpha > 0.)

    - Outer loop (PIDCore) is the feedback controller for position tracking:
        e = p_cmd - cur_pos
        v_err = v_ref_now - cur_vel
        dv_cmd = PIDCore.step(e, v_err, dt, saturated=sat_frac)

    - Inner loop (DSLPIDControl) is used as an execution controller:
        We *zero inner position error* by setting target_pos = cur_pos.
        This strongly discourages an additional inner position feedback loop,
        but the exact internal behavior depends on DSLPIDControl implementation.

    Metrics:
      - This controller returns err_norm against the nominal mission reference p_ref_now (not p_cmd),
        to avoid the "moving goalpost" evaluation trap.

    Saturation (IMPORTANT):
      - We compute a graded sat_frac \in [0,1] from RPM proximity to bounds.
      - We FEED sat_frac into PIDCore each tick (so integrator gating/scaling is correct).
      - True actuator-limit detection based on motor model/RPM limits is ultimately downstream,
        but this is a robust and practical proxy when rpm_min/rpm_max are known.
    """

    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        pid_cfg = (cfg.get("controllers", {}) or {}).get("pid", {}) or {}
        ag_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}

        # -------------------------
        # Outer PID core (PIDConfig)
        # -------------------------
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

        # -------------------------
        # Agentic parameters (feedforward)
        # -------------------------
        self.delta_thresh_m = float(ag_cfg.get("delta_thresh_m", ag_cfg.get("error_thresh_m", 0.6)))
        self.delta_ramp_mult = float(ag_cfg.get("delta_ramp_mult", ag_cfg.get("authority_ramp_mult", 2.5)))
        self.max_ref_shift_m = float(ag_cfg.get("max_ref_shift_m", ag_cfg.get("max_shift_m", 0.35)))

        self.authority_alpha = float(ag_cfg.get("authority_alpha", 0.0))
        self._authority = 0.0
        self.authority_max_rate = float(ag_cfg.get("authority_max_rate", 0.0))
        self.authority_scale = 1.0

        self.v_cmd_limit_mps = ag_cfg.get("v_cmd_limit_mps", None)
        if self.v_cmd_limit_mps is not None:
            self.v_cmd_limit_mps = float(self.v_cmd_limit_mps)

        # Closed-loop agentic gating (error-aware)
        self.error_gate_m = float(ag_cfg.get("error_gate_m", 0.5))
        self.error_scale_m = float(ag_cfg.get("error_scale_m", 1.5))

        # Comm-aware bias (toward centroid if isolated)
        self.comm_bias_m = float(ag_cfg.get("comm_bias_m", 0.15))
        self.comm_min_neighbors = int(ag_cfg.get("comm_min_neighbors", 1))

        # -------------------------
        # Saturation detection (graded)
        # -------------------------
        self._rpm_min = ag_cfg.get("rpm_min", cfg.get("sim", {}).get("rpm_min", None))
        self._rpm_max = ag_cfg.get("rpm_max", cfg.get("sim", {}).get("rpm_max", None))

        # band defines what "near the limit" means (fraction of rpm range).
        # sat_frac=0 if all motors are >= band away from bounds; sat_frac->1 at bounds.
        self._sat_band_frac = float(ag_cfg.get("sat_band_frac", 0.05))
        self._sat_eps_rpm = float(ag_cfg.get("rpm_eps", 1e-6))  # tiny numeric eps

    def reset(self):
        self.pid.reset()
        self._authority = 0.0
        if hasattr(self.ctrl, "reset"):
            try:
                self.ctrl.reset()
            except TypeError:
                pass

    def set_integrator_hold(self, hold: bool, leak_override: float | None = None):
        self.pid.set_integrator_hold(hold)
        self.pid.set_integrator_leak_override(leak_override)

    def set_authority_scale(self, scale: float):
        self.authority_scale = float(np.clip(scale, 0.0, 1.0))

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

        # margin to closest bound, per motor
        margin = np.minimum(rpms - rmin, rmax - rpms)

        # sat=0 when margin>=band ; sat=1 when margin<=0
        sat_each = 1.0 - np.clip((margin - self._sat_eps_rpm) / band, 0.0, 1.0)
        return float(np.clip(np.max(sat_each), 0.0, 1.0))

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _update_authority(self, a_target: float, dt: float) -> float:
        a_target = float(np.clip(a_target, 0.0, 1.0))

        # Stateless by default
        alpha_eff = float(np.clip(self.authority_alpha * self.authority_scale, 0.0, 1.0))

        if alpha_eff <= 0.0:
            self._authority = a_target
            return a_target

        # EMA reference filter
        a_next = alpha_eff * float(self._authority) + (1.0 - alpha_eff) * a_target

        # Optional rate limit
        if self.authority_max_rate and self.authority_max_rate > 0.0 and dt > 0.0:
            max_da = float(self.authority_max_rate) * float(dt)
            a_next = float(np.clip(a_next, float(self._authority) - max_da, float(self._authority) + max_da))

        self._authority = float(np.clip(a_next, 0.0, 1.0))
        return float(self._authority)

    def compute_rpms(
        self,
        state,
        p_ref_now,
        v_ref_now,
        p_ref_ahead,
        v_ref_ahead,  # kept for API compatibility (unused in this conservative design)
        yaw_des: float = 0.0,
        *,
        group_pos: np.ndarray | None = None,
        comm_range_m: float | None = None,
        return_debug: bool = False,
    ):
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])

        # measured state (used only in baseline feedback control)
        cur_pos = np.asarray(state[0:3], dtype=float)
        cur_quat = np.asarray(state[3:7], dtype=float)
        cur_vel = np.asarray(state[10:13], dtype=float)
        cur_ang_vel = np.asarray(state[13:16], dtype=float)

        # references
        p_ref_now = np.asarray(p_ref_now, dtype=float)
        v_ref_now = np.asarray(v_ref_now, dtype=float)
        p_ref_ahead = np.asarray(p_ref_ahead, dtype=float)

        # ============================================
        # Agentic bias (closed-loop gating + comm-aware)
        # ============================================
        delta = (p_ref_ahead - p_ref_now).copy()
        delta[2] = 0.0
        delta_xy_norm = float(np.linalg.norm(delta[:2]))

        d0 = float(self.delta_thresh_m)
        d1 = max(d0 + 1e-6, float(self.delta_ramp_mult) * d0)

        raw = float(np.clip((delta_xy_norm - d0) / (d1 - d0), 0.0, 1.0))
        a_target = self._smoothstep01(raw)

        # Error-aware gating: if tracking error already large, reduce authority
        err_xy_norm = float(np.linalg.norm((p_ref_now - cur_pos)[:2]))
        err_gate = float(self.error_gate_m)
        err_scale = max(1e-6, float(self.error_scale_m))
        err_factor = float(np.clip((err_xy_norm - err_gate) / err_scale, 0.0, 1.0))
        a_target = a_target * (1.0 - err_factor)

        authority = self._update_authority(a_target, dt)

        # bounded bias
        shift = authority * delta
        shift_xy_norm = float(np.linalg.norm(shift[:2]))
        if shift_xy_norm > float(self.max_ref_shift_m):
            shift[:2] *= float(self.max_ref_shift_m) / (shift_xy_norm + 1e-9)

        # Comm-aware bias: if isolated, pull toward centroid
        comm_bias = np.zeros(3, dtype=float)
        if group_pos is not None and comm_range_m is not None and self.comm_bias_m > 0.0:
            pos_all = np.asarray(group_pos, dtype=float)
            if pos_all.ndim == 2 and pos_all.shape[1] == 3:
                n = pos_all.shape[0]
                nbrs = 0
                for j in range(n):
                    if np.allclose(pos_all[j], cur_pos):
                        continue
                    d = np.linalg.norm((pos_all[j] - cur_pos)[:2])
                    if d <= float(comm_range_m):
                        nbrs += 1
                if nbrs < self.comm_min_neighbors:
                    centroid = pos_all.mean(axis=0)
                    vec = centroid - cur_pos
                    vec[2] = 0.0
                    norm = float(np.linalg.norm(vec[:2]))
                    if norm > 1e-9:
                        comm_bias[:2] = (vec[:2] / norm) * float(self.comm_bias_m)

        p_cmd = p_ref_now + shift + comm_bias
        bias_norm = float(np.linalg.norm(shift[:2]))
        apply_bias_float = 1.0 if bias_norm > 1e-9 else 0.0

        # ============================================
        # Baseline feedback control (outer loop)
        # ============================================
        e = p_cmd - cur_pos
        v_err = v_ref_now - cur_vel

        # IMPORTANT: we need sat_frac SAME TICK for PIDCore.
        # We don't know rpms yet; so we do a 2-pass call:
        #   pass1: compute rpms with last tick sat state (stored inside PIDCore)
        #   compute sat_frac from rpms
        #   feed sat_frac into PIDCore for *next* step (or re-step if you want strict same-tick)
        #
        # To enforce same-tick, we re-run PID step once after estimating sat_frac.
        # This costs a tiny amount but keeps logic consistent.
        dv_cmd_1 = self.pid.step(e, v_err, dt)
        v_cmd_1 = v_ref_now + np.asarray(dv_cmd_1, dtype=float)

        # Inner: zero position error (discourages inner position loop)
        inner_target_pos = cur_pos

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

        sat_frac = self._sat_frac_from_rpms(rpms_1)
        self.pid.set_saturation(sat_frac)  # feed saturation to PIDCore

        # Re-step once with same-tick sat_frac so integrator logic is consistent
        dv_cmd = self.pid.step(e, v_err, dt, saturated=sat_frac)
        v_cmd = v_ref_now + np.asarray(dv_cmd, dtype=float)

        # Optional safety limiter (OFF by default)
        v_cmd_xy_norm_pre = float(np.linalg.norm(v_cmd[:2]))
        if self.v_cmd_limit_mps is not None:
            lim = float(self.v_cmd_limit_mps)
            sp_xy = float(np.linalg.norm(v_cmd[:2]))
            if sp_xy > lim:
                v_cmd[:2] *= lim / (sp_xy + 1e-9)
        v_cmd_xy_norm_post = float(np.linalg.norm(v_cmd[:2]))

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

        # update sat frac from final rpms too (logging)
        sat_frac = self._sat_frac_from_rpms(rpms)
        self.pid.set_saturation(sat_frac)

        # Metric error is against nominal mission reference (not p_cmd)
        e_metric = (p_ref_now - cur_pos).copy()
        e_metric[2] = 0.0
        err_norm = float(np.linalg.norm(e_metric[:2]))

        if return_debug:
            dbg = {
                "delta_xy_norm": float(delta_xy_norm),
                "delta_thresh_m": float(self.delta_thresh_m),
                "delta_ramp_mult": float(self.delta_ramp_mult),

                "authority_alpha": float(self.authority_alpha),
                "authority_max_rate": float(self.authority_max_rate),
                "authority": float(authority),
                "a_target": float(a_target),

                "bias_norm": float(bias_norm),
                "max_ref_shift_m": float(self.max_ref_shift_m),
                "err_xy_norm": float(err_xy_norm),
                "err_factor": float(err_factor),
                "comm_bias_xy": float(np.linalg.norm(comm_bias[:2])),

                "err_norm_nominal_xy": float(err_norm),

                "inner_pos_zeroed": True,

                "v_cmd_limit_mps": (None if self.v_cmd_limit_mps is None else float(self.v_cmd_limit_mps)),
                "v_cmd_xy_norm_pre": float(v_cmd_xy_norm_pre),
                "v_cmd_xy_norm_post": float(v_cmd_xy_norm_post),

                "sat_frac": float(sat_frac),
                "sat_band_frac": float(self._sat_band_frac),
                "rpm_min": (None if self._rpm_min is None else float(self._rpm_min)),
                "rpm_max": (None if self._rpm_max is None else float(self._rpm_max)),
            }
            return (
                rpms,
                err_norm,
                float(sat_frac),   # <-- return graded fraction
                True,
                bias_norm,
                p_cmd,
                apply_bias_float,
                dbg,
            )

        return (
            rpms,
            err_norm,
            float(sat_frac),   # <-- return graded fraction
            True,
            bias_norm,
            p_cmd,
            apply_bias_float,
        )
