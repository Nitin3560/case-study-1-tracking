from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from .pid_tracker import PIDTracker


def _split_state(state: np.ndarray):
    s = np.asarray(state, dtype=float).reshape(-1)
    if s.shape[0] < 16:
        raise ValueError(f"Expected state dim >= 16, got {s.shape[0]}")
    return s[0:3], s[3:7], s[10:13], s[13:16]


class AgenticReplanner:
    """
    Event-driven supervisory layer on top of PID (diagram-aligned).

    Modes:
      - NORMAL: PID runs normally; bias may be applied (slow updates)
      - PROTECT: integrator supervision ONLY (freeze/leak/reset), no bias updates/application
      - RECOVER: cooldown; bias not updated/applied

    Returns:
      (rpms, err, sat, gate_ok, bias_norm, target_pos_adj, apply_bias_float)
    """

    def __init__(self, cfg: Dict[str, Any], drone_model):
        self.cfg = cfg
        self.inner = PIDTracker(cfg, drone_model)

        a = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
        self._printed_cfg = False

        self.ctrl_hz = float((cfg.get("sim", {}) or {}).get("ctrl_hz", 48.0))
        self.dt = 1.0 / self.ctrl_hz

        # Supervisor cadence
        self.supervisor_period_s = float(a.get("supervisor_period_s", 0.5))
        self.protect_duration_s = float(a.get("protect_duration_s", 1.0))
        self.recover_duration_s = float(a.get("recover_duration_s", 1.0))

        self._sup_steps = max(1, int(round(self.supervisor_period_s * self.ctrl_hz)))
        self._protect_steps = max(1, int(round(self.protect_duration_s * self.ctrl_hz)))
        self._recover_steps = max(1, int(round(self.recover_duration_s * self.ctrl_hz)))

        # Reference lookahead blending
        self.lookahead_alpha = float(np.clip(a.get("lookahead_alpha", 0.7), 0.0, 1.0))

        # Health thresholds
        self.quiet_err_m = float(a.get("quiet_err_m", 0.6))
        self.quiet_vel_mps = float(a.get("quiet_vel_mps", 0.4))

        self.act_err_m = float(a.get("act_err_m", 1.2))
        self.hold_s = float(a.get("hold_s", 0.8))
        self.min_improve_mps = float(a.get("min_improve_mps", 0.05))
        self.err_ema_beta = float(np.clip(a.get("err_ema_beta", 0.05), 0.001, 0.5))

        self.int_gate_v_mps = float(a.get("int_gate_v_mps", 2.0))

        # Windup trigger (cause-based)
        self.int_trigger_norm = float(a.get("int_trigger_norm", 0.8))
        self.int_trigger_growth = float(a.get("int_trigger_growth", 0.002))

        # Bias (secondary, slow)
        self.ki_bias = float(a.get("ki_bias", a.get("ki", 0.01)))
        self.bias_int_clamp = float(a.get("bias_int_clamp", a.get("int_clamp", 1.0)))
        self.pos_corr_max_m = float(a.get("pos_corr_max_m", 1.0))
        self.bias_leak = float(a.get("bias_leak", a.get("int_leak", 0.995)))

        # PID integrator supervision
        self.pid_int_freeze_on_bad = bool(a.get("pid_int_freeze_on_bad", True))
        self.pid_int_leak_good = float(a.get("pid_int_leak_good", 1.0))
        self.pid_int_leak_bad = float(a.get("pid_int_leak_bad", 0.90))
        self.pid_int_clamp = float(a.get("pid_int_clamp", 2.0))

        # Optional hard reset
        self.pid_int_reset_err_m = float(a.get("pid_int_reset_err_m", 0.0))
        self.pid_int_reset_hold_s = float(a.get("pid_int_reset_hold_s", 1.5))
        self._reset_steps = max(1, int(round(self.pid_int_reset_hold_s * self.ctrl_hz)))
        # How long to apply the estimated bias after PROTECT ends
        self.bias_apply_s = float(a.get("bias_apply_s", 1.5))
        self._bias_apply_steps = max(1, int(round(self.bias_apply_s * self.ctrl_hz)))

        # Optional smoothing for estimated bias (0 = no smoothing)
        self.bias_est_alpha = float(np.clip(a.get("bias_est_alpha", 0.0), 0.0, 0.95))

        self.reset()

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()

        self._sup_k = 0
        self._protect_left = 0
        self._recover_left = 0

        self.err_ema = 0.0
        self.prev_err_ema = 0.0
        self.bad_count = 0
        self._bias_accum = np.zeros(2, dtype=float)
        self._bias_accum_n = 0
        self._int_norm_prev = None
        self._reset_count = 0

        self.bias_xy = np.zeros(2, dtype=float)
        self._latched_bias_xy = np.zeros(2, dtype=float)
        self._bias_apply_left = 0
        

        self.active = False

    def _supervisor_tick(self) -> bool:
        self._sup_k += 1
        if self._sup_k >= self._sup_steps:
            self._sup_k = 0
            return True
        return False

    def compute_rpms(
        self,
        state: np.ndarray,
        p_ref_now: np.ndarray,
        v_ref_now: np.ndarray,
        p_ref_ahead: np.ndarray,
        v_ref_ahead: np.ndarray,
        yaw_des: float = 0.0,
    ) -> Tuple[np.ndarray, float, bool, bool, float, np.ndarray, float]:

        cur_pos, cur_quat, cur_vel, cur_ang_vel = _split_state(state)

        agentic_cfg = (self.cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
        if not self._printed_cfg:
            print("AGENTIC CFG USED:", agentic_cfg)
            self._printed_cfg = True

        dt = self.dt

        # Reference blend
        alpha = self.lookahead_alpha
        p_now = np.asarray(p_ref_now, dtype=float)
        p_ahead = np.asarray(p_ref_ahead, dtype=float)
        v_now = np.asarray(v_ref_now, dtype=float)
        v_ahead = np.asarray(v_ref_ahead, dtype=float)

        target_pos = (1.0 - alpha) * p_now + alpha * p_ahead
        v_ref = (1.0 - alpha) * v_now + alpha * v_ahead

        target_pos = target_pos.copy()
        v_ref = v_ref.copy()
        target_pos[2] = float(p_now[2])
        v_ref[2] = float(v_now[2])

        # Error + gate
        cur_pos = np.asarray(cur_pos, dtype=float)
        e = target_pos - cur_pos
        e[2] = 0.0
        err = float(np.linalg.norm(e[:2]))

        vel_xy = float(np.linalg.norm(np.asarray(cur_vel, dtype=float)[:2]))
        gate_ok = (vel_xy < self.int_gate_v_mps)

        # Timers
        protect_active = (self._protect_left > 0)
        recover_active = (self._recover_left > 0)

        if protect_active:
            self._protect_left -= 1
            if self._protect_left == 0:
                # Compute a bias estimate from the protect episode (mean residual)
                if self._bias_accum_n > 0:
                    mean_e = self._bias_accum / float(self._bias_accum_n)

                    # Convert mean residual into a reference bias (bounded)
                    est = -self.ki_bias * mean_e  # negative cancels drift direction

                    # Optional smoothing with previous latched bias
                    if self.bias_est_alpha > 0.0:
                        est = (1.0 - self.bias_est_alpha) * est + self.bias_est_alpha * self._latched_bias_xy

                    # Clamp per-axis and magnitude
                    est[0] = float(np.clip(est[0], -self.bias_int_clamp, self.bias_int_clamp))
                    est[1] = float(np.clip(est[1], -self.bias_int_clamp, self.bias_int_clamp))

                    nrm = float(np.linalg.norm(est))
                    if nrm > self.pos_corr_max_m:
                        est *= self.pos_corr_max_m / (nrm + 1e-9)

                    self._latched_bias_xy = est
                    self._bias_apply_left = self._bias_apply_steps

                # enter recovery cooldown
                self._recover_left = self._recover_steps

        elif recover_active:
            self._recover_left -= 1

        # Supervisor tick
        is_sup_tick = self._supervisor_tick()

        # Quiet regime: true do-nothing (except normal PID)
        if (err < self.quiet_err_m) and (vel_xy < self.quiet_vel_mps) and (not protect_active):
            self.err_ema = 0.0
            self.prev_err_ema = 0.0
            self.bad_count = 0
            self._reset_count = 0
            self._int_norm_prev = None

            # Bias decay only on sup ticks (optional, but diagram-aligned)
            if is_sup_tick:
                self.bias_xy *= float(self.bias_leak)

            rpms = self.inner.compute_rpms(
                state=state,
                target_pos=target_pos,
                target_vel=v_ref,
                integrate=True,
                int_leak=self.pid_int_leak_good,
                int_clamp=self.pid_int_clamp,
            )
            self.active = False
            return rpms, err, False, gate_ok, float(np.linalg.norm(self.bias_xy)), target_pos, 0.0

        # Health monitor
        self.prev_err_ema = float(self.err_ema)
        self.err_ema = (1.0 - self.err_ema_beta) * self.err_ema + self.err_ema_beta * err
        improve_mps = (self.prev_err_ema - self.err_ema) / (dt + 1e-9)

        hold_steps = max(1, int(round(self.hold_s * self.ctrl_hz)))
        if self.err_ema > self.act_err_m:
            self.bad_count += 1
        else:
            self.bad_count = max(0, self.bad_count - 1)

        persistent_bad = (self.bad_count >= hold_steps)
        not_improving = (self.err_ema > self.act_err_m) and (improve_mps < self.min_improve_mps)

        # Windup detection
        int_xy = self.inner.get_integral_xy()
        int_norm = float(np.linalg.norm(int_xy))
        if self._int_norm_prev is None:
            self._int_norm_prev = int_norm
        int_growth = int_norm - float(self._int_norm_prev)
        self._int_norm_prev = int_norm

        windup_like = (int_norm > self.int_trigger_norm) and (int_growth > self.int_trigger_growth)

        # Optional hard reset counter
        if self.pid_int_reset_err_m > 0.0 and self.err_ema > self.pid_int_reset_err_m:
            self._reset_count += 1
        else:
            self._reset_count = max(0, self._reset_count - 1)

        do_hard_reset = (self.pid_int_reset_err_m > 0.0) and (self._reset_count >= self._reset_steps)

        # ENTER PROTECT only on supervisor tick
        if is_sup_tick and (not protect_active) and (not recover_active):
            if gate_ok and windup_like:
                self._protect_left = self._protect_steps
                self._bias_accum[:] = 0.0
                self._bias_accum_n = 0
                protect_active = True

        # Role separation
        apply_bias = False
        target_pos_adj = target_pos.copy()
        latched_active = (self._bias_apply_left > 0) and (not protect_active) and (not recover_active)


        if latched_active:
            target_pos_adj[0] += float(self._latched_bias_xy[0])
            target_pos_adj[1] += float(self._latched_bias_xy[1])
            apply_bias = True
            self._bias_apply_left -= 1
        else:
            self._latched_bias_xy *= float(self.bias_leak)

        sat = False

        if protect_active:
            # PROTECT: integrator supervision only
            if do_hard_reset:
                self.inner.set_integral_xy(np.zeros(2, dtype=float))
            else:
                self.inner.leak_integral_xy(self.pid_int_leak_bad)

            integrate_pid = (not self.pid_int_freeze_on_bad)
            # Collect residual direction while protecting (no bias applied yet)
            if gate_ok:
                self._bias_accum += e[:2]        # e = target_pos - cur_pos (XY)
                self._bias_accum_n += 1

            # Bias decay only on sup ticks
            if is_sup_tick:
                self.bias_xy *= float(self.bias_leak)

            # Avoid double-leaking inside PID
            pid_int_leak = 1.0

        else:
            # NORMAL / RECOVER
            integrate_pid = True
            pid_int_leak = self.pid_int_leak_good

            bias_allowed = (not recover_active) and (not latched_active)

            if bias_allowed and is_sup_tick:
                if windup_like and (int_norm > 0.8 * self.pid_int_clamp) and gate_ok:
                    self.bias_xy = self.bias_xy - self.ki_bias * e[:2] * self.supervisor_period_s
                    self.bias_xy[0] = float(np.clip(self.bias_xy[0], -self.bias_int_clamp, self.bias_int_clamp))
                    self.bias_xy[1] = float(np.clip(self.bias_xy[1], -self.bias_int_clamp, self.bias_int_clamp))
                else:
                    self.bias_xy *= float(self.bias_leak)

            # Apply bias (bounded)
            bias_norm_now = float(np.linalg.norm(self.bias_xy))
            if bias_allowed and (bias_norm_now > 1e-6):
                if bias_norm_now > self.pos_corr_max_m:
                    self.bias_xy *= self.pos_corr_max_m / (bias_norm_now + 1e-9)
                    sat = True  # <-- correct "sat": we actually clamped

                target_pos_adj[0] += float(self.bias_xy[0])
                target_pos_adj[1] += float(self.bias_xy[1])
                apply_bias = True
        
        if latched_active:
            bias_norm = float(np.linalg.norm(self._latched_bias_xy))
        else:
            bias_norm = float(np.linalg.norm(self.bias_xy))

        rpms = self.inner.compute_rpms(
            state=state,
            target_pos=target_pos_adj,
            target_vel=v_ref,
            integrate=integrate_pid,
            integrate_z=True,          # recommended (see note below)
            int_leak=pid_int_leak,
            int_clamp=self.pid_int_clamp,
        )
        self.active = bool(apply_bias)
        return rpms, err, sat, gate_ok, bias_norm, target_pos_adj, float(1.0 if apply_bias else 0.0)
