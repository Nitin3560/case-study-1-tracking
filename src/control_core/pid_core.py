from __future__ import annotations
"""
PIDCore – outer-loop PID producing velocity command.

IMPORTANT NOTE ON SATURATION
----------------------------
True actuator saturation MUST be provided by downstream layers
(e.g., motor RPM / thrust limits). This file:
  - consumes `saturated` (bool) or `sat_frac` ([0,1])
  - does NOT infer saturation on its own

Best default for PID → DSLPIDControl stack:
  - axis_mask = (1, 1, 0)
  - d_mode = "vel"
  - clamp_mode = "xy"
  - integrator hysteresis enabled
"""

import numpy as np
from dataclasses import dataclass
from math import exp


def _clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


@dataclass
class PIDConfig:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.6

    # Velocity clamp
    v_max: float = 3.0
    axis_mask: tuple[float, float, float] = (1.0, 1.0, 0.0)
    clamp_mode: str = "xy"
    v_max_xy: float | None = None
    v_max_z: float | None = None
    v_max_axis: tuple[float, float, float] | None = None

    # Integrator gating
    int_gate_on_err_m: float = 2.0
    int_gate_off_err_m: float = 3.0
    int_gate_on_vel_mps: float = 4.0
    int_gate_off_vel_mps: float = 5.0

    int_leak: float = 0.995
    int_clamp: float = 2.0

    # Derivative
    d_mode: str = "vel"
    d_blend: float = 0.5
    d_tau_s: float = 0.05

    # Saturation-aware gating
    int_sat_hold: bool = True
    sat_eps: float = 1e-6
    int_sat_scale: bool = True
    int_sat_scale_pow: float = 1.0
    int_sat_leak: float = 1.0

    # --- BACK-CALCULATION (ANTI-WINDUP) ---
    aw_enabled: bool = True
    aw_gain: float = 0.5          # dimensionless
    aw_tau_s: float = 0.2         # smoothing time constant
    aw_max_unwind: float = 0.5    # max |unwind| per step


class PIDCore:
    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg

        self.e_int = np.zeros(3)
        self.e_prev = np.zeros(3)

        self.d_filt = np.zeros(3)
        self.v_filt = np.zeros(3)

        self._sat_frac = 0.0
        self._int_enabled = False
        self._int_hold = False
        self._int_leak_override = None

        self._mask = np.asarray(cfg.axis_mask, float)

    def reset(self):
        self.e_int[:] = 0.0
        self.e_prev[:] = 0.0
        self.d_filt[:] = 0.0
        self.v_filt[:] = 0.0
        self._sat_frac = 0.0
        self._int_enabled = False
        self._int_hold = False
        self._int_leak_override = None

    def set_saturation(self, saturated: bool | float):
        if isinstance(saturated, (float, np.floating)):
            self._sat_frac = float(_clamp(saturated, 0.0, 1.0))
        else:
            self._sat_frac = 1.0 if saturated else 0.0

    def set_integrator_hold(self, hold: bool):
        self._int_hold = bool(hold)

    def set_integrator_leak_override(self, leak: float | None):
        if leak is None:
            self._int_leak_override = None
        else:
            self._int_leak_override = float(leak)

    def reset_integrator(self):
        self.e_int[:] = 0.0

    def _alpha(self, tau: float, dt: float) -> float:
        if tau <= 0:
            return 0.0
        return exp(-dt / tau)

    def step(
        self,
        e,
        v,
        dt,
        *,
        integrate_xy: bool = True,
        saturated: bool | float | None = None,
        return_debug: bool = False,
    ):
        e = np.asarray(e, float)
        v = np.asarray(v, float)
        dt = float(dt)

        if dt <= 0:
            return np.zeros(3)

        # Saturation
        if saturated is None:
            sat_frac = self._sat_frac
        else:
            self.set_saturation(saturated)
            sat_frac = self._sat_frac

        sat_now = sat_frac > self.cfg.sat_eps

        # Mask
        e_m = e * self._mask
        v_m = v * self._mask

        e_xy_n = np.linalg.norm(e_m[:2])
        v_xy_n = np.linalg.norm(v_m[:2])

        # Integrator hysteresis
        if self.cfg.ki > 0:
            if self._int_enabled:
                if e_xy_n > self.cfg.int_gate_off_err_m or v_xy_n > self.cfg.int_gate_off_vel_mps:
                    self._int_enabled = False
            else:
                if e_xy_n < self.cfg.int_gate_on_err_m and v_xy_n < self.cfg.int_gate_on_vel_mps:
                    self._int_enabled = True
        else:
            self._int_enabled = False
            self.e_int[:] = 0.0

        allow_int = (
            self._int_enabled
            and (not sat_now or not self.cfg.int_sat_hold)
            and self.cfg.ki > 0
            and not self._int_hold
        )

        # Integral scaling
        if self.cfg.int_sat_scale:
            int_scale = 1.0 - sat_frac ** self.cfg.int_sat_scale_pow
        else:
            int_scale = 1.0

        # Integrate
        if allow_int:
            if integrate_xy:
                self.e_int[:2] += e[:2] * dt * int_scale
            else:
                self.e_int += e * dt * int_scale
        else:
            leak = self._int_leak_override if self._int_leak_override is not None else self.cfg.int_leak
            leak = leak * (self.cfg.int_sat_leak if sat_now else 1.0)
            self.e_int *= leak

        # Clamp integral
        if integrate_xy:
            self.e_int[:2] = _clamp(self.e_int[:2], -self.cfg.int_clamp, self.cfg.int_clamp)
        else:
            self.e_int[:] = _clamp(self.e_int, -self.cfg.int_clamp, self.cfg.int_clamp)

        # Derivative
        a = self._alpha(self.cfg.d_tau_s, dt)
        d_raw = (e_m - self.e_prev) / dt
        self.d_filt = a * self.d_filt + (1 - a) * d_raw
        self.v_filt = a * self.v_filt + (1 - a) * v_m
        self.e_prev = e_m.copy()

        if self.cfg.d_mode == "vel":
            d_term = -self.cfg.kd * self.v_filt
        elif self.cfg.d_mode == "err":
            d_term = self.cfg.kd * self.d_filt
        else:
            b = self.cfg.d_blend
            d_term = self.cfg.kd * (b * self.d_filt - (1 - b) * self.v_filt)

        # PID sum (pre-clamp)
        v_cmd_pre = (
            self.cfg.kp * e_m
            + self.cfg.ki * (self.e_int * self._mask)
            + d_term
        )

        # Clamp output
        v_cmd = v_cmd_pre.copy()
        vlim_xy = self.cfg.v_max_xy or self.cfg.v_max

        if self.cfg.clamp_mode == "xy":
            n = np.linalg.norm(v_cmd[:2])
            if n > vlim_xy:
                v_cmd[:2] *= vlim_xy / (n + 1e-9)

        elif self.cfg.clamp_mode == "xyz":
            n = np.linalg.norm(v_cmd)
            if n > self.cfg.v_max:
                v_cmd *= self.cfg.v_max / (n + 1e-9)

        elif self.cfg.clamp_mode == "per_axis":
            vlims = np.asarray(self.cfg.v_max_axis)
            v_cmd = _clamp(v_cmd, -vlims, vlims)

        # -----------------------------
        # BACK-CALCULATION (ANTI-WINDUP)
        # -----------------------------
        if self.cfg.aw_enabled and self.cfg.ki > 0:
            clamp_err = v_cmd - v_cmd_pre
            if sat_now or np.linalg.norm(clamp_err[:2]) > 1e-6:
                aw_alpha = self._alpha(self.cfg.aw_tau_s, dt)
                aw = (
                    self.cfg.aw_gain
                    * aw_alpha
                    * clamp_err
                    / max(self.cfg.ki, 1e-6)
                )
                aw = _clamp(aw, -self.cfg.aw_max_unwind, self.cfg.aw_max_unwind)
                self.e_int += aw * self._mask

        if return_debug:
            return v_cmd, {
                "sat_frac": float(sat_frac),
                "clamp_err_xy": float(np.linalg.norm((v_cmd - v_cmd_pre)[:2])),
                "e_int_xy": float(np.linalg.norm(self.e_int[:2])),
            }

        return v_cmd
