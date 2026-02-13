from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Dict, Any


@dataclass
class SupervisorState:
    ref_shift: np.ndarray
    formation_scale: float
    smooth_alpha: float
    int_hold: bool
    supervisor_mode: float
    supervisor_active: float


class AgenticSupervisor:
    """
    Slow, selective supervisor that modifies reference/constraints only.
    Priority modes:
      3 = connectivity rescue
      2 = anti-windup
      1 = drift correction
      0 = off
    """

    def __init__(self, cfg: Dict[str, Any]):
        agentic_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}
        sup_cfg = (agentic_cfg.get("supervisor", {}) or {})

        self.hz = float(sup_cfg.get("hz", 5.0))
        self.hold_s = float(sup_cfg.get("hold_s", 1.0))
        self.err_thresh_m = float(sup_cfg.get("err_thresh_m", 1.5))
        self.err_recover_slope_mps = float(sup_cfg.get("err_recover_slope_mps", 0.1))
        self.conn_thresh = float(sup_cfg.get("conn_thresh", 0.6))
        self.form_thresh = float(sup_cfg.get("form_thresh", 1.2))
        self.ref_shift_rate_mps = float(sup_cfg.get("ref_shift_rate_mps", 0.2))
        self.ref_shift_max_m = float(sup_cfg.get("ref_shift_max_m", agentic_cfg.get("max_ref_shift_m", 0.2)))
        self.smooth_alpha_active = float(sup_cfg.get("smooth_alpha_active", 0.4))
        self.formation_scale_min = float(sup_cfg.get("formation_scale_min", 0.8))
        self.formation_scale_max = float(sup_cfg.get("formation_scale_max", 1.2))
        self.sat_hold_thresh = float(sup_cfg.get("sat_hold_thresh", 0.6))
        self.sat_release_thresh = float(sup_cfg.get("sat_release_thresh", self.sat_hold_thresh * 0.6))
        self.int_leak_override = float(sup_cfg.get("int_leak_override", 0.98))
        self.drift_disable_phase_ge = int(sup_cfg.get("drift_disable_phase_ge", 10))
        self.ref_shift_decay_mps = float(sup_cfg.get("ref_shift_decay_mps", 0.0))
        self.err_hard_m = float(sup_cfg.get("err_hard_m", 1e9))
        self.hard_hold_s = float(sup_cfg.get("hard_hold_s", 0.0))
        self.recovery_ref_shift_rate_mps = float(sup_cfg.get("recovery_ref_shift_rate_mps", 0.0))
        self.phase_boost_s = float(sup_cfg.get("phase_boost_s", 0.0))
        self.phase_boost_rate_mps = float(sup_cfg.get("phase_boost_rate_mps", 0.0))
        self.phase_boost_smooth_alpha = float(sup_cfg.get("phase_boost_smooth_alpha", self.smooth_alpha_active))
        self.ref_shift_hold_decay = float(sup_cfg.get("ref_shift_hold_decay", 0.995))
        self.comm_freeze_refshift = bool(sup_cfg.get("comm_freeze_refshift", True))
        self.comm_freeze_conn_thresh = float(sup_cfg.get("comm_freeze_conn_thresh", 0.18))
        self.comm_freeze_ratio_thresh = float(sup_cfg.get("comm_freeze_ratio_thresh", 0.60))
        self.ref_shift_freeze_mode = str(sup_cfg.get("ref_shift_freeze_mode", "hold")).lower()
        self.ref_shift_freeze_decay = float(sup_cfg.get("ref_shift_freeze_decay", 0.995))

        self._last_phase_id = None
        self._phase_boost_timer = 0.0

        self._sup_timer = 0.0
        self._hold_timer = 0.0
        self._last_err_com = None
        self._hard_timer = 0.0
        self._prev_com = np.zeros(3, dtype=float)
        self._conn_ref = None

        self.state = SupervisorState(
            ref_shift=np.zeros(3, dtype=float),
            formation_scale=1.0,
            smooth_alpha=1.0,
            int_hold=False,
            supervisor_mode=0.0,
            supervisor_active=0.0,
        )

    def should_tick(self, dt_ctrl: float) -> bool:
        self._sup_timer += float(dt_ctrl)
        if self._sup_timer >= max(1e-6, 1.0 / self.hz):
            return True
        return False

    def step(
        self,
        *,
        dt_ctrl: float,
        pos_true: np.ndarray,
        p_ref_now: np.ndarray,
        formation_err: float,
        connectivity_rate: float,
        sat_mean: float,
        phase_id: int,
        comm_scale: float = 1.0,
        active_mask: np.ndarray | None = None,
        trusted_mask: np.ndarray | None = None,
        sensor_corrupt_mask: np.ndarray | None = None,
    ) -> SupervisorState:
        sup_dt = self._sup_timer
        self._sup_timer = 0.0

        n = pos_true.shape[0]
        if active_mask is None:
            active_mask = np.ones(n, dtype=bool)
        if trusted_mask is None:
            trusted_mask = np.asarray(active_mask, dtype=bool).copy()
        if sensor_corrupt_mask is None:
            sensor_corrupt_mask = np.zeros(n, dtype=bool)

        trusted_idx = np.where(np.asarray(trusted_mask, dtype=bool))[0]
        active_idx = np.where(np.asarray(active_mask, dtype=bool))[0]
        if trusted_idx.size >= 2:
            com = pos_true[trusted_idx].mean(axis=0)
        elif active_idx.size > 0:
            com = pos_true[active_idx].mean(axis=0)
        else:
            com = self._prev_com.copy()
        self._prev_com = np.asarray(com, dtype=float).copy()
        com_ref = p_ref_now
        err_com = float(np.linalg.norm((com - com_ref)[:2]))

        if err_com > self.err_thresh_m:
            self._hold_timer += sup_dt
        else:
            self._hold_timer = 0.0

        recovering = False
        if self._last_err_com is not None:
            slope = (err_com - self._last_err_com) / max(sup_dt, 1e-9)
            recovering = slope < -abs(self.err_recover_slope_mps)
        self._last_err_com = err_com

        if err_com > self.err_hard_m:
            self._hard_timer = max(self._hard_timer, self.hard_hold_s)
        self._hard_timer = max(0.0, self._hard_timer - sup_dt)

        mode = 0
        # phase change detection
        if self._last_phase_id is None or phase_id != self._last_phase_id:
            self._phase_boost_timer = self.phase_boost_s
        self._last_phase_id = phase_id
        if self._phase_boost_timer > 0.0:
            self._phase_boost_timer = max(0.0, self._phase_boost_timer - sup_dt)

        if connectivity_rate < self.conn_thresh:
            mode = 3
        elif self._hard_timer > 0.0 or sat_mean > self.sat_hold_thresh:
            mode = 2
        elif (phase_id < self.drift_disable_phase_ge) and (self._hold_timer >= self.hold_s and not recovering) and (err_com > self.err_thresh_m):
            mode = 1

        # Exit anti-windup sooner if saturation clears
        if mode == 2 and sat_mean < self.sat_release_thresh and connectivity_rate >= self.conn_thresh:
            mode = 0

        self.state.supervisor_mode = float(mode)
        self.state.supervisor_active = 1.0 if mode > 0 else 0.0

        # Reset to defaults each tick; apply single-mode action only
        self.state.smooth_alpha = 1.0
        self.state.formation_scale = 1.0
        self.state.int_hold = False

        sensing_compromised = bool(np.any(np.asarray(sensor_corrupt_mask, dtype=bool) & np.asarray(active_mask, dtype=bool)))
        comm_fault_active = float(comm_scale) < 0.999
        if self._conn_ref is None:
            self._conn_ref = float(connectivity_rate)
        elif not comm_fault_active:
            # Track nominal connectivity baseline only outside explicit comm faults.
            self._conn_ref = 0.98 * float(self._conn_ref) + 0.02 * float(connectivity_rate)
        conn_ratio = float(connectivity_rate) / max(1e-6, float(self._conn_ref))
        freeze_refshift = bool(
            self.comm_freeze_refshift
            and (
                float(connectivity_rate) < self.comm_freeze_conn_thresh
                or conn_ratio < self.comm_freeze_ratio_thresh
            )
        )

        if sensing_compromised:
            self.state.ref_shift[:2] *= float(np.clip(self.ref_shift_hold_decay, 0.0, 1.0))

        if freeze_refshift:
            if self.ref_shift_freeze_mode == "decay":
                self.state.ref_shift[:2] *= float(np.clip(self.ref_shift_freeze_decay, 0.0, 1.0))

        if mode == 3:
            self.state.formation_scale = self.formation_scale_min
            self.state.smooth_alpha = self.smooth_alpha_active
        elif mode == 2:
            self.state.int_hold = True
        elif mode == 1 and not sensing_compromised:
            delta = (com - com_ref)
            delta[2] = 0.0
            norm = float(np.linalg.norm(delta[:2]))
            if norm > 1e-9:
                step = (delta / norm) * min(norm, self.ref_shift_rate_mps * sup_dt)
                self.state.ref_shift += step
                rnorm = float(np.linalg.norm(self.state.ref_shift[:2]))
                if rnorm > self.ref_shift_max_m:
                    self.state.ref_shift[:2] *= self.ref_shift_max_m / (rnorm + 1e-9)

        # Late-phase recovery: allow small, bounded re-anchor even when drift correction is disabled
        if (
            (not sensing_compromised)
            and (not freeze_refshift)
            and phase_id >= self.drift_disable_phase_ge
            and self.recovery_ref_shift_rate_mps > 0.0
            and err_com > self.err_thresh_m
        ):
            delta = (com - com_ref)
            delta[2] = 0.0
            norm = float(np.linalg.norm(delta[:2]))
            if norm > 1e-9:
                step = (delta / norm) * min(norm, self.recovery_ref_shift_rate_mps * sup_dt)
                self.state.ref_shift += step
                rnorm = float(np.linalg.norm(self.state.ref_shift[:2]))
                if rnorm > self.ref_shift_max_m:
                    self.state.ref_shift[:2] *= self.ref_shift_max_m / (rnorm + 1e-9)

        # Phase transition boost: faster re-anchor + less smoothing for a short window
        if self._phase_boost_timer > 0.0:
            self.state.smooth_alpha = max(self.state.smooth_alpha, self.phase_boost_smooth_alpha)
            if (not sensing_compromised) and (not freeze_refshift) and self.phase_boost_rate_mps > 0.0 and err_com > self.err_thresh_m:
                delta = (com - com_ref)
                delta[2] = 0.0
                norm = float(np.linalg.norm(delta[:2]))
                if norm > 1e-9:
                    step = (delta / norm) * min(norm, self.phase_boost_rate_mps * sup_dt)
                    self.state.ref_shift += step
                    rnorm = float(np.linalg.norm(self.state.ref_shift[:2]))
                    if rnorm > self.ref_shift_max_m:
                        self.state.ref_shift[:2] *= self.ref_shift_max_m / (rnorm + 1e-9)

        # decay ref_shift toward 0
        if self.ref_shift_decay_mps > 0.0:
            rnorm = float(np.linalg.norm(self.state.ref_shift[:2]))
            if rnorm > 1e-9:
                decay = min(rnorm, self.ref_shift_decay_mps * sup_dt)
                self.state.ref_shift[:2] *= max(0.0, (rnorm - decay) / rnorm)

        return self.state

    def integrator_leak_override(self) -> float | None:
        return self.int_leak_override if self.state.int_hold else None
