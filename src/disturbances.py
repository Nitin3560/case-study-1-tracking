from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np
import random
import pybullet as p


@dataclass
class WindState:
    gust_active_until: float = 0.0
    next_gust_time: float = 0.0
    current_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class DriftState:
    bias: np.ndarray
    outage_until: float = 0.0
    last_meas: Optional[np.ndarray] = None


def _sorted_pair(x) -> Tuple[float, float]:
    """Return (lo, hi) sorted and safe."""
    lo, hi = float(x[0]), float(x[1])
    return (lo, hi) if lo <= hi else (hi, lo)


class DisturbanceModel:
    """
    Applies wind as external force in world frame.
    Produces a 'GPS measurement' with noise + random-walk drift + optional outage.

    Supports schemas:

    New (nested):
      disturbance:
        wind:
          mode: gust | phased | constant | gaussian | none
          ... mode-specific ...
        gps_drift:
          enabled: true/false
          random_walk_std_m_per_s: ...
          noise_std_m: ...
          outage: ...

    Legacy/simple:
      disturbance:
        wind_force_std: ...
        gps_drift_std: ...
    """

    def __init__(self, cfg: Dict[str, Any], seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        dist_cfg = cfg.get("disturbance", {}) or {}

        # --- WIND config ---
        if "wind" in dist_cfg and isinstance(dist_cfg["wind"], dict):
            self.wind_cfg = dist_cfg["wind"]
        else:
            # Legacy/simple schema: interpret as zero-mean Gaussian wind force in XY
            wind_std = float(dist_cfg.get("wind_force_std", 0.0))
            self.wind_cfg = {
                "mode": "gaussian",
                "gaussian_force_std_n": wind_std,
            }

        # --- GPS drift config ---
        if "gps_drift" in dist_cfg and isinstance(dist_cfg["gps_drift"], dict):
            self.drift_cfg = dist_cfg["gps_drift"]
        else:
            gps_std = float(dist_cfg.get("gps_drift_std", 0.0))
            self.drift_cfg = {
                "enabled": gps_std > 0.0,
                "noise_std_m": gps_std,
                "random_walk_std_m_per_s": 0.0,
                "outage": {"enabled": False},
            }

        self.wind = WindState()
        self.drift = DriftState(bias=np.zeros(3, dtype=float))

    # -------------------------
    # Wind helpers
    # -------------------------
    def _phase_cfg_at(self, t: float) -> Dict[str, Any]:
        """
        If wind mode is 'phased', pick the active phase dict.
        Each phase is a dict with:
          - t0_s, t1_s (optional, t1_s can be omitted => infinity)
          - constant_force_n
          - gust_force_n_max
          - gust_interval_s
          - gust_duration_s
          - gust_theta_range_deg (optional; e.g., [-20, 20] around wind direction)
        """
        phases = self.wind_cfg.get("phases", None)
        if not phases:
            return {}

        for ph in phases:
            t0 = float(ph.get("t0_s", 0.0))
            t1 = float(ph.get("t1_s", 1e9))
            if t0 <= t < t1:
                return ph
        return phases[-1] if phases else {}

    def _gust_force(self, t: float, base: np.ndarray, peak: float,
                    interval_s, duration_s, theta_range_deg=None) -> np.ndarray:
        """
        Common gust logic with safety guards.
        """
        # Safety: always sort pairs to avoid "high - low < 0"
        dur_lo, dur_hi = _sorted_pair(duration_s)
        int_lo, int_hi = _sorted_pair(interval_s)

        # If parameters degenerate, treat as no gust
        if peak <= 0.0 or dur_hi <= 0.0 or int_hi <= 0.0:
            return base

        if t >= self.wind.gust_active_until and t >= self.wind.next_gust_time:
            dur = self.rng.uniform(dur_lo, max(dur_hi, dur_lo + 1e-9))
            gap = self.rng.uniform(int_lo, max(int_hi, int_lo + 1e-9))
            self.wind.gust_active_until = t + dur
            self.wind.next_gust_time = t + dur + gap

            # Choose gust direction
            if theta_range_deg is not None:
                # Gust around the base direction
                # If base is zero, fall back to uniform angle
                base_xy = base.copy()
                base_xy[2] = 0.0
                n = np.linalg.norm(base_xy[:2])
                if n < 1e-9:
                    theta = self.rng.uniform(0.0, 2*np.pi)
                else:
                    base_theta = float(np.arctan2(base_xy[1], base_xy[0]))
                    lo_deg, hi_deg = _sorted_pair(theta_range_deg)
                    dtheta = np.deg2rad(self.rng.uniform(lo_deg, hi_deg))
                    theta = base_theta + dtheta
            else:
                theta = self.rng.uniform(0.0, 2*np.pi)

            fx = peak * np.cos(theta)
            fy = peak * np.sin(theta)
            self.wind.current_force = np.array([fx, fy, 0.0], dtype=float)

        if t < self.wind.gust_active_until:
            return base + self.wind.current_force
        return base

    def step_wind_force(self, t: float) -> np.ndarray:
        mode = str(self.wind_cfg.get("mode", "none")).lower()

        if mode == "none":
            return np.zeros(3, dtype=float)

        if mode == "constant":
            return np.array(self.wind_cfg.get("constant_force_n", [0.0, 0.0, 0.0]), dtype=float)

        if mode == "gaussian":
            std = float(self.wind_cfg.get("gaussian_force_std_n", 0.0))
            fxy = self.rng.normal(0.0, std, size=2)
            return np.array([fxy[0], fxy[1], 0.0], dtype=float)

        # Original gust mode (single regime for whole episode)
        if mode == "gust":
            base = np.array(self.wind_cfg.get("constant_force_n", [0.0, 0.0, 0.0]), dtype=float)
            peak = float(self.wind_cfg.get("gust_force_n_max", 0.0))
            interval_s = self.wind_cfg.get("gust_interval_s", [2.0, 5.0])
            duration_s = self.wind_cfg.get("gust_duration_s", [0.3, 0.8])
            return self._gust_force(t, base, peak, interval_s, duration_s, theta_range_deg=None)

        # NEW: phased wind (deceptive early -> failure -> recovery)
        if mode == "phased":
            ph = self._phase_cfg_at(t)

            base = np.array(ph.get("constant_force_n", self.wind_cfg.get("constant_force_n", [0.0, 0.0, 0.0])),
                            dtype=float)

            peak = float(ph.get("gust_force_n_max", self.wind_cfg.get("gust_force_n_max", 0.0)))
            interval_s = ph.get("gust_interval_s", self.wind_cfg.get("gust_interval_s", [2.0, 5.0]))
            duration_s = ph.get("gust_duration_s", self.wind_cfg.get("gust_duration_s", [0.3, 0.8]))

            theta_rng = ph.get("gust_theta_range_deg", None)
            return self._gust_force(t, base, peak, interval_s, duration_s, theta_range_deg=theta_rng)

        # Unknown mode -> safe fallback
        return np.zeros(3, dtype=float)

    def apply_wind(self, env, t: float):
        f = self.step_wind_force(t)
        if np.allclose(f, 0.0):
            return
        for drone_id in env.DRONE_IDS:
            p.applyExternalForce(
                objectUniqueId=drone_id,
                linkIndex=-1,
                forceObj=f.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=env.CLIENT,
            )

    # -------------------------
    # GPS measurement model
    # -------------------------
    def gps_measurement(self, true_pos: np.ndarray, t: float, dt: float) -> np.ndarray:
        if not bool(self.drift_cfg.get("enabled", False)):
            return true_pos.copy()

        outage_cfg = self.drift_cfg.get("outage", {"enabled": False})
        if outage_cfg.get("enabled", False):
            if t >= self.drift.outage_until:
                if self.rng.uniform() < float(outage_cfg.get("prob_per_s", 0.0)) * dt:
                    lo, hi = outage_cfg.get("duration_s", [0.0, 0.0])
                    lo, hi = _sorted_pair([lo, hi])
                    self.drift.outage_until = t + self.rng.uniform(lo, max(hi, lo + 1e-9))

        if t < self.drift.outage_until and self.drift.last_meas is not None:
            return self.drift.last_meas.copy()

        rw = float(self.drift_cfg.get("random_walk_std_m_per_s", 0.0))
        self.drift.bias += self.rng.normal(0.0, rw * np.sqrt(max(dt, 1e-9)), size=3)

        ns = float(self.drift_cfg.get("noise_std_m", 0.0))
        meas = true_pos + self.drift.bias + self.rng.normal(0.0, ns, size=3)
        self.drift.last_meas = meas.copy()
        return meas
