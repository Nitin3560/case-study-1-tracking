from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
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


class DisturbanceModel:
    """
    Applies wind as external force in world frame.
    Produces a 'GPS measurement' with noise + random-walk drift + optional outage.
    Supports two config schemas:

    New (nested):
      disturbance:
        wind: {mode: ..., ...}
        gps_drift: {enabled: ..., ...}

    Legacy/simple (your scenario_default.yaml):
      disturbance:
        wind_force_std: 0.8
        gps_drift_std: 0.05
    """
    def __init__(self, cfg: Dict[str, Any], seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        dist_cfg = cfg.get("disturbance", {}) or {}

        # --- WIND config (support both schemas) ---
        if "wind" in dist_cfg and isinstance(dist_cfg["wind"], dict):
            self.wind_cfg = dist_cfg["wind"]
        else:
            # Legacy/simple schema: interpret as zero-mean Gaussian wind force in XY
            wind_std = float(dist_cfg.get("wind_force_std", 0.0))
            self.wind_cfg = {
                "mode": "gaussian",
                "gaussian_force_std_n": wind_std,
            }

        # --- GPS drift config (support both schemas) ---
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

    def step_wind_force(self, t: float) -> np.ndarray:
        mode = str(self.wind_cfg.get("mode", "none")).lower()
        if mode == "none":
            return np.zeros(3, dtype=float)

        if mode == "constant":
            return np.array(self.wind_cfg.get("constant_force_n", [0.0, 0.0, 0.0]), dtype=float)

        # NEW: gaussian wind (legacy/simple schema support)
        if mode == "gaussian":
            std = float(self.wind_cfg.get("gaussian_force_std_n", 0.0))
            fxy = self.rng.normal(0.0, std, size=2)
            return np.array([fxy[0], fxy[1], 0.0], dtype=float)

        # gust (your existing behavior, unchanged)
        if mode == "gust":
            # Constant bias + intermittent gusts
            base = np.array(self.wind_cfg.get("constant_force_n", [0.0, 0.0, 0.0]), dtype=float)

            if t >= self.wind.gust_active_until and t >= self.wind.next_gust_time:
                dur_lo, dur_hi = self.wind_cfg["gust_duration_s"]
                int_lo, int_hi = self.wind_cfg["gust_interval_s"]
                dur = self.rng.uniform(dur_lo, dur_hi)
                gap = self.rng.uniform(int_lo, int_hi)
                self.wind.gust_active_until = t + dur
                self.wind.next_gust_time = t + dur + gap

                peak = float(self.wind_cfg["gust_force_n_max"])
                theta = self.rng.uniform(0, 2*np.pi)
                fx = peak * np.cos(theta)
                fy = peak * np.sin(theta)
                self.wind.current_force = np.array([fx, fy, 0.0], dtype=float)

            if t < self.wind.gust_active_until:
                return base + self.wind.current_force
            return base


        # unknown mode -> safe fallback
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

    def gps_measurement(self, true_pos: np.ndarray, t: float, dt: float) -> np.ndarray:
        if not bool(self.drift_cfg.get("enabled", False)):
            return true_pos.copy()

        outage_cfg = self.drift_cfg.get("outage", {"enabled": False})
        if outage_cfg.get("enabled", False):
            if t >= self.drift.outage_until:
                if self.rng.uniform() < float(outage_cfg.get("prob_per_s", 0.0)) * dt:
                    lo, hi = outage_cfg.get("duration_s", [0.0, 0.0])
                    self.drift.outage_until = t + self.rng.uniform(lo, hi)

        if t < self.drift.outage_until and self.drift.last_meas is not None:
            return self.drift.last_meas.copy()

        rw = float(self.drift_cfg.get("random_walk_std_m_per_s", 0.0))
        self.drift.bias += self.rng.normal(0.0, rw * np.sqrt(dt), size=3)

        ns = float(self.drift_cfg.get("noise_std_m", 0.0))
        meas = true_pos + self.drift.bias + self.rng.normal(0.0, ns, size=3)
        self.drift.last_meas = meas.copy()
        return meas
