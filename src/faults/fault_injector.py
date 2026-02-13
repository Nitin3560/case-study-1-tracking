from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


class FaultInjector:
    """
    Runtime fault injector wrapper.

    Supports:
    - dropout (agent-level): hover / freeze / no_control / offline
    - sensor_corruption (agent-level): bias + Gaussian noise on pos/vel
    - comm_degrade (global): comm range scaling
    """

    def __init__(self, faults_cfg: Dict[str, Any] | None):
        faults_cfg = faults_cfg or {}
        self.enabled = bool(faults_cfg.get("enabled", False))
        self.events: List[Dict[str, Any]] = list(faults_cfg.get("events", []) or [])
        self._last_cmd: Dict[int, np.ndarray] = {}

    @staticmethod
    def _time_active(ev: Dict[str, Any], t: float) -> bool:
        t0 = float(ev.get("t0_s", 0.0))
        t1 = ev.get("t1_s", None)
        if t < t0:
            return False
        if t1 is None:
            return True
        return t < float(t1)

    def active_events(self, t: float) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return [ev for ev in self.events if self._time_active(ev, t)]

    def apply_sensing(self, agent_id: int, cur_pos: np.ndarray, cur_vel: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return cur_pos, cur_vel

        pos = np.asarray(cur_pos, dtype=float).copy()
        vel = np.asarray(cur_vel, dtype=float).copy()

        for ev in self.active_events(t):
            if str(ev.get("type", "")).lower() != "sensor_corruption":
                continue
            if int(ev.get("agent_id", -1)) != int(agent_id):
                continue
            pos_bias = np.asarray(ev.get("pos_bias_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
            pos_std = float(ev.get("pos_noise_std_m", 0.0))
            vel_std = float(ev.get("vel_noise_std_mps", 0.0))
            pos += pos_bias
            if pos_std > 0.0:
                pos += np.random.normal(0.0, pos_std, size=3)
            if vel_std > 0.0:
                vel += np.random.normal(0.0, vel_std, size=3)

        return pos, vel

    def apply_control(self, agent_id: int, cmd: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        cmd is expected to be per-agent motor command shape (4,).
        """
        cmd = np.asarray(cmd, dtype=float)
        meta = {"dropout_active": False, "dropout_mode": ""}
        if not self.enabled:
            self._last_cmd[int(agent_id)] = cmd.copy()
            return cmd, meta

        out = cmd.copy()
        for ev in self.active_events(t):
            if str(ev.get("type", "")).lower() != "dropout":
                continue
            if int(ev.get("agent_id", -1)) != int(agent_id):
                continue

            mode = str(ev.get("mode", "hover")).lower()
            meta["dropout_active"] = True
            meta["dropout_mode"] = mode

            if mode == "freeze":
                prev = self._last_cmd.get(int(agent_id), out)
                out = np.asarray(prev, dtype=float).copy()
            elif mode == "offline":
                # True offline semantics at actuator level: no command update.
                out = np.zeros_like(out)
            elif mode == "no_control":
                out = np.zeros_like(out)
            else:
                # hover-safe fallback: keep near previous command if available; else neutral zeros.
                prev = self._last_cmd.get(int(agent_id), np.zeros_like(out))
                out = np.asarray(prev, dtype=float).copy()

        self._last_cmd[int(agent_id)] = out.copy()
        return out, meta

    def comm_range_scale(self, t: float) -> float:
        if not self.enabled:
            return 1.0
        scale = 1.0
        for ev in self.active_events(t):
            if str(ev.get("type", "")).lower() != "comm_degrade":
                continue
            scale *= float(ev.get("comm_range_scale", 1.0))
        return float(max(scale, 1e-3))

    def agent_has_sensor_fault(self, agent_id: int, t: float) -> bool:
        if not self.enabled:
            return False
        for ev in self.active_events(t):
            if str(ev.get("type", "")).lower() == "sensor_corruption" and int(ev.get("agent_id", -1)) == int(agent_id):
                return True
        return False

    def sensor_corrupt_mask(self, n_agents: int, t: float) -> np.ndarray:
        mask = np.zeros(int(n_agents), dtype=bool)
        if not self.enabled:
            return mask
        for i in range(int(n_agents)):
            mask[i] = self.agent_has_sensor_fault(i, t)
        return mask

    def is_agent_offline(self, agent_id: int, t: float) -> bool:
        if not self.enabled:
            return False
        for ev in self.active_events(t):
            if str(ev.get("type", "")).lower() != "dropout":
                continue
            if int(ev.get("agent_id", -1)) != int(agent_id):
                continue
            if str(ev.get("mode", "")).lower() == "offline":
                return True
        return False

    def offline_mask(self, n_agents: int, t: float) -> np.ndarray:
        mask = np.zeros(int(n_agents), dtype=bool)
        if not self.enabled:
            return mask
        for i in range(int(n_agents)):
            mask[i] = self.is_agent_offline(i, t)
        return mask
