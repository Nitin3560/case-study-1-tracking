from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np

def formation_offsets(cfg: Dict[str, Any], n: int) -> np.ndarray:
    f = cfg["formation"]
    spacing = float(f["spacing_m"])
    if f["type"] == "square":
        # For n=4: corners. For others: fall back line.
        if n == 4:
            return np.array([
                [-spacing/2, -spacing/2, 0.0],
                [-spacing/2, +spacing/2, 0.0],
                [+spacing/2, -spacing/2, 0.0],
                [+spacing/2, +spacing/2, 0.0],
            ])
    # line
    offs = np.zeros((n, 3))
    for i in range(n):
        offs[i] = np.array([(i - (n-1)/2) * spacing, 0.0, 0.0])
    return offs


def p_ref(cfg: Dict[str, Any], t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pos, vel) reference for the swarm centroid in world frame."""
    tr = cfg["trajectory"]
    typ = tr["type"]
    R = float(tr["radius_m"])
    z = float(tr["altitude_m"])
    T = float(tr["period_s"])
    cx, cy = tr["center_xy"]
    w = 2.0 * np.pi / T

    if typ == "circle":
        x = cx + R * np.cos(w * t)
        y = cy + R * np.sin(w * t)
        vx = -R * w * np.sin(w * t)
        vy = +R * w * np.cos(w * t)
        return np.array([x, y, z]), np.array([vx, vy, 0.0])

    if typ == "lemniscate":
        # simple figure-8
        x = cx + R * np.sin(w * t)
        y = cy + R * np.sin(w * t) * np.cos(w * t)
        vx = R * w * np.cos(w * t)
        vy = R * w * (np.cos(2*w*t))
        return np.array([x, y, z]), np.array([vx, vy, 0.0])

    if typ == "lawnmower":
        # back-and-forth in x with slow y progression
        L = 2 * R
        speed = L / (T / 4)
        phase = (t % T) / T
        y = cy + (phase * 2 - 1) * R
        if int(phase * 4) % 2 == 0:
            x = cx - R + speed * (t % (T/4))
            vx = speed
        else:
            x = cx + R - speed * (t % (T/4))
            vx = -speed
        return np.array([x, y, z]), np.array([vx, 0.0, 0.0])

    raise ValueError(f"Unknown trajectory type: {typ}")
def waypoint_by_time(cfg: Dict[str, Any], t: float) -> np.ndarray:
    """
    Open-loop target selection by time.
    Supports:
      - cfg["controllers"]["openloop"]["waypoint_dt_s"]   (new)
      - cfg["controller"]["openloop"]["steps_per_waypoint"] (legacy/simple)
    """
    # New schema
    dt = None
    ctrls = cfg.get("controllers", {}) or {}
    if "openloop" in ctrls and isinstance(ctrls["openloop"], dict):
        dt = ctrls["openloop"].get("waypoint_dt_s", None)

    # Legacy schema
    if dt is None:
        ctrl = cfg.get("controller", {}) or {}
        ol = ctrl.get("openloop", {}) or {}
        spw = ol.get("steps_per_waypoint", None)
        if spw is not None:
            ctrl_hz = float(cfg["sim"]["ctrl_hz"])
            dt = float(spw) / ctrl_hz

    # Default fallback
    if dt is None:
        dt = 0.5  # seconds

    k = int(t // float(dt))
    tw = k * float(dt)
    pos, _ = p_ref(cfg, tw)
    return pos


from typing import Any, Dict, Tuple
import numpy as np

def formation_offsets(cfg: Dict[str, Any], n: int) -> np.ndarray:
    """
    Returns per-drone formation offsets (N,3).

    Supports:
      - cfg["formation"]["type"], cfg["formation"]["spacing_m"]  (new schema)
      - If cfg["formation"] missing (your current scenario_default.yaml),
        falls back to a safe default that avoids collisions.
    """
    f = cfg.get("formation", None)

    # ---- Safe defaults if formation section is missing ----
    # Choose a formation that prevents drones from targeting the exact same point.
    # Default spacing is conservative for Crazyflie scale in pybullet.
    if f is None:
        spacing = float(cfg.get("sim", {}).get("formation_spacing_m", 0.6))
        # Use square for n=4 (nice visuals), otherwise a line
        f = {"type": "square" if n == 4 else "line", "spacing_m": spacing}

    spacing = float(f.get("spacing_m", 0.6))
    ftype = str(f.get("type", "line")).lower()

    if ftype == "square" and n == 4:
        return np.array([
            [-spacing/2, -spacing/2, 0.0],
            [-spacing/2, +spacing/2, 0.0],
            [+spacing/2, -spacing/2, 0.0],
            [+spacing/2, +spacing/2, 0.0],
        ], dtype=float)

    # Default: line formation
    offs = np.zeros((n, 3), dtype=float)
    for i in range(n):
        offs[i] = np.array([(i - (n-1)/2) * spacing, 0.0, 0.0], dtype=float)
    return offs
