from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np


def formation_offsets(cfg: Dict[str, Any], n: int) -> np.ndarray:
    f = cfg.get("formation", None)
    if f is None:
        spacing = float((cfg.get("formation", {}) or {}).get("spacing_m", 0.6))
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

    offs = np.zeros((n, 3), dtype=float)
    for i in range(n):
        offs[i] = np.array([(i - (n - 1) / 2) * spacing, 0.0, 0.0], dtype=float)
    return offs


def p_ref(cfg: Dict[str, Any], t: float) -> Tuple[np.ndarray, np.ndarray]:
    tr = cfg["trajectory"]
    typ = str(tr["type"]).lower()
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
        return np.array([x, y, z], dtype=float), np.array([vx, vy, 0.0], dtype=float)

    if typ == "lemniscate":
        x = cx + R * np.sin(w * t)
        y = cy + R * np.sin(w * t) * np.cos(w * t)
        vx = R * w * np.cos(w * t)
        vy = R * w * (np.cos(2 * w * t))
        return np.array([x, y, z], dtype=float), np.array([vx, vy, 0.0], dtype=float)

    if typ == "lawnmower":
        L = 2 * R
        speed = L / (T / 4)
        phase = (t % T) / T
        y = cy + (phase * 2 - 1) * R
        if int(phase * 4) % 2 == 0:
            x = cx - R + speed * (t % (T / 4))
            vx = speed
        else:
            x = cx + R - speed * (t % (T / 4))
            vx = -speed
        return np.array([x, y, z], dtype=float), np.array([vx, 0.0, 0.0], dtype=float)

    raise ValueError(f"Unknown trajectory type: {typ}")
