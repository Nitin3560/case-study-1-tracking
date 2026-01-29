from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class MetricsLog:
    ts: List[float]
    mean_err: List[float]
    max_err: List[float]
    formation_err: List[float]
    connectivity_ok: List[float]


def connectivity_rate(pos: np.ndarray, comm_range_m: float = 10.0, use_3d: bool = False) -> float:
    """
    Fraction of drones that have at least one neighbor within communication range.

    pos: (N,3) positions
    comm_range_m: range in meters
    use_3d: if False, range uses XY only; if True, full XYZ.
    """
    pos = np.asarray(pos, dtype=float)
    assert pos.ndim == 2 and pos.shape[1] == 3, f"pos must be (N,3), got {pos.shape}"

    n = pos.shape[0]
    if n <= 1:
        return 1.0

    ok = 0
    for i in range(n):
        has_neighbor = False
        for j in range(n):
            if i == j:
                continue
            if use_3d:
                d = np.linalg.norm(pos[i] - pos[j])
            else:
                d = np.linalg.norm(pos[i, :2] - pos[j, :2])
            if d <= comm_range_m:
                has_neighbor = True
                break
        ok += 1 if has_neighbor else 0

    return ok / n


def formation_error(pos: np.ndarray, desired: np.ndarray) -> float:
    """
    Mean XY formation error against desired (N,3).
    """
    pos = np.asarray(pos, dtype=float)
    desired = np.asarray(desired, dtype=float)
    assert pos.shape == desired.shape and pos.shape[1] == 3, f"shape mismatch pos={pos.shape}, desired={desired.shape}"
    return float(np.mean(np.linalg.norm((pos - desired)[:, :2], axis=1)))


def formation_error_relative(pos: np.ndarray, offs: np.ndarray) -> float:
    """
    Mean XY formation error using COM-relative offsets.
    pos: (N,3) absolute positions
    offs: (N,3) desired offsets around the centroid/COM
    """
    pos = np.asarray(pos, dtype=float)
    offs = np.asarray(offs, dtype=float)
    assert pos.shape == offs.shape and pos.shape[1] == 3
    com = pos.mean(axis=0)
    rel = pos - com
    return float(np.mean(np.linalg.norm((rel - offs)[:, :2], axis=1)))
