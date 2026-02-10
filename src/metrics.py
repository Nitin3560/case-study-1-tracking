from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np


# ----------------------------
# Backward-compatible log type
# ----------------------------
@dataclass
class MetricsLog:
    ts: List[float]
    mean_err: List[float]
    max_err: List[float]
    formation_err: List[float]
    connectivity_ok: List[float]


# -----------------------------------------
# New (optional) richer log for paper clarity
# -----------------------------------------
@dataclass
class MetricsLogV2:
    """
    Optional richer log that makes "nominal vs commanded reference" explicit.

    - *_nominal: error against mission reference (what you *evaluate*)
    - *_cmd:     error against commanded reference (what controller *tracks*)

    You can populate either/both depending on the experiment.
    """
    ts: List[float]

    # tracking error vs nominal reference
    mean_err_nominal: List[float]
    max_err_nominal: List[float]

    # tracking error vs commanded reference (optional)
    mean_err_cmd: Optional[List[float]] = None
    max_err_cmd: Optional[List[float]] = None

    # formation + connectivity (same meaning as before)
    formation_err: Optional[List[float]] = None
    formation_err_relative: Optional[List[float]] = None
    connectivity_ok: Optional[List[float]] = None

    # optional debugging channels (not required)
    extra: Optional[Dict[str, List[float]]] = None


# ----------------------------
# Connectivity + formation
# ----------------------------
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


def adjacency_matrix(pos: np.ndarray, comm_range_m: float, use_3d: bool = False) -> np.ndarray:
    """
    Build an undirected adjacency matrix (N,N) where A[i,j]=1 if j is within comm range of i.
    """
    pos = np.asarray(pos, dtype=float)
    assert pos.ndim == 2 and pos.shape[1] == 3
    n = pos.shape[0]
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if use_3d:
                d = np.linalg.norm(pos[i] - pos[j])
            else:
                d = np.linalg.norm(pos[i, :2] - pos[j, :2])
            if d <= comm_range_m:
                A[i, j] = 1
    return A


def connected_components(adj: np.ndarray) -> List[List[int]]:
    """
    Return list of connected components from adjacency matrix (undirected assumed).
    """
    n = adj.shape[0]
    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = np.where(adj[u] > 0)[0]
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def avg_shortest_path(adj: np.ndarray) -> float:
    """
    Average shortest-path length over connected pairs only.
    Returns np.inf if no connected pairs exist.
    """
    n = adj.shape[0]
    if n <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        # BFS
        dist = -np.ones(n, dtype=int)
        dist[i] = 0
        queue = [i]
        for u in queue:
            for v in np.where(adj[u] > 0)[0]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for j in range(i + 1, n):
            if dist[j] > 0:
                total += dist[j]
                count += 1
    return float(total / count) if count > 0 else float("inf")


def throughput_ratio(adj: np.ndarray, demand: np.ndarray) -> float:
    """
    Simple throughput proxy: fraction of traffic demand pairs that are connected.
    demand: (N,N) symmetric, nonnegative, zeros on diagonal.
    """
    n = adj.shape[0]
    comps = connected_components(adj)
    comp_id = np.full(n, -1, dtype=int)
    for k, comp in enumerate(comps):
        for i in comp:
            comp_id[i] = k
    total = float(np.sum(demand) / 2.0)
    if total <= 0:
        return 0.0
    ok = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if comp_id[i] == comp_id[j]:
                ok += float(demand[i, j])
    return ok / total


def control_overhead_ratio(adj: np.ndarray) -> float:
    """
    Control overhead proxy: average neighbor fraction per drone.
    """
    n = adj.shape[0]
    if n <= 1:
        return 0.0
    neighbors = np.sum(adj > 0, axis=1)
    return float(np.mean(neighbors / max(n - 1, 1)))


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


# -----------------------------------------
# New: explicit tracking error computations
# -----------------------------------------
def tracking_errors_xy(pos: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and max XY tracking error for a swarm.

    pos: (N,3) actual positions
    ref: (N,3) reference positions (nominal mission ref OR commanded ref)
    Returns: (mean_xy_err, max_xy_err)
    """
    pos = np.asarray(pos, dtype=float)
    ref = np.asarray(ref, dtype=float)
    assert pos.shape == ref.shape and pos.shape[1] == 3, f"shape mismatch pos={pos.shape}, ref={ref.shape}"
    e = (pos - ref)[:, :2]
    norms = np.linalg.norm(e, axis=1)
    return float(np.mean(norms)), float(np.max(norms))


def tracking_errors_com_xy(pos: np.ndarray, ref_com: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean and max XY error of drone positions relative to a reference COM trajectory.
    Useful when you only have COM reference (1,3) or (3,).

    pos: (N,3)
    ref_com: (3,) or (1,3) or (T,3) if you pass single timestep slice
    """
    pos = np.asarray(pos, dtype=float)
    ref_com = np.asarray(ref_com, dtype=float).reshape(-1)
    assert pos.ndim == 2 and pos.shape[1] == 3 and ref_com.shape[0] == 3
    com = pos.mean(axis=0)
    e_xy = (com - ref_com)[:2]
    # mean/max are same for COM scalar; return as (abs, abs)
    val = float(np.linalg.norm(e_xy))
    return val, val


# ---------------------------------------------------
# New: temporal (time-series) summary helper functions
# ---------------------------------------------------
@dataclass
class TemporalSummary:
    """
    Simple transient/steady-state summary for a 1D error time series.
    """
    peak: float
    rms: float
    steady_mean: float
    steady_max: float
    settling_time_s: Optional[float]


def temporal_summary(
    ts: np.ndarray,
    err: np.ndarray,
    *,
    steady_start_s: Optional[float] = None,
    settle_band: float = 0.1,
    settle_hold_s: float = 1.0,
) -> TemporalSummary:
    """
    Summarize temporal structure of an error signal.

    Args:
      ts: (T,) timestamps in seconds, increasing
      err: (T,) nonnegative error magnitude over time (e.g., mean XY error each tick)
      steady_start_s: if provided, steady-state window begins at this time
      settle_band: band for settling, expressed as a fraction of peak (e.g., 0.1 => within 10% of peak)
      settle_hold_s: must remain within band for this duration to count as settled

    Notes:
      - This is conservative and robust for reviewer-friendly reporting.
      - If err never settles, settling_time_s=None.
    """
    ts = np.asarray(ts, dtype=float).reshape(-1)
    err = np.asarray(err, dtype=float).reshape(-1)
    assert ts.shape == err.shape and ts.size > 1, f"ts/err shape mismatch {ts.shape} vs {err.shape}"
    assert np.all(np.diff(ts) >= 0.0), "ts must be nondecreasing"
    err = np.maximum(err, 0.0)

    peak = float(np.max(err))
    rms = float(np.sqrt(np.mean(err ** 2)))

    # steady window
    if steady_start_s is None:
        # default: last 25% of samples
        idx0 = int(0.75 * len(ts))
    else:
        idx0 = int(np.searchsorted(ts, float(steady_start_s), side="left"))
        idx0 = max(0, min(idx0, len(ts) - 1))

    steady = err[idx0:]
    steady_mean = float(np.mean(steady))
    steady_max = float(np.max(steady))

    # settling time: first time after which signal stays within band for settle_hold_s
    # band is defined relative to peak; if peak=0 -> settled at t0
    if peak <= 1e-12:
        settling_time_s = float(ts[0])
    else:
        band_val = float(settle_band) * peak
        # Find earliest t such that for next hold window, err <= band_val
        settling_time_s = None
        for k in range(len(ts)):
            t0 = ts[k]
            t1 = t0 + float(settle_hold_s)
            j = int(np.searchsorted(ts, t1, side="right"))
            if j <= k:
                continue
            if np.all(err[k:j] <= band_val):
                settling_time_s = float(t0)
                break

    return TemporalSummary(
        peak=peak,
        rms=rms,
        steady_mean=steady_mean,
        steady_max=steady_max,
        settling_time_s=settling_time_s,
    )
