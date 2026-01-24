# src/controllers/baseline_openloop.py

from __future__ import annotations
import numpy as np


class OpenLoopWaypointBaseline:
    """
    TRUE open-loop baseline (IEEE-correct):
    - No state feedback
    - Ignores position/velocity error
    - Commands constant hover RPM (drifts under wind/disturbances)

    IMPORTANT:
    run_one.py instantiates this as:
        OpenLoopWaypointBaseline(cfg, hover_rpm=float(env.HOVER_RPM))
    so __init__ MUST accept hover_rpm.
    """

    def __init__(self, cfg, hover_rpm: float, **kwargs):
        self.cfg = cfg
        self.hover_rpm = float(hover_rpm)

    def compute_rpms(self, state, target_pos):
        # Ignores both 'state' and 'target_pos' to remain open-loop.
        return np.array([self.hover_rpm]*4, dtype=float)
