# Case Study 1: Configs + Controller Definitions (Single File)

This file consolidates the requested YAML configs and controller code entry points.

## YAML Configs

### `configs/scenario_case_study1.yaml`
```yaml
sim:
  drone_model: "cf2x"
  physics: "pyb"
  gui: false

  # Physics & control
  freq_hz: 240
  ctrl_hz: 48
  duration_s: 55.0
  num_drones: 4
  seed: 1
  debug_obs: false

  # Actuator limits (used for saturation-aware PID)
  rpm_min: 0
  rpm_max: 18000


trajectory:
  type: "circle"
  radius_m: 8.0
  period_s: 18.0
  altitude_m: 2.0
  center_xy: [0.0, 0.0]


formation:
  type: "square"
  spacing_m: 2.0


disturbance:
  wind:
    mode: "phased"
    phases:
    - t0_s: 0.0
      t1_s: 15.0
      constant_force_n: [0.05, 0.0, 0.0]
      gust_force_n_max: 0.09
      gust_interval_s: [0.8, 1.6]
      gust_duration_s: [0.4, 0.8]

    - t0_s: 15.0
      t1_s: 40.0
      constant_force_n: [0.08, 0.00, 0.0]
      gust_force_n_max: 0.14
      gust_interval_s: [0.8, 1.8]
      gust_duration_s: [0.6, 1.2]

    - t0_s: 40.0
      t1_s: 55.0
      constant_force_n: [0.06, 0.00, 0.0]
      gust_force_n_max: 0.11
      gust_interval_s: [1.6, 3.0]
      gust_duration_s: [0.5, 1.1]

  gps_drift:
    enabled: True
    random_walk_std_m_per_s: 0.02
    noise_std_m: 0.04


metrics:
  steady_state_after_s: 40.0


network:
  num_flows: 6
  demand_range: [0.5, 2.0]


tasks:
  reassign_margin_m: 0.5


controllers:

  sensing:
    openloop: "gps"
    pid: "gps"
    agentic: "gps"

  openloop:
    openloop_delay_s: 0.2

  pid:
    kp: 1.15
    kd: 0.50
    ki: 0.08
    v_max: 5.8

    # Integrator hysteresis gates (matches pid_core.py)
    int_gate_on_err_m: 2.0
    int_gate_off_err_m: 3.0
    int_gate_on_vel_mps: 4.0
    int_gate_off_vel_mps: 5.0

    int_leak: 0.995
    int_clamp: 1.5

    # D-term filtering (matches pid_core.py)
    d_mode: "vel"
    d_tau_s: 0.05

    int_sat_hold: false
    int_sat_leak: 1.0

  agentic:
    # Time horizon: low-bandwidth supervisory layer
    lookahead_s: 1.0
    error_thresh_m: 1.3
    max_ref_shift_m: 0.20
    authority_alpha: 0.0
    authority_ramp_mult: 2.5
    error_gate_m: 0.6
    error_scale_m: 1.5
    comm_bias_m: 0.18
    comm_min_neighbors: 1
    rpm_eps: 1e-3

    supervisor:
      hz: 4.0
      hold_s: 1.2
      err_thresh_m: 1.4
      err_recover_slope_mps: 0.12
      conn_thresh: 0.7
      form_thresh: 1.15
      ref_shift_rate_mps: 0.10
      ref_shift_max_m: 0.20
      smooth_alpha_active: 0.5
      formation_scale_min: 0.9
      formation_scale_max: 1.15
      sat_hold_thresh: 0.55
      int_leak_override: 0.985
      drift_disable_phase_ge: 2
      ref_shift_decay_mps: 0.05
      err_hard_m: 3.5
      hard_hold_s: 1.5
```

### `configs/scenario_paper_figure.yaml`
```yaml
sim:
  drone_model: "cf2x"
  physics: "pyb"
  gui: false

  # Physics & control
  freq_hz: 240
  ctrl_hz: 48
  duration_s: 60.0
  num_drones: 4
  seed: 1
  debug_obs: false

  # Actuator limits
  rpm_min: 0
  rpm_max: 18000


trajectory:
  type: "circle"
  radius_m: 8.0
  period_s: 18.0
  altitude_m: 2.0
  center_xy: [0.0, 0.0]


formation:
  type: "square"
  spacing_m: 2.0


disturbance:
  wind:
    mode: "phased"
    phases:
    - t0_s: 0.0
      t1_s: 20.0
      constant_force_n: [0.06, 0.0, 0.0]
      gust_force_n_max: 0.10
      gust_interval_s: [0.7, 1.5]
      gust_duration_s: [0.4, 0.8]

    - t0_s: 20.0
      t1_s: 45.0
      constant_force_n: [0.09, 0.0, 0.0]
      gust_force_n_max: 0.16
      gust_interval_s: [0.9, 2.0]
      gust_duration_s: [0.6, 1.2]

    - t0_s: 45.0
      t1_s: 60.0
      constant_force_n: [0.07, 0.0, 0.0]
      gust_force_n_max: 0.12
      gust_interval_s: [1.4, 2.8]
      gust_duration_s: [0.5, 1.1]

  gps_drift:
    enabled: True
    random_walk_std_m_per_s: 0.02
    noise_std_m: 0.04


metrics:
  steady_state_after_s: 45.0


controllers:

  sensing:
    openloop: "gps"
    pid: "gps"
    agentic: "gps"

  openloop:
    openloop_delay_s: 0.25

  pid:
    kp: 1.2
    kd: 0.55
    ki: 0.08
    v_max: 6.0

    int_gate_on_err_m: 2.0
    int_gate_off_err_m: 3.0
    int_gate_on_vel_mps: 4.0
    int_gate_off_vel_mps: 5.0
    int_leak: 0.995
    int_clamp: 1.5
    d_mode: "vel"
    d_tau_s: 0.05
    int_sat_hold: false
    int_sat_leak: 1.0

  agentic:
    lookahead_s: 1.0
    error_thresh_m: 1.3
    max_ref_shift_m: 0.20
    authority_alpha: 0.0
    authority_ramp_mult: 2.5
    error_gate_m: 0.6
    error_scale_m: 1.5
    comm_bias_m: 0.18
    comm_min_neighbors: 1
    rpm_eps: 1e-3

    supervisor:
      hz: 4.0
      hold_s: 1.2
      err_thresh_m: 1.4
      err_recover_slope_mps: 0.12
      conn_thresh: 0.7
      form_thresh: 1.15
      ref_shift_rate_mps: 0.10
      ref_shift_max_m: 0.20
      smooth_alpha_active: 0.65
      formation_scale_min: 0.9
      formation_scale_max: 1.15
      sat_hold_thresh: 0.55
      sat_release_thresh: 0.35
      int_leak_override: 0.985
      drift_disable_phase_ge: 2
      ref_shift_decay_mps: 0.05
      err_hard_m: 3.5
      hard_hold_s: 1.5
      recovery_ref_shift_rate_mps: 0.06
      phase_boost_s: 4.0
      phase_boost_rate_mps: 0.12
      phase_boost_smooth_alpha: 0.85
```

### `configs/scenario_disturbance_low.yaml`
```yaml
sim:
  drone_model: "cf2x"
  physics: "pyb"
  gui: false
  freq_hz: 240
  ctrl_hz: 48
  duration_s: 60.0
  num_drones: 4
  seed: 1
  debug_obs: false
  rpm_min: 0
  rpm_max: 18000

trajectory:
  type: "circle"
  radius_m: 8.0
  period_s: 18.0
  altitude_m: 2.0
  center_xy: [0.0, 0.0]

formation:
  type: "square"
  spacing_m: 2.0

disturbance:
  wind:
    mode: "phased"
    phases:
    - t0_s: 0.0
      t1_s: 20.0
      constant_force_n: [0.03, 0.0, 0.0]
      gust_force_n_max: 0.06
      gust_interval_s: [0.8, 1.6]
      gust_duration_s: [0.4, 0.8]
    - t0_s: 20.0
      t1_s: 45.0
      constant_force_n: [0.05, 0.0, 0.0]
      gust_force_n_max: 0.08
      gust_interval_s: [1.0, 2.0]
      gust_duration_s: [0.6, 1.2]
    - t0_s: 45.0
      t1_s: 60.0
      constant_force_n: [0.04, 0.0, 0.0]
      gust_force_n_max: 0.07
      gust_interval_s: [1.6, 3.0]
      gust_duration_s: [0.5, 1.1]

  gps_drift:
    enabled: True
    random_walk_std_m_per_s: 0.01
    noise_std_m: 0.02

metrics:
  steady_state_after_s: 45.0

controllers:
  sensing:
    openloop: "gps"
    pid: "gps"
    agentic: "gps"
  openloop:
    openloop_delay_s: 0.25
  pid:
    kp: 1.2
    kd: 0.55
    ki: 0.08
    v_max: 6.0
    int_gate_on_err_m: 2.0
    int_gate_off_err_m: 3.0
    int_gate_on_vel_mps: 4.0
    int_gate_off_vel_mps: 5.0
    int_leak: 0.995
    int_clamp: 1.5
    d_mode: "vel"
    d_tau_s: 0.05
    int_sat_hold: false
    int_sat_leak: 1.0
  agentic:
    lookahead_s: 1.0
    error_thresh_m: 1.3
    max_ref_shift_m: 0.20
    authority_alpha: 0.0
    authority_ramp_mult: 2.5
    error_gate_m: 0.6
    error_scale_m: 1.5
    comm_bias_m: 0.18
    comm_min_neighbors: 1
    rpm_eps: 1e-3
    supervisor:
      hz: 4.0
      hold_s: 1.2
      err_thresh_m: 1.4
      err_recover_slope_mps: 0.12
      conn_thresh: 0.7
      form_thresh: 1.15
      ref_shift_rate_mps: 0.10
      ref_shift_max_m: 0.20
      smooth_alpha_active: 0.65
      formation_scale_min: 0.9
      formation_scale_max: 1.15
      sat_hold_thresh: 0.55
      sat_release_thresh: 0.35
      int_leak_override: 0.985
      drift_disable_phase_ge: 2
      ref_shift_decay_mps: 0.05
      err_hard_m: 3.5
      hard_hold_s: 1.5
      recovery_ref_shift_rate_mps: 0.06
```

### `configs/scenario_disturbance_medium.yaml`
```yaml
sim:
  drone_model: "cf2x"
  physics: "pyb"
  gui: false
  freq_hz: 240
  ctrl_hz: 48
  duration_s: 60.0
  num_drones: 4
  seed: 1
  debug_obs: false
  rpm_min: 0
  rpm_max: 18000

trajectory:
  type: "circle"
  radius_m: 8.0
  period_s: 18.0
  altitude_m: 2.0
  center_xy: [0.0, 0.0]

formation:
  type: "square"
  spacing_m: 2.0

disturbance:
  wind:
    mode: "phased"
    phases:
    - t0_s: 0.0
      t1_s: 20.0
      constant_force_n: [0.06, 0.0, 0.0]
      gust_force_n_max: 0.10
      gust_interval_s: [0.8, 1.6]
      gust_duration_s: [0.4, 0.8]
    - t0_s: 20.0
      t1_s: 45.0
      constant_force_n: [0.08, 0.0, 0.0]
      gust_force_n_max: 0.14
      gust_interval_s: [0.9, 2.0]
      gust_duration_s: [0.6, 1.2]
    - t0_s: 45.0
      t1_s: 60.0
      constant_force_n: [0.07, 0.0, 0.0]
      gust_force_n_max: 0.11
      gust_interval_s: [1.4, 2.8]
      gust_duration_s: [0.5, 1.1]

  gps_drift:
    enabled: True
    random_walk_std_m_per_s: 0.02
    noise_std_m: 0.04

metrics:
  steady_state_after_s: 45.0

controllers:
  sensing:
    openloop: "gps"
    pid: "gps"
    agentic: "gps"
  openloop:
    openloop_delay_s: 0.25
  pid:
    kp: 1.2
    kd: 0.55
    ki: 0.08
    v_max: 6.0
    int_gate_on_err_m: 2.0
    int_gate_off_err_m: 3.0
    int_gate_on_vel_mps: 4.0
    int_gate_off_vel_mps: 5.0
    int_leak: 0.995
    int_clamp: 1.5
    d_mode: "vel"
    d_tau_s: 0.05
    int_sat_hold: false
    int_sat_leak: 1.0
  agentic:
    lookahead_s: 1.0
    error_thresh_m: 1.3
    max_ref_shift_m: 0.20
    authority_alpha: 0.0
    authority_ramp_mult: 2.5
    error_gate_m: 0.6
    error_scale_m: 1.5
    comm_bias_m: 0.18
    comm_min_neighbors: 1
    rpm_eps: 1e-3
    supervisor:
      hz: 4.0
      hold_s: 1.2
      err_thresh_m: 1.4
      err_recover_slope_mps: 0.12
      conn_thresh: 0.7
      form_thresh: 1.15
      ref_shift_rate_mps: 0.10
      ref_shift_max_m: 0.20
      smooth_alpha_active: 0.65
      formation_scale_min: 0.9
      formation_scale_max: 1.15
      sat_hold_thresh: 0.55
      sat_release_thresh: 0.35
      int_leak_override: 0.985
      drift_disable_phase_ge: 2
      ref_shift_decay_mps: 0.05
      err_hard_m: 3.5
      hard_hold_s: 1.5
      recovery_ref_shift_rate_mps: 0.06
```

### `configs/scenario_disturbance_high.yaml`
```yaml
sim:
  drone_model: "cf2x"
  physics: "pyb"
  gui: false
  freq_hz: 240
  ctrl_hz: 48
  duration_s: 60.0
  num_drones: 4
  seed: 1
  debug_obs: false
  rpm_min: 0
  rpm_max: 18000

trajectory:
  type: "circle"
  radius_m: 8.0
  period_s: 18.0
  altitude_m: 2.0
  center_xy: [0.0, 0.0]

formation:
  type: "square"
  spacing_m: 2.0

disturbance:
  wind:
    mode: "phased"
    phases:
    - t0_s: 0.0
      t1_s: 20.0
      constant_force_n: [0.09, 0.0, 0.0]
      gust_force_n_max: 0.14
      gust_interval_s: [0.7, 1.5]
      gust_duration_s: [0.4, 0.8]
    - t0_s: 20.0
      t1_s: 45.0
      constant_force_n: [0.12, 0.0, 0.0]
      gust_force_n_max: 0.18
      gust_interval_s: [0.8, 1.8]
      gust_duration_s: [0.6, 1.2]
    - t0_s: 45.0
      t1_s: 60.0
      constant_force_n: [0.10, 0.0, 0.0]
      gust_force_n_max: 0.15
      gust_interval_s: [1.2, 2.4]
      gust_duration_s: [0.5, 1.1]

  gps_drift:
    enabled: True
    random_walk_std_m_per_s: 0.03
    noise_std_m: 0.05

metrics:
  steady_state_after_s: 45.0

controllers:
  sensing:
    openloop: "gps"
    pid: "gps"
    agentic: "gps"
  openloop:
    openloop_delay_s: 0.25
  pid:
    kp: 1.2
    kd: 0.55
    ki: 0.08
    v_max: 6.0
    int_gate_on_err_m: 2.0
    int_gate_off_err_m: 3.0
    int_gate_on_vel_mps: 4.0
    int_gate_off_vel_mps: 5.0
    int_leak: 0.995
    int_clamp: 1.5
    d_mode: "vel"
    d_tau_s: 0.05
    int_sat_hold: false
    int_sat_leak: 1.0
  agentic:
    lookahead_s: 1.0
    error_thresh_m: 1.3
    max_ref_shift_m: 0.20
    authority_alpha: 0.0
    authority_ramp_mult: 2.5
    error_gate_m: 0.6
    error_scale_m: 1.5
    comm_bias_m: 0.18
    comm_min_neighbors: 1
    rpm_eps: 1e-3
    supervisor:
      hz: 4.0
      hold_s: 1.2
      err_thresh_m: 1.4
      err_recover_slope_mps: 0.12
      conn_thresh: 0.7
      form_thresh: 1.15
      ref_shift_rate_mps: 0.10
      ref_shift_max_m: 0.20
      smooth_alpha_active: 0.65
      formation_scale_min: 0.9
      formation_scale_max: 1.15
      sat_hold_thresh: 0.55
      sat_release_thresh: 0.35
      int_leak_override: 0.985
      drift_disable_phase_ge: 2
      ref_shift_decay_mps: 0.05
      err_hard_m: 3.5
      hard_hold_s: 1.5
      recovery_ref_shift_rate_mps: 0.06
```

### `configs/scenario_disturbance_very_high.yaml`
```yaml
sim:
  drone_model: "cf2x"
  physics: "pyb"
  gui: false
  freq_hz: 240
  ctrl_hz: 48
  duration_s: 60.0
  num_drones: 4
  seed: 1
  debug_obs: false
  rpm_min: 0
  rpm_max: 18000

trajectory:
  type: "circle"
  radius_m: 8.0
  period_s: 18.0
  altitude_m: 2.0
  center_xy: [0.0, 0.0]

formation:
  type: "square"
  spacing_m: 2.0

disturbance:
  wind:
    mode: "phased"
    phases:
    - t0_s: 0.0
      t1_s: 20.0
      constant_force_n: [0.12, 0.0, 0.0]
      gust_force_n_max: 0.18
      gust_interval_s: [0.6, 1.4]
      gust_duration_s: [0.4, 0.8]
    - t0_s: 20.0
      t1_s: 45.0
      constant_force_n: [0.15, 0.0, 0.0]
      gust_force_n_max: 0.22
      gust_interval_s: [0.7, 1.6]
      gust_duration_s: [0.6, 1.2]
    - t0_s: 45.0
      t1_s: 60.0
      constant_force_n: [0.13, 0.0, 0.0]
      gust_force_n_max: 0.20
      gust_interval_s: [1.0, 2.0]
      gust_duration_s: [0.5, 1.1]

  gps_drift:
    enabled: True
    random_walk_std_m_per_s: 0.04
    noise_std_m: 0.06

metrics:
  steady_state_after_s: 45.0

controllers:
  sensing:
    openloop: "gps"
    pid: "gps"
    agentic: "gps"
  openloop:
    openloop_delay_s: 0.25
  pid:
    kp: 1.2
    kd: 0.55
    ki: 0.08
    v_max: 6.0
    int_gate_on_err_m: 2.0
    int_gate_off_err_m: 3.0
    int_gate_on_vel_mps: 4.0
    int_gate_off_vel_mps: 5.0
    int_leak: 0.995
    int_clamp: 1.5
    d_mode: "vel"
    d_tau_s: 0.05
    int_sat_hold: false
    int_sat_leak: 1.0
  agentic:
    lookahead_s: 1.0
    error_thresh_m: 1.0
    max_ref_shift_m: 0.35
    authority_alpha: 0.0
    authority_ramp_mult: 2.5
    error_gate_m: 0.6
    error_scale_m: 1.5
    comm_bias_m: 0.18
    comm_min_neighbors: 1
    rpm_eps: 1e-3
    supervisor:
      hz: 5.0
      hold_s: 0.8
      err_thresh_m: 1.1
      err_recover_slope_mps: 0.08
      conn_thresh: 0.55
      form_thresh: 1.35
      ref_shift_rate_mps: 0.18
      ref_shift_max_m: 0.35
      smooth_alpha_active: 0.8
      formation_scale_min: 0.8
      formation_scale_max: 1.25
      sat_hold_thresh: 0.5
      sat_release_thresh: 0.3
      int_leak_override: 0.985
      drift_disable_phase_ge: 2
      ref_shift_decay_mps: 0.03
      err_hard_m: 5.0
      hard_hold_s: 1.0
      recovery_ref_shift_rate_mps: 0.10
```

## Controller Code (Key Functions + Full Files)

### `src/controllers/openloop.py`
Key function: `OpenLoopController.compute_rpms(...)`
```python
from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


class OpenLoopController:
    """
    Open-loop (feedforward) baseline.

    IMPORTANT:
    - This controller MUST NOT use measured state feedback.
    - It consumes only the reference trajectory (target_pos, target_vel, yaw_des)
      and produces motor RPMs.

    Implementation detail:
    - DSLPIDControl is inherently a closed-loop controller, but we use it here only
      as a motor-command mapping utility by feeding it a *synthetic* "current state"
      derived purely from the reference:
          cur_pos  = target_pos
          cur_vel  = target_vel
          cur_quat = identity (assume level)
          cur_ang_vel = 0
    - This prevents any real feedback correction while keeping the RPM interface.
    """

    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        self.ctrl = DSLPIDControl(drone_model=drone_model)

        self._ref_queue = []

    def reset(self):
        self._ref_queue = []

    def _delay_steps(self) -> int:
        ol_cfg = (self.cfg.get("controllers", {}) or {}).get("openloop", {}) or {}
        delay_s = float(ol_cfg.get("openloop_delay_s", 0.0))
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])
        return max(0, int(round(delay_s / max(dt, 1e-9))))

    def compute_rpms(self, state, target_pos, target_vel, yaw_des: float = 0.0):
        # dt comes from config only (not from measured timing)
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])

        # Optional open-loop command delay (more realistic baseline)
        self._ref_queue.append((np.asarray(target_pos, dtype=float), np.asarray(target_vel, dtype=float)))
        delay = self._delay_steps()
        if delay > 0 and len(self._ref_queue) > delay:
            target_pos, target_vel = self._ref_queue.pop(0)
        else:
            target_pos = np.asarray(target_pos, dtype=float)
            target_vel = np.asarray(target_vel, dtype=float)

        # --- OPEN-LOOP: no outer-loop correction; use measured state for stabilization ---
        cur_pos = np.asarray(state[0:3], dtype=float)
        cur_quat = np.asarray(state[3:7], dtype=float)
        cur_vel = np.asarray(state[10:13], dtype=float)
        cur_ang_vel = np.asarray(state[13:16], dtype=float)

        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=np.asarray(target_pos, dtype=float),
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=np.asarray(target_vel, dtype=float),
        )
        return rpms
```

### `src/controllers/pid.py`
Key function: `PIDController.compute_rpms(...)`
```python
from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from src.control_core.pid_core import PIDCore, PIDConfig


class PIDController:
    """
    Cascaded closed-loop baseline (clean architecture).

    Outer loop (PIDCore):
      - uses position error (e) and velocity error (v_err)
      - outputs dv_cmd (a velocity correction)

    Inner loop (DSLPIDControl):
      - tracks ONLY velocity reference (target_vel + dv_cmd)
      - position error is intentionally zeroed by setting target_pos = cur_pos

    Saturation (IMPORTANT for fair comparison + anti-windup correctness):
      - computes graded sat_frac in [0,1] from RPM proximity to bounds
      - feeds sat_frac into PIDCore SAME TICK via 2-pass re-step
      - exposes self.sat_frac for logging by run_one.py
    """

    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        pid_cfg = (cfg.get("controllers", {}) or {}).get("pid", {}) or {}

        self.pid = PIDCore(
            PIDConfig(
                kp=float(pid_cfg.get("kp", 1.0)),
                ki=float(pid_cfg.get("ki", 0.0)),
                kd=float(pid_cfg.get("kd", 0.6)),

                v_max=float(pid_cfg.get("v_max", 3.0)),
                axis_mask=tuple(pid_cfg.get("axis_mask", (1.0, 1.0, 0.0))),
                clamp_mode=str(pid_cfg.get("clamp_mode", "xy")),
                v_max_xy=(None if pid_cfg.get("v_max_xy", None) is None else float(pid_cfg["v_max_xy"])),
                v_max_z=(None if pid_cfg.get("v_max_z", None) is None else float(pid_cfg["v_max_z"])),

                int_gate_on_err_m=float(pid_cfg.get("int_gate_on_err_m", 2.0)),
                int_gate_off_err_m=float(pid_cfg.get("int_gate_off_err_m", 3.0)),
                int_gate_on_vel_mps=float(pid_cfg.get("int_gate_on_vel_mps", 4.0)),
                int_gate_off_vel_mps=float(pid_cfg.get("int_gate_off_vel_mps", 5.0)),
                int_leak=float(pid_cfg.get("int_leak", 0.995)),
                int_clamp=float(pid_cfg.get("int_clamp", 2.0)),

                d_mode=str(pid_cfg.get("d_mode", "vel")),
                d_blend=float(pid_cfg.get("d_blend", 0.5)),
                d_tau_s=float(pid_cfg.get("d_tau_s", 0.05)),

                int_sat_hold=bool(pid_cfg.get("int_sat_hold", True)),
                sat_eps=float(pid_cfg.get("sat_eps", 1e-6)),
                int_sat_scale=bool(pid_cfg.get("int_sat_scale", True)),
                int_sat_scale_pow=float(pid_cfg.get("int_sat_scale_pow", 1.0)),
                int_sat_leak=float(pid_cfg.get("int_sat_leak", 1.0)),
            )
        )

        self.ctrl = DSLPIDControl(drone_model=drone_model)

        # --- Saturation detection (graded) ---
        # Prefer sim.rpm_min/rpm_max; allow override in controllers.pid
        self._rpm_min = pid_cfg.get("rpm_min", cfg.get("sim", {}).get("rpm_min", None))
        self._rpm_max = pid_cfg.get("rpm_max", cfg.get("sim", {}).get("rpm_max", None))

        # band defines what "near the limit" means (fraction of rpm range).
        # sat_frac=0 if all motors are >= band away from bounds; sat_frac->1 at bounds.
        self._sat_band_frac = float(pid_cfg.get("sat_band_frac", 0.05))
        self._sat_eps_rpm = float(pid_cfg.get("rpm_eps", 1e-6))

        # Exposed for logging
        self.sat_frac: float = float("nan")

    def reset(self):
        self.pid.reset()
        self.sat_frac = float("nan")
        if hasattr(self.ctrl, "reset"):
            try:
                self.ctrl.reset()
            except TypeError:
                pass

    def _sat_frac_from_rpms(self, rpms: np.ndarray) -> float:
        """
        Graded saturation fraction in [0,1] based on proximity to rpm bounds.
        Returns 0.0 if bounds unknown.
        """
        if self._rpm_min is None or self._rpm_max is None:
            return 0.0

        rmin = float(self._rpm_min)
        rmax = float(self._rpm_max)
        if rmax <= rmin:
            return 0.0

        rpms = np.asarray(rpms, dtype=float)
        rng = rmax - rmin
        band = max(rng * float(self._sat_band_frac), 1e-6)

        margin = np.minimum(rpms - rmin, rmax - rpms)

        # sat=0 when margin>=band ; sat=1 when margin<=0
        sat_each = 1.0 - np.clip((margin - self._sat_eps_rpm) / band, 0.0, 1.0)
        return float(np.clip(np.max(sat_each), 0.0, 1.0))

    def compute_rpms(
        self,
        state,
        target_pos,
        target_vel,
        *,
        integrate_z: bool = False,
        yaw_des: float = 0.0,
    ):
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])

        cur_pos = np.asarray(state[0:3], dtype=float)
        cur_quat = np.asarray(state[3:7], dtype=float)
        cur_vel = np.asarray(state[10:13], dtype=float)
        cur_ang_vel = np.asarray(state[13:16], dtype=float)

        target_pos = np.asarray(target_pos, dtype=float)
        target_vel = np.asarray(target_vel, dtype=float)

        e = target_pos - cur_pos
        v_err = target_vel - cur_vel

        # -----------------------------
        # SAME-TICK sat_frac (2-pass)
        # -----------------------------
        dv_cmd_1 = self.pid.step(e, v_err, dt, integrate_xy=not integrate_z)
        v_cmd_1 = target_vel + np.asarray(dv_cmd_1, dtype=float)

        inner_target_pos = cur_pos  # zero inner position error

        rpms_1, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=inner_target_pos,
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=v_cmd_1,
        )

        sat_frac_1 = self._sat_frac_from_rpms(rpms_1)
        self.pid.set_saturation(sat_frac_1)

        dv_cmd = self.pid.step(e, v_err, dt, integrate_xy=not integrate_z, saturated=sat_frac_1)
        v_cmd = target_vel + np.asarray(dv_cmd, dtype=float)

        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=inner_target_pos,
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=v_cmd,
        )

        self.sat_frac = self._sat_frac_from_rpms(rpms)
        self.pid.set_saturation(self.sat_frac)

        return rpms```

### `src/controllers/pid_agentic.py`
Key function: `PIDAgenticController.compute_rpms(...)`
```python
from __future__ import annotations

import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from src.control_core.pid_core import PIDCore, PIDConfig


def _clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


class PIDAgenticController:
    """
    PID + Agentic (clean + defensible):

    - Agentic layer is FEEDFORWARD w.r.t. measured state:
        * It does NOT use cur_pos/cur_vel/tracking error to compute its bias.
        * It only uses reference geometry: (p_ref_ahead - p_ref_now).
      (Optionally, it can be a *reference filter* if authority_alpha > 0.)

    - Outer loop (PIDCore) is the feedback controller for position tracking:
        e = p_cmd - cur_pos
        v_err = v_ref_now - cur_vel
        dv_cmd = PIDCore.step(e, v_err, dt, saturated=sat_frac)

    - Inner loop (DSLPIDControl) is used as an execution controller:
        We *zero inner position error* by setting target_pos = cur_pos.
        This strongly discourages an additional inner position feedback loop,
        but the exact internal behavior depends on DSLPIDControl implementation.

    Metrics:
      - This controller returns err_norm against the nominal mission reference p_ref_now (not p_cmd),
        to avoid the "moving goalpost" evaluation trap.

    Saturation (IMPORTANT):
      - We compute a graded sat_frac \in [0,1] from RPM proximity to bounds.
      - We FEED sat_frac into PIDCore each tick (so integrator gating/scaling is correct).
      - True actuator-limit detection based on motor model/RPM limits is ultimately downstream,
        but this is a robust and practical proxy when rpm_min/rpm_max are known.
    """

    def __init__(self, cfg, drone_model):
        self.cfg = cfg
        pid_cfg = (cfg.get("controllers", {}) or {}).get("pid", {}) or {}
        ag_cfg = (cfg.get("controllers", {}) or {}).get("agentic", {}) or {}

        # -------------------------
        # Outer PID core (PIDConfig)
        # -------------------------
        self.pid = PIDCore(
            PIDConfig(
                kp=float(pid_cfg.get("kp", 1.0)),
                ki=float(pid_cfg.get("ki", 0.0)),
                kd=float(pid_cfg.get("kd", 0.6)),

                v_max=float(pid_cfg.get("v_max", 3.0)),
                axis_mask=tuple(pid_cfg.get("axis_mask", (1.0, 1.0, 0.0))),
                clamp_mode=str(pid_cfg.get("clamp_mode", "xy")),
                v_max_xy=(None if pid_cfg.get("v_max_xy", None) is None else float(pid_cfg["v_max_xy"])),
                v_max_z=(None if pid_cfg.get("v_max_z", None) is None else float(pid_cfg["v_max_z"])),

                int_gate_on_err_m=float(pid_cfg.get("int_gate_on_err_m", 2.0)),
                int_gate_off_err_m=float(pid_cfg.get("int_gate_off_err_m", 3.0)),
                int_gate_on_vel_mps=float(pid_cfg.get("int_gate_on_vel_mps", 4.0)),
                int_gate_off_vel_mps=float(pid_cfg.get("int_gate_off_vel_mps", 5.0)),
                int_leak=float(pid_cfg.get("int_leak", 0.995)),
                int_clamp=float(pid_cfg.get("int_clamp", 2.0)),

                d_mode=str(pid_cfg.get("d_mode", "vel")),
                d_blend=float(pid_cfg.get("d_blend", 0.5)),
                d_tau_s=float(pid_cfg.get("d_tau_s", 0.05)),

                int_sat_hold=bool(pid_cfg.get("int_sat_hold", True)),
                sat_eps=float(pid_cfg.get("sat_eps", 1e-6)),
                int_sat_scale=bool(pid_cfg.get("int_sat_scale", True)),
                int_sat_scale_pow=float(pid_cfg.get("int_sat_scale_pow", 1.0)),
                int_sat_leak=float(pid_cfg.get("int_sat_leak", 1.0)),
            )
        )

        self.ctrl = DSLPIDControl(drone_model=drone_model)

        # -------------------------
        # Agentic parameters (feedforward)
        # -------------------------
        self.delta_thresh_m = float(ag_cfg.get("delta_thresh_m", ag_cfg.get("error_thresh_m", 0.6)))
        self.delta_ramp_mult = float(ag_cfg.get("delta_ramp_mult", ag_cfg.get("authority_ramp_mult", 2.5)))
        self.max_ref_shift_m = float(ag_cfg.get("max_ref_shift_m", ag_cfg.get("max_shift_m", 0.35)))

        self.authority_alpha = float(ag_cfg.get("authority_alpha", 0.0))
        self._authority = 0.0
        self.authority_max_rate = float(ag_cfg.get("authority_max_rate", 0.0))

        self.v_cmd_limit_mps = ag_cfg.get("v_cmd_limit_mps", None)
        if self.v_cmd_limit_mps is not None:
            self.v_cmd_limit_mps = float(self.v_cmd_limit_mps)

        # Closed-loop agentic gating (error-aware)
        self.error_gate_m = float(ag_cfg.get("error_gate_m", 0.5))
        self.error_scale_m = float(ag_cfg.get("error_scale_m", 1.5))

        # Comm-aware bias (toward centroid if isolated)
        self.comm_bias_m = float(ag_cfg.get("comm_bias_m", 0.15))
        self.comm_min_neighbors = int(ag_cfg.get("comm_min_neighbors", 1))

        # -------------------------
        # Saturation detection (graded)
        # -------------------------
        self._rpm_min = ag_cfg.get("rpm_min", cfg.get("sim", {}).get("rpm_min", None))
        self._rpm_max = ag_cfg.get("rpm_max", cfg.get("sim", {}).get("rpm_max", None))

        # band defines what "near the limit" means (fraction of rpm range).
        # sat_frac=0 if all motors are >= band away from bounds; sat_frac->1 at bounds.
        self._sat_band_frac = float(ag_cfg.get("sat_band_frac", 0.05))
        self._sat_eps_rpm = float(ag_cfg.get("rpm_eps", 1e-6))  # tiny numeric eps

    def reset(self):
        self.pid.reset()
        self._authority = 0.0
        if hasattr(self.ctrl, "reset"):
            try:
                self.ctrl.reset()
            except TypeError:
                pass

    def set_integrator_hold(self, hold: bool, leak_override: float | None = None):
        self.pid.set_integrator_hold(hold)
        self.pid.set_integrator_leak_override(leak_override)

    def _sat_frac_from_rpms(self, rpms: np.ndarray) -> float:
        """
        Graded saturation fraction in [0,1] based on proximity to rpm bounds.
        Returns 0.0 if bounds unknown.
        """
        if self._rpm_min is None or self._rpm_max is None:
            return 0.0

        rmin = float(self._rpm_min)
        rmax = float(self._rpm_max)
        if rmax <= rmin:
            return 0.0

        rpms = np.asarray(rpms, dtype=float)
        rng = rmax - rmin
        band = max(rng * float(self._sat_band_frac), 1e-6)

        # margin to closest bound, per motor
        margin = np.minimum(rpms - rmin, rmax - rpms)

        # sat=0 when margin>=band ; sat=1 when margin<=0
        sat_each = 1.0 - np.clip((margin - self._sat_eps_rpm) / band, 0.0, 1.0)
        return float(np.clip(np.max(sat_each), 0.0, 1.0))

    @staticmethod
    def _smoothstep01(x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _update_authority(self, a_target: float, dt: float) -> float:
        a_target = float(np.clip(a_target, 0.0, 1.0))

        # Stateless by default
        if self.authority_alpha <= 0.0:
            self._authority = a_target
            return a_target

        # EMA reference filter
        a_next = self.authority_alpha * float(self._authority) + (1.0 - self.authority_alpha) * a_target

        # Optional rate limit
        if self.authority_max_rate and self.authority_max_rate > 0.0 and dt > 0.0:
            max_da = float(self.authority_max_rate) * float(dt)
            a_next = float(np.clip(a_next, float(self._authority) - max_da, float(self._authority) + max_da))

        self._authority = float(np.clip(a_next, 0.0, 1.0))
        return float(self._authority)

    def compute_rpms(
        self,
        state,
        p_ref_now,
        v_ref_now,
        p_ref_ahead,
        v_ref_ahead,  # kept for API compatibility (unused in this conservative design)
        yaw_des: float = 0.0,
        *,
        group_pos: np.ndarray | None = None,
        comm_range_m: float | None = None,
        return_debug: bool = False,
    ):
        dt = 1.0 / float(self.cfg["sim"]["ctrl_hz"])

        # measured state (used only in baseline feedback control)
        cur_pos = np.asarray(state[0:3], dtype=float)
        cur_quat = np.asarray(state[3:7], dtype=float)
        cur_vel = np.asarray(state[10:13], dtype=float)
        cur_ang_vel = np.asarray(state[13:16], dtype=float)

        # references
        p_ref_now = np.asarray(p_ref_now, dtype=float)
        v_ref_now = np.asarray(v_ref_now, dtype=float)
        p_ref_ahead = np.asarray(p_ref_ahead, dtype=float)

        # ============================================
        # Agentic bias (closed-loop gating + comm-aware)
        # ============================================
        delta = (p_ref_ahead - p_ref_now).copy()
        delta[2] = 0.0
        delta_xy_norm = float(np.linalg.norm(delta[:2]))

        d0 = float(self.delta_thresh_m)
        d1 = max(d0 + 1e-6, float(self.delta_ramp_mult) * d0)

        raw = float(np.clip((delta_xy_norm - d0) / (d1 - d0), 0.0, 1.0))
        a_target = self._smoothstep01(raw)

        # Error-aware gating: if tracking error already large, reduce authority
        err_xy_norm = float(np.linalg.norm((p_ref_now - cur_pos)[:2]))
        err_gate = float(self.error_gate_m)
        err_scale = max(1e-6, float(self.error_scale_m))
        err_factor = float(np.clip((err_xy_norm - err_gate) / err_scale, 0.0, 1.0))
        a_target = a_target * (1.0 - err_factor)

        authority = self._update_authority(a_target, dt)

        # bounded bias
        shift = authority * delta
        shift_xy_norm = float(np.linalg.norm(shift[:2]))
        if shift_xy_norm > float(self.max_ref_shift_m):
            shift[:2] *= float(self.max_ref_shift_m) / (shift_xy_norm + 1e-9)

        # Comm-aware bias: if isolated, pull toward centroid
        comm_bias = np.zeros(3, dtype=float)
        if group_pos is not None and comm_range_m is not None and self.comm_bias_m > 0.0:
            pos_all = np.asarray(group_pos, dtype=float)
            if pos_all.ndim == 2 and pos_all.shape[1] == 3:
                n = pos_all.shape[0]
                nbrs = 0
                for j in range(n):
                    if np.allclose(pos_all[j], cur_pos):
                        continue
                    d = np.linalg.norm((pos_all[j] - cur_pos)[:2])
                    if d <= float(comm_range_m):
                        nbrs += 1
                if nbrs < self.comm_min_neighbors:
                    centroid = pos_all.mean(axis=0)
                    vec = centroid - cur_pos
                    vec[2] = 0.0
                    norm = float(np.linalg.norm(vec[:2]))
                    if norm > 1e-9:
                        comm_bias[:2] = (vec[:2] / norm) * float(self.comm_bias_m)

        p_cmd = p_ref_now + shift + comm_bias
        bias_norm = float(np.linalg.norm(shift[:2]))
        apply_bias_float = 1.0 if bias_norm > 1e-9 else 0.0

        # ============================================
        # Baseline feedback control (outer loop)
        # ============================================
        e = p_cmd - cur_pos
        v_err = v_ref_now - cur_vel

        # IMPORTANT: we need sat_frac SAME TICK for PIDCore.
        # We don't know rpms yet; so we do a 2-pass call:
        #   pass1: compute rpms with last tick sat state (stored inside PIDCore)
        #   compute sat_frac from rpms
        #   feed sat_frac into PIDCore for *next* step (or re-step if you want strict same-tick)
        #
        # To enforce same-tick, we re-run PID step once after estimating sat_frac.
        # This costs a tiny amount but keeps logic consistent.
        dv_cmd_1 = self.pid.step(e, v_err, dt)
        v_cmd_1 = v_ref_now + np.asarray(dv_cmd_1, dtype=float)

        # Inner: zero position error (discourages inner position loop)
        inner_target_pos = cur_pos

        rpms_1, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=inner_target_pos,
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=v_cmd_1,
        )

        sat_frac = self._sat_frac_from_rpms(rpms_1)
        self.pid.set_saturation(sat_frac)  # feed saturation to PIDCore

        # Re-step once with same-tick sat_frac so integrator logic is consistent
        dv_cmd = self.pid.step(e, v_err, dt, saturated=sat_frac)
        v_cmd = v_ref_now + np.asarray(dv_cmd, dtype=float)

        # Optional safety limiter (OFF by default)
        v_cmd_xy_norm_pre = float(np.linalg.norm(v_cmd[:2]))
        if self.v_cmd_limit_mps is not None:
            lim = float(self.v_cmd_limit_mps)
            sp_xy = float(np.linalg.norm(v_cmd[:2]))
            if sp_xy > lim:
                v_cmd[:2] *= lim / (sp_xy + 1e-9)
        v_cmd_xy_norm_post = float(np.linalg.norm(v_cmd[:2]))

        rpms, _, _ = self.ctrl.computeControl(
            control_timestep=dt,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=inner_target_pos,
            target_rpy=np.array([0.0, 0.0, float(yaw_des)], dtype=float),
            target_vel=v_cmd,
        )

        # update sat frac from final rpms too (logging)
        sat_frac = self._sat_frac_from_rpms(rpms)
        self.pid.set_saturation(sat_frac)

        # Metric error is against nominal mission reference (not p_cmd)
        e_metric = (p_ref_now - cur_pos).copy()
        e_metric[2] = 0.0
        err_norm = float(np.linalg.norm(e_metric[:2]))

        if return_debug:
            dbg = {
                "delta_xy_norm": float(delta_xy_norm),
                "delta_thresh_m": float(self.delta_thresh_m),
                "delta_ramp_mult": float(self.delta_ramp_mult),

                "authority_alpha": float(self.authority_alpha),
                "authority_max_rate": float(self.authority_max_rate),
                "authority": float(authority),
                "a_target": float(a_target),

                "bias_norm": float(bias_norm),
                "max_ref_shift_m": float(self.max_ref_shift_m),
                "err_xy_norm": float(err_xy_norm),
                "err_factor": float(err_factor),
                "comm_bias_xy": float(np.linalg.norm(comm_bias[:2])),

                "err_norm_nominal_xy": float(err_norm),

                "inner_pos_zeroed": True,

                "v_cmd_limit_mps": (None if self.v_cmd_limit_mps is None else float(self.v_cmd_limit_mps)),
                "v_cmd_xy_norm_pre": float(v_cmd_xy_norm_pre),
                "v_cmd_xy_norm_post": float(v_cmd_xy_norm_post),

                "sat_frac": float(sat_frac),
                "sat_band_frac": float(self._sat_band_frac),
                "rpm_min": (None if self._rpm_min is None else float(self._rpm_min)),
                "rpm_max": (None if self._rpm_max is None else float(self._rpm_max)),
            }
            return (
                rpms,
                err_norm,
                float(sat_frac),   # <-- return graded fraction
                True,
                bias_norm,
                p_cmd,
                apply_bias_float,
                dbg,
            )

        return (
            rpms,
            err_norm,
            float(sat_frac),   # <-- return graded fraction
            True,
            bias_norm,
            p_cmd,
            apply_bias_float,
        )
```

### `src/controllers/agentic_supervisor.py`
Key function: `AgenticSupervisor.step(...)`
```python
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

        self._last_phase_id = None
        self._phase_boost_timer = 0.0

        self._sup_timer = 0.0
        self._hold_timer = 0.0
        self._last_err_com = None
        self._hard_timer = 0.0

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
    ) -> SupervisorState:
        sup_dt = self._sup_timer
        self._sup_timer = 0.0

        com = pos_true.mean(axis=0)
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

        if mode == 3:
            self.state.formation_scale = self.formation_scale_min
            self.state.smooth_alpha = self.smooth_alpha_active
        elif mode == 2:
            self.state.int_hold = True
        elif mode == 1:
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
        if phase_id >= self.drift_disable_phase_ge and self.recovery_ref_shift_rate_mps > 0.0 and err_com > self.err_thresh_m:
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
            if self.phase_boost_rate_mps > 0.0 and err_com > self.err_thresh_m:
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
```

## Optional Diff Artifact
- Focused diff is available at:
  - `/Users/nitin/Desktop/case_study_1_tracking/review_bundle/changes.patch`
