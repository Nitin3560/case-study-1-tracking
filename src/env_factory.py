from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


@dataclass
class EnvHandles:
    env: CtrlAviary
    dt_sim: float
    dt_ctrl: float


def _drone_model_from_str(s: str) -> DroneModel:
    s = s.lower()
    if s in ["cf2x", "crazyflie2x"]:
        return DroneModel.CF2X
    if s in ["cf2p", "crazyflie2p"]:
        return DroneModel.CF2P
    raise ValueError(f"Unknown drone_model: {s}")


def _physics_from_str(s: str) -> Physics:
    s = s.lower()
    if s in ["pyb", "pybullet"]:
        return Physics.PYB
    if s in ["dyn", "dynamics"]:
        return Physics.DYN
    raise ValueError(f"Unknown physics: {s}")


def make_env(cfg: Dict[str, Any]) -> EnvHandles:
    sim = cfg.get("sim", {})
    traj = cfg.get("trajectory", {})

    num = int(sim.get("num_drones", sim.get("n", 3)))

    dt_default = 1.0 / 60.0
    dt_in = float(sim.get("dt", dt_default))
    freq = int(sim.get("freq_hz", round(1.0 / dt_in)))
    ctrl_hz = int(sim.get("ctrl_hz", freq))

    dt_sim = 1.0 / float(freq)
    dt_ctrl = 1.0 / float(ctrl_hz)

    altitude = float(traj.get("altitude_m", traj.get("z_m", 1.2)))

    init_xyzs = np.zeros((num, 3), dtype=float)
    for i in range(num):
        init_xyzs[i] = np.array([0.0 + 0.5 * i, 0.0, altitude], dtype=float)

    drone_model_str = str(sim.get("drone_model", "cf2x"))
    physics_str = str(sim.get("physics", "pyb"))
    gui = bool(sim.get("gui", False))

    env = CtrlAviary(
        drone_model=_drone_model_from_str(drone_model_str),
        num_drones=num,
        initial_xyzs=init_xyzs,
        physics=_physics_from_str(physics_str),
        pyb_freq=freq,
        ctrl_freq=ctrl_hz,
        gui=gui,
    )
    # Hard-disable user debug GUI input reading in headless runs
    if not gui and hasattr(env, "INPUT_SWITCH"):
        try:
            env.INPUT_SWITCH = -1
        except Exception:
            pass


    return EnvHandles(env=env, dt_sim=dt_sim, dt_ctrl=dt_ctrl)
