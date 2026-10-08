"""Human grasp-point trajectory (table frame) -> robot EE target trajectory at 10 Hz.

The real marker rectangle is defined to coincide with the sim workspace, so the map is 1:1 metric with a fixed
offset (default zero). A fixed z offset relates the human pinch point to the Panda fingertip center.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class RetargetConfig:
    xy_offset: tuple = (0.0, 0.0)
    z_offset: float = 0.0          # pinch point -> fingertip center
    z_min: float = 0.012           # never command the fingertips into the table
    hz: float = 10.0
    ee_bounds: tuple = ((-0.05, 0.55), (-0.08, 0.42), (0.008, 0.40))


def retarget(ex, cfg: RetargetConfig = RetargetConfig()):
    """ex: output of extract.extract_clip. Returns (ee [N, 3], gripper [N] in {0, 1}, t [N])."""
    t_src = ex["t"]
    t = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / cfg.hz)
    x = np.interp(t, t_src, ex["xy"][:, 0]) + cfg.xy_offset[0]
    y = np.interp(t, t_src, ex["xy"][:, 1]) + cfg.xy_offset[1]
    z = np.interp(t, t_src, ex["z"]) + cfg.z_offset
    # gripper: sample the cleaned binary signal at the nearest source frame
    idx = np.clip(np.searchsorted(t_src, t), 0, len(t_src) - 1)
    g = ex["grip"][idx].astype(np.float32)
    ee = np.column_stack([x, y, np.maximum(z, cfg.z_min)])
    lo = np.array([b[0] for b in cfg.ee_bounds])
    hi = np.array([b[1] for b in cfg.ee_bounds])
    return np.clip(ee, lo, hi), g, t
