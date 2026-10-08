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
    contact_z: float = None        # if set: pinch height (human frame) at grasp/release, used to re-anchor z
    z_min: float = 0.012           # never command the fingertips into the table
    hz: float = 10.0
    ee_bounds: tuple = ((-0.05, 0.55), (-0.08, 0.42), (0.008, 0.40))


def anchor_contact_z(t, z, grip, contact_z):
    """Contact-anchored height. At the moments the grip closes and opens, the fingertips are at the object's
    grasp height (contact_z). The size-based z is biased by hand posture (a pinching hand differs from the flat
    calibration hand), so subtract the bias measured at those two contacts, linearly interpolated between them
    and held constant outside. No grasp detected: z unchanged."""
    g = np.asarray(grip, bool)
    closes = np.flatnonzero(~g[:-1] & g[1:]) + 1
    if len(closes) == 0:
        return z
    tc = closes[0]
    opens = np.flatnonzero(g[tc:-1] & ~g[tc + 1:]) + tc + 1
    to = opens[0] if len(opens) else len(z) - 1
    b = np.interp(t, [t[tc], t[to]], [z[tc] - contact_z, z[to] - contact_z])
    return z - b


def retarget(ex, cfg: RetargetConfig = RetargetConfig()):
    """ex: output of extract.extract_clip. Returns (ee [N, 3], gripper [N] in {0, 1}, t [N])."""
    t_src = ex["t"]
    z_src = np.asarray(ex["z"], float)
    if cfg.contact_z is not None:
        z_src = anchor_contact_z(t_src, z_src, ex["grip"], cfg.contact_z)
    t = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / cfg.hz)
    x = np.interp(t, t_src, ex["xy"][:, 0]) + cfg.xy_offset[0]
    y = np.interp(t, t_src, ex["xy"][:, 1]) + cfg.xy_offset[1]
    z = np.interp(t, t_src, z_src) + cfg.z_offset
    # gripper: sample the cleaned binary signal at the nearest source frame
    idx = np.clip(np.searchsorted(t_src, t), 0, len(t_src) - 1)
    g = ex["grip"][idx].astype(np.float32)
    ee = np.column_stack([x, y, np.maximum(z, cfg.z_min)])
    lo = np.array([b[0] for b in cfg.ee_bounds])
    hi = np.array([b[1] for b in cfg.ee_bounds])
    return np.clip(ee, lo, hi), g, t
