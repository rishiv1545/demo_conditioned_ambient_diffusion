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
    dwell_s: float = 0.0           # hold the EE still this long before and after each gripper switch
    approach_clear: float = 0.0    # > 0: approach the grasp point from this high above it (gripper open), descend
    #                                vertically, and rise as high after the release before moving on (0 = off)
    approach_radius: float = 0.06  # the approach/retreat rule applies within this xy distance of the contact
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


def retarget(ex, cfg: RetargetConfig = RetargetConfig(), t_corr=None):
    """ex: output of extract.extract_clip. Returns (ee [N, 3], gripper [N] in {0, 1}, t [N]).
    t_corr: "miss and correct" clips: source time where the correction starts (human/correction.py); the approach
    rule then keeps the deliberate miss (lifted to the clearance height near the object) and starts at it."""
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
    if cfg.approach_clear > 0:
        k_corr = None if t_corr is None else int(np.searchsorted(t, t_corr))
        ee, g = approach_from_above(ee, g, cfg.approach_clear, cfg.approach_radius, keep_until=k_corr)
    lo = np.array([b[0] for b in cfg.ee_bounds])
    hi = np.array([b[1] for b in cfg.ee_bounds])
    ee = np.clip(ee, lo, hi)
    if cfg.dwell_s > 0:
        ee, g, t = add_dwell(ee, g, int(round(cfg.dwell_s * cfg.hz)), t)
    return ee, g, t   # t: source (phone) time of each step; repeated during dwells


def approach_from_above(ee, g, clear, radius, keep_until=None):
    """Embodiment rule for the parallel gripper. A human hand comes in low from the side and opens right at the
    object; fingers slide around it, but the Panda's fingers following that path hit and push the cube (phone
    session 3: the cube moved a median 1.1 cm before the grasp, up to 5 cm; 19/47 replays never lifted it).
    Within `radius` of the grasp point the gripper is open, moves over the point `clear` above it and descends
    vertically; after the release it rises vertically by `clear` before following the human path again. The grasp
    and release points (the human's actual choice, incl. its noise) and the number of steps are unchanged.
    keep_until (step index, "miss and correct" clips): the path before it is kept, except that within `radius` of the
    grasp point it is lifted to at least `clear` above it (the deliberate miss hovers beside the object instead of
    pushing it); the move over the grasp point and the descent start at keep_until.
    Returns modified copies; no grasp (open -> closed after an open phase) found: unchanged."""
    ee, g = ee.copy(), g.copy()
    closed = g > 0.5
    opened = np.flatnonzero(~closed)
    if not len(opened):
        return ee, g
    tc = next((k for k in range(opened[0] + 1, len(g)) if closed[k] and not closed[k - 1]), None)
    if tc is None:
        return ee, g
    p = ee[tc].copy()
    d = np.linalg.norm(ee[:tc, :2] - p[:2], axis=1)
    far = np.flatnonzero(d > radius)
    ta = far[-1] if len(far) else 0
    if keep_until is not None and 0 < keep_until < tc:
        near = np.flatnonzero(d[:keep_until] <= radius)
        ee[near, 2] = np.maximum(ee[near, 2], p[2] + clear)
        ta = max(ta, keep_until)
    n = tc - ta
    if n >= 2:
        zc = max(p[2] + clear, ee[ta, 2])
        n1 = max(1, n // 2)                       # move over the grasp point at the clearance height ...
        u = np.arange(1, n1 + 1) / n1
        ee[ta + 1:ta + n1 + 1, :2] = ee[ta, :2] + u[:, None] * (p[:2] - ee[ta, :2])
        ee[ta + 1:ta + n1 + 1, 2] = ee[ta, 2] + np.minimum(1.0, 3 * u) * (zc - ee[ta, 2])
        n2 = n - n1                               # ... then straight down
        v = np.arange(1, n2 + 1) / n2
        ee[ta + n1 + 1:tc + 1, :2] = p[:2]
        ee[ta + n1 + 1:tc + 1, 2] = zc + v[:n2] * (p[2] - zc)
        ee[tc] = p
        g[ta:tc] = 0.0                            # fully open before the descent
    to = next((k for k in range(tc + 1, len(g)) if not closed[k] and closed[k - 1]), None)
    if to is None:
        return ee, g
    q = ee[to].copy()
    away = np.flatnonzero(np.linalg.norm(ee[to:, :2] - q[:2], axis=1) > radius)
    tr = to + away[0] if len(away) else len(ee) - 1
    m = tr - to
    if m >= 2:
        zr = q[2] + clear
        m1 = max(1, m // 2)                       # straight up ...
        u = np.arange(1, m1 + 1) / m1
        ee[to + 1:to + m1 + 1, :2] = q[:2]
        ee[to + 1:to + m1 + 1, 2] = q[2] + u * (zr - q[2])
        m2 = m - m1                               # ... then over to where the human path leaves the radius
        v = np.arange(1, m2 + 1) / m2
        ee[to + m1 + 1:tr + 1, :2] = q[:2] + v[:, None] * (ee[tr, :2] - q[:2])
        ee[to + m1 + 1:tr + 1, 2] = np.maximum(zr + v * (ee[tr, 2] - zr), ee[to + m1 + 1:tr + 1, 2])
        g[to:tr] = 0.0                            # stay open while clearing the cube
    return ee, g


def add_dwell(ee, g, n, t=None):
    """At every gripper switch, hold the EE where the switch happens for n steps with the old gripper state, then
    n steps with the new one. The arm settles on the object before closing (the human closes on contact while
    the arm lags), and the gripper finishes closing/opening before the arm moves on."""
    t = np.arange(len(g), dtype=float) if t is None else np.asarray(t, float)
    sw = np.flatnonzero(np.diff(g) != 0) + 1
    if len(sw) == 0 or n <= 0:
        return ee, g, t
    E, G, Ts, prev = [], [], [], 0
    for k in sw:
        E += [ee[prev:k], np.repeat(ee[k:k + 1], n, 0), np.repeat(ee[k:k + 1], n, 0)]
        G += [g[prev:k], np.full(n, g[k - 1]), np.full(n, g[k])]
        Ts += [t[prev:k], np.full(2 * n, t[k])]
        prev = k
    E.append(ee[prev:])
    G.append(g[prev:])
    Ts.append(t[prev:])
    return np.concatenate(E), np.concatenate(G).astype(np.float32), np.concatenate(Ts)
