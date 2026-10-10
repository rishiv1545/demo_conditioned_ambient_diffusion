""""Miss and correct" phone clips: find the pause before the correction.

Protocol (RECORDING.md, section 5b): start the reach, deliberately stop 3-5 cm beside the object (hovering, not
touching), pause ~0.5 s, then move onto the object, grasp and finish the task. The policy should learn only the
correction: actions before the end of the pause get no loss (observations are kept).

Detection on the extracted pinch-point track (source frame rate, table frame): the last pause of the hand (3D speed
below `v_pause` for at least `min_pause_s`) before the grasp close, after the reach has started, whose position is
`min_off`-`max_off` from the grasp point in xy. The hand's natural slowdown right at the object is closer than
`min_off`, so it is not taken for the deliberate miss.
"""
import numpy as np


def first_grasp_close(closed):
    """Index of the first open -> closed switch after an open phase, or None."""
    closed = np.asarray(closed, bool)
    opened = np.flatnonzero(~closed)
    if not len(opened):
        return None
    return next((k for k in range(opened[0] + 1, len(closed)) if closed[k] and not closed[k - 1]), None)


def runs(mask):
    """[(start, end_exclusive)] of the True runs of a boolean array."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(d[::2], d[1::2]))


def detect_correction(ex, fps, v_pause=0.03, min_pause_s=0.25, min_off=0.02, max_off=0.10, reach_start=0.05,
                      smooth_s=0.2):
    """ex: extract.extract_clip output. Returns a dict (t_corr = end of the pause in source seconds, the pause
    interval, its xy offset from the grasp point) or None with a reason in `why`."""
    xy, z = np.asarray(ex["xy"], float), np.asarray(ex["z"], float)
    p = np.column_stack([xy, z])
    ic = first_grasp_close(ex["grip"])
    if ic is None:
        return {"why": "no grasp close"}
    k = max(1, int(round(smooth_s * fps)))
    v = np.linalg.norm(np.gradient(p, axis=0), axis=1) * fps
    v = np.convolve(v, np.ones(k) / k, mode="same")
    moved = np.flatnonzero(np.linalg.norm(p[:, :2] - p[0, :2], axis=1) > reach_start)
    if not len(moved) or moved[0] >= ic:
        return {"why": "reach start not found before the grasp"}
    i0 = moved[0]
    g = xy[ic]
    cands = []
    for s, e in runs(v < v_pause):
        s, e = max(s, i0), min(e, ic)
        if e - s < min_pause_s * fps:
            continue
        off = np.linalg.norm(xy[s:e].mean(0) - g)
        if min_off <= off <= max_off:
            cands.append((s, e, off))
    if not cands:
        return {"why": f"no pause {min_off * 100:.0f}-{max_off * 100:.0f} cm from the grasp point before the grasp"}
    s, e, off = cands[-1]
    vec = xy[s:e].mean(0) - g
    return {"t_corr": float(e / fps), "t_pause": [float(s / fps), float(e / fps)], "i_corr": int(e),
            "offset_cm": round(float(off) * 100, 2), "offset_xy_cm": [round(float(x) * 100, 2) for x in vec],
            "pause_z_cm": round(float(z[s:e].mean()) * 100, 2), "t_grasp": float(ic / fps), "n_pauses": len(cands)}
