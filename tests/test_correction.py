import numpy as np

from human.correction import detect_correction
from human.retarget import approach_from_above


def synthetic_clip(fps=30, miss=(0.04, 0.0), with_pause=True):
    """Reach from (0, -0.25) toward a point beside the object at (0, 0), pause, correct onto it, grasp, carry."""
    seg = []

    def move(a, b, s):
        n = int(s * fps)
        u = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
        seg.append(a + u[:, None] * (b - a))
    start, obj = np.array([0.0, -0.25, 0.15]), np.array([0.0, 0.0, 0.02])
    m = obj + np.array([miss[0], miss[1], 0.06])
    move(start, m, 1.2)
    if with_pause:
        seg.append(np.repeat(m[None], int(0.6 * fps), 0))
    move(m, obj, 0.8)
    seg.append(np.repeat(obj[None], int(0.4 * fps), 0))      # natural pause at the object, then grasp
    n_pre = sum(len(s) for s in seg)
    move(obj, np.array([0.2, 0.1, 0.1]), 1.5)
    p = np.concatenate(seg)
    grip = np.zeros(len(p), bool)
    grip[:5] = True                                            # rest closed, opens, closes at the grasp
    grip[n_pre - 3:] = True
    return {"xy": p[:, :2], "z": p[:, 2], "grip": grip}, n_pre


def test_detects_the_deliberate_pause_not_the_one_at_the_object():
    ex, n_pre = synthetic_clip()
    c = detect_correction(ex, 30)
    assert "t_corr" in c, c
    assert 3.0 <= c["offset_cm"] <= 5.0
    assert abs(c["t_corr"] - 1.8) < 0.15                       # end of the 0.6 s pause after the 1.2 s reach


def test_no_pause_gives_none():
    ex, _ = synthetic_clip(with_pause=False)
    assert "t_corr" not in detect_correction(ex, 30)


def test_approach_rule_keeps_the_miss_lifted():
    ee = np.array([[0.0, -0.2, 0.10], [0.04, -0.02, 0.03], [0.04, 0.0, 0.03], [0.04, 0.0, 0.03],
                   [0.02, 0.0, 0.03], [0.0, 0.0, 0.02], [0.0, 0.0, 0.02], [0.0, 0.0, 0.02], [0.1, 0.1, 0.1]])
    g = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], np.float32)
    out, gg = approach_from_above(ee, g, clear=0.05, radius=0.06, keep_until=3)
    assert np.allclose(out[1:3, :2], ee[1:3, :2])              # the miss keeps its xy ...
    assert (out[1:3, 2] >= 0.07 - 1e-9).all()                  # ... but hovers at the clearance height
    assert np.allclose(out[6], ee[6])                          # grasp point unchanged
    assert np.allclose(out[5, :2], [0.0, 0.0])                 # descends vertically onto it
