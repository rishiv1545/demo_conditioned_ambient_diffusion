"""Phone pipeline tests on synthetic inputs (no real video needed)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human.calibrate import marker_table_xy, to_table, video_homographies  # noqa: E402
from human.extract import (detect_objects, fill_gaps, height_from_size, hysteresis, min_duration,  # noqa: E402
                           parallax_correct)
from human.replay import replay  # noqa: E402
from human.retarget import retarget  # noqa: E402
from sim.env import PickPlaceEnv  # noqa: E402
from sim.expert import run_expert_episode  # noqa: E402

RECT = {"d01": 50.0, "d12": 35.0, "d23": 50.0, "d30": 35.0, "d02": 61.03, "d13": 61.03}
BGR = {"red": (30, 30, 210), "green": (50, 170, 40), "blue": (200, 70, 30),
       "yellow": (30, 210, 240), "purple": (150, 40, 110), "orange": (0, 110, 245)}


def synth_frame(objs, tilt=True):
    """Top-down canvas (2 px/mm) with 6 cm ArUco markers + flat colored squares, then a perspective warp."""
    ppm, m = 2000, 0.08
    W, H = int((0.5 + 2 * m) * ppm), int((0.35 + 2 * m) * ppm)
    img = np.full((H, W, 3), 200, np.uint8)
    to_px = lambda x, y: (int((x + m) * ppm), int((0.35 + m - y) * ppm))
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for i, (x, y) in enumerate(marker_table_xy(RECT)):
        s = int(0.06 * ppm)
        mk = cv2.cvtColor(cv2.aruco.generateImageMarker(d, i, s, borderBits=1), cv2.COLOR_GRAY2BGR)
        cx, cy = to_px(x, y)
        img[cy - s // 2 - 20:cy + s // 2 + 20, cx - s // 2 - 20:cx + s // 2 + 20] = 255
        img[cy - s // 2:cy - s // 2 + s, cx - s // 2:cx - s // 2 + s] = mk
    for name, (xy, half) in objs.items():
        p0, p1 = to_px(xy[0] - half, xy[1] + half), to_px(xy[0] + half, xy[1] - half)
        cv2.rectangle(img, p0, p1, BGR[name], -1)
    if not tilt:
        return img
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[60, 40], [W - 20, 0], [W - 90, H - 30], [10, H]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H))


def test_marker_geometry():
    xy = marker_table_xy(RECT)
    assert np.allclose(xy, [[0, 0], [0.5, 0], [0.5, 0.35], [0, 0.35]], atol=2e-3)


def test_homography_and_objects():
    objs = {"red": ((0.1, 0.1), 0.02), "green": ((0.25, 0.08), 0.02), "blue": ((0.42, 0.12), 0.02),
            "yellow": ((0.1, 0.26), 0.05), "purple": ((0.25, 0.25), 0.05), "orange": ((0.4, 0.27), 0.05)}
    f = synth_frame(objs)
    Hs, frac = video_homographies([f, f, f], marker_table_xy(RECT), smooth=1)
    assert np.all(frac == 1.0)
    det = detect_objects(f, Hs[0])  # flat synthetic squares: no parallax correction
    for name, (xy, _) in objs.items():
        assert det[name] is not None, name
        assert np.linalg.norm(det[name] - xy) < 0.006, (name, det[name], xy)


def test_height_and_parallax_roundtrip():
    H, L, z = 0.75, 0.085, np.array([0.0, 0.05, 0.15])
    s = L * H / (H - z)
    assert np.allclose(height_from_size(s, s[0], H), z)
    cam, true_xy = np.array([0.25, 0.17]), np.array([[0.1, 0.3], [0.4, 0.05], [0.25, 0.17]])
    proj = cam + (true_xy - cam) * (H / (H - z))[:, None]
    assert np.allclose(parallax_correct(proj, z, cam, H), true_xy)


def test_gripper_signal_cleanup():
    sig = np.array([1.2] * 10 + [0.5] * 20 + [0.95, 0.5] + [0.5] * 10 + [1.3] * 10)
    g = hysteresis(sig, 0.65, 0.9)
    g = min_duration(g, 4)
    assert not g[:10].any() and g[10:42].all() and not g[42:].any()


def test_fill_gaps():
    x = np.array([0.0, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, 7.0])
    y, long_gap = fill_gaps(x, max_gap=2)
    assert np.allclose(y, np.arange(8)) and long_gap.tolist() == [0, 0, 0, 1, 1, 1, 1, 0]


def test_retarget_replay_noisy_expert():
    """A 'human' trajectory = expert EE path at 30 fps + noise and a late gripper close; replay should succeed."""
    env = PickPlaceEnv()
    r = run_expert_episode(env, (1, 2), seed=11)
    lay = {k: v.copy() for k, v in env.layout.items()}
    ee10, g10 = r["obs"][:, :3], r["action"][:, 3]
    t10 = np.arange(len(ee10)) / 10.0
    t30 = np.arange(0, t10[-1], 1 / 30.0)
    rng = np.random.default_rng(0)
    xy = np.column_stack([np.interp(t30, t10, ee10[:, k]) for k in range(2)]) + rng.normal(0, 0.002, (len(t30), 2))
    z = np.interp(t30, t10, ee10[:, 2]) + rng.normal(0, 0.003, len(t30))
    grip = np.interp(t30 - 0.1, t10, g10) > 0.5
    ex = {"t": t30, "xy": xy, "z": z, "grip": grip}
    ee, g, _ = retarget(ex)
    out = replay(env, (1, 2), lay, ee, g)
    assert out["success"], out["track_err"]
    assert out["obs"].shape[0] == out["action"].shape[0]
