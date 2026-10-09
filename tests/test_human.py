"""Phone pipeline tests on synthetic inputs (no real video needed)."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human.calibrate import marker_table_xy, to_table, video_homographies  # noqa: E402
from human.extract import (detect_objects, fill_gaps, height_from_size, hysteresis, min_duration,  # noqa: E402
                           parallax_correct)
from human.objects import (ORDER, hsv_ranges_from_samples, layout_json_path, load_layout_json,  # noqa: E402
                           sample_patch, save_layout_json, session_specs)
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
    det = detect_objects(f, Hs[0], session_specs({}))  # default colors; flat squares: no parallax correction
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


# ---------------------------------------------------------------------------- household objects / manual fallback
def test_hsv_fit_handles_red_wraparound():
    px = np.array([[178, 200, 150], [1, 210, 160], [3, 190, 140], [176, 220, 170]] * 10)
    r = hsv_ranges_from_samples(px)
    assert len(r) == 2  # split at 0/179
    hsv = px.reshape(-1, 1, 3).astype(np.uint8)
    import cv2 as _cv2
    m = sum(_cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) for lo, hi in r)
    assert (m > 0).all()


def test_custom_household_colors_from_clicks():
    """Non-default colors (e.g. a pink eraser as "red", a cyan sticky note as "yellow"): defaults fail, ranges
    fitted from one click per item succeed."""
    import cv2 as _cv2
    from human.calibrate import warp_topdown
    house = {"red": (180, 105, 255), "green": (40, 90, 40), "blue": (120, 60, 20),
             "yellow": (230, 220, 40), "purple": (60, 60, 60), "orange": (80, 200, 170)}
    old = dict(BGR)
    BGR.update(house)
    try:
        objs = {"red": ((0.1, 0.1), 0.02), "green": ((0.25, 0.08), 0.02), "blue": ((0.42, 0.12), 0.02),
                "yellow": ((0.1, 0.26), 0.045), "purple": ((0.25, 0.25), 0.045), "orange": ((0.4, 0.27), 0.045)}
        f = synth_frame(objs)
    finally:
        BGR.clear()
        BGR.update(old)
    Hs, _ = video_homographies([f, f], marker_table_xy(RECT), smooth=1)
    top, A = warp_topdown(f, Hs[0], ppm=1600)
    hsv = _cv2.cvtColor(_cv2.GaussianBlur(top, (5, 5), 0), _cv2.COLOR_BGR2HSV)
    session = {"objects": [], "zones": []}
    for n, (xy, _) in objs.items():
        px = A @ [xy[0], xy[1], 1]
        ent = {"name": n, "label": f"thing {n}", "hsv": hsv_ranges_from_samples(sample_patch(hsv, px[:2], 8))}
        session["objects" if n in ("red", "green", "blue") else "zones"].append(ent)
    det = detect_objects(f, Hs[0], session_specs(session))
    for n, (xy, _) in objs.items():
        assert det[n] is not None and np.linalg.norm(det[n] - xy) < 0.006, (n, det[n], xy)


def test_layout_json_roundtrip_and_click_conversion(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from click_layout import clicks_to_table
    specs = session_specs({"objects": [{"name": "red", "height_cm": 5.0}]})
    A = np.array([[1600, 0, 96], [0, -1600, 656], [0, 0, 1.0]])  # warp_topdown(ppm=1600) affine
    cam, Hc = np.array([0.25, 0.175]), 0.75
    true = {"red": np.array([0.1, 0.1]), "yellow": np.array([0.3, 0.25])}
    clicks = {}
    for n, xy in true.items():
        z = specs[n]["height_m"]
        proj = cam + (xy - cam) * Hc / (Hc - z)          # where the object's top appears on the table plane
        clicks[n] = (A @ [*proj, 1])[:2]
    got = clicks_to_table(clicks, A, specs, cam, Hc)
    for n in true:
        assert np.allclose(got[n], true[n], atol=1e-6)
    full = {n: np.array([0.01 * i, 0.02 * i]) for i, n in enumerate(ORDER)}
    path = layout_json_path(str(tmp_path), "red-yellow_01.mp4")
    assert load_layout_json(path) is None
    save_layout_json(path, full)
    back = load_layout_json(path)
    assert all(np.allclose(back[n], full[n]) for n in ORDER)


def test_corner_ids_remap_markers(tmp_path):
    import json
    from human.calibrate import load_session
    base = {"camera_height_cm": 75, "rect_cm": RECT, "calib_box_height_cm": 15}
    for ids in ([0, 1, 2, 3], [1, 2, 3, 0]):
        (tmp_path / "session.json").write_text(json.dumps({**base, "corner_ids": ids}))
        s = load_session(str(tmp_path))
        corners = marker_table_xy(RECT)
        for corner, mid in enumerate(ids):
            assert np.allclose(s["marker_xy"][mid], corners[corner])


def test_contact_anchoring_removes_posture_bias():
    from human.retarget import anchor_contact_z
    t = np.linspace(0, 6, 61)
    true = np.interp(t, [0, 2, 3, 4, 5, 6], [0.10, 0.01, 0.01, 0.10, 0.01, 0.10])  # down, grasp, carry, place, up
    grip = (t >= 2) & (t < 5)
    biased = true + np.interp(t, [2, 5], [0.06, 0.04])                         # posture bias at contacts
    fixed = anchor_contact_z(t, biased, grip, 0.01)
    assert abs(fixed[np.argmax(grip)] - 0.01) < 1e-9 and abs(fixed[np.flatnonzero(grip)[-1] + 1] - 0.01) < 1e-9
    assert np.allclose(anchor_contact_z(t, biased, np.zeros_like(grip), 0.01), biased)


def test_grasp_interval_relative_thresholds():
    from human.extract import grasp_interval
    fps = 10
    # exaggerated pattern: rest 0.9 -> spread 2.0 -> grasp 0.85 (a wide object) -> spread 2.0 -> rest 0.9
    ap = np.r_[np.full(20, 0.9), np.full(10, 2.0), np.full(30, 0.85), np.full(10, 2.0), np.full(20, 0.9)]
    closed, info = grasp_interval(ap, fps, rest_closed=False)
    assert closed[30:60].all() and not closed[:30].any() and not closed[60:].any()   # rest at HOME is not a grasp
    # env v2: the gripper mirrors the hand: closed at rest, open while spread, closed on the object
    mirror, _ = grasp_interval(ap, fps)
    assert mirror[:20].all() and not mirror[20:30].any() and mirror[30:60].all()
    assert not mirror[60:70].any() and mirror[70:].all()
    # small object, barely-opening release: rest 0.9, grasp 0.25, release to only 0.6
    ap2 = np.r_[np.full(20, 0.9), np.full(30, 0.25), np.full(20, 0.6), np.full(10, 0.9)]
    c2, info2 = grasp_interval(ap2, fps, rest_closed=False)
    assert c2[20:50].all() and not c2[50:].any()


def test_dwell_inserted_at_switches():
    from human.retarget import add_dwell
    ee = np.arange(10, dtype=float)[:, None].repeat(3, 1)
    g = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], np.float32)
    E, G, T = add_dwell(ee, g, 2)
    assert len(T) == len(E) and T[0] == 0 and np.all(np.diff(T) >= 0)
    assert len(E) == 10 + 2 * 2 * 2
    i = np.flatnonzero(np.diff(G) != 0)[0] + 1        # first switch in the output
    assert np.allclose(E[i - 2:i + 2], ee[3]) and G[i - 1] == 0 and G[i] == 1


def test_approach_from_above_rule():
    from human.retarget import approach_from_above
    # rest closed -> open late, low approach from +x -> close at the cube -> carry -> release -> leave low
    T = 60
    ee = np.zeros((T, 3))
    ee[:, 2] = 0.02
    ee[:20, 0] = np.linspace(0.30, 0.10, 20)          # approach along -x at fingertip height
    ee[20:30, 0] = 0.10
    ee[30:40, :2] = np.linspace([0.10, 0.0], [0.10, 0.20], 10)
    ee[30:40, 2] = 0.06
    ee[40:, :2] = [0.10, 0.20]
    ee[45:, 0] = np.linspace(0.10, 0.30, 15)           # leave low after the release
    g = np.ones(T)
    g[17:22] = 0                                       # opens only 3 steps before arriving, closes at t=22
    g[42:50] = 0                                       # release at t=42
    out, go = approach_from_above(ee, g, clear=0.05, radius=0.06)
    tc, to = 22, 42
    assert np.allclose(out[tc], ee[tc]) and np.allclose(out[to], ee[to])    # contact points unchanged
    assert len(out) == T
    near = np.flatnonzero(np.linalg.norm(ee[:tc, :2] - ee[tc, :2], axis=1) <= 0.06)
    ta = near[0] - 1
    assert (go[ta:tc] == 0).all()                      # open for the whole approach
    # never low while still away from the grasp point: above contact + 2 cm whenever > 1 cm away in xy
    for k in range(ta + 1, tc):
        if np.linalg.norm(out[k, :2] - ee[tc, :2]) > 0.01:
            assert out[k, 2] >= ee[tc, 2] + 0.02, k
    # after the release: rises over the release point before moving away
    k_up = to + 1
    assert np.allclose(out[k_up, :2], ee[to, :2]) and out[k_up, 2] > ee[to, 2]
    assert out[to + 1:to + 8, 2].max() >= ee[to, 2] + 0.05 - 1e-9
    # no grasp -> unchanged
    o2, g2 = approach_from_above(ee, np.ones(T), 0.05, 0.06)
    assert np.allclose(o2, ee)
