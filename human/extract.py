"""Hand tracking (MediaPipe Tasks HandLandmarker), grasp point, height from apparent hand size, gripper signal,
and cube/zone detection by HSV thresholding.

Height model: a point at height z above the table projects onto the table plane (through the camera center at
height H) magnified by H / (H - z). We measure a rigid hand length in *table-plane* meters (landmarks mapped by
the homography, which also removes the camera tilt), so s = L * H / (H - z) and, with s0 from the hand-flat
calibration (z = 0), z = H * (1 - s0 / s).
"""
import os
import urllib.request

import cv2
import numpy as np
from scipy.signal import savgol_filter

from human.calibrate import camera_ground_xy, to_table, warp_topdown
from human.objects import color_mask, largest_blob, session_specs

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
             "hand_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models",
                          "hand_landmarker.task")

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17



# ---------------------------------------------------------------------------- hand tracking
def hand_model_path():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def track_hand(frames, fps, handedness="right"):
    """Return landmarks_px [T, 21, 2] (NaN where no hand was found)."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=hand_model_path()),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    out = np.full((len(frames), 21, 2), np.nan)
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for t, f in enumerate(frames):
            h, w = f.shape[:2]
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            r = lm.detect_for_video(img, int(round(t * 1000.0 / fps)))
            if not r.hand_landmarks:
                continue
            # prefer the requested hand; MediaPipe's handedness label assumes a mirrored (selfie) image, so we
            # don't trust it blindly: take the matching label if present, else the most confident hand.
            k = 0
            labels = [hh[0].category_name.lower() for hh in r.handedness]
            if len(labels) > 1 and handedness in labels:
                k = labels.index(handedness)
            out[t] = [[p.x * w, p.y * h] for p in r.hand_landmarks[k]]
    return out


def hand_size_table(lm_tab):
    """Rigid hand length in table-plane meters: mean of wrist->middle MCP and index MCP->pinky MCP."""
    a = np.linalg.norm(lm_tab[..., WRIST, :] - lm_tab[..., MIDDLE_MCP, :], axis=-1)
    b = np.linalg.norm(lm_tab[..., INDEX_MCP, :] - lm_tab[..., PINKY_MCP, :], axis=-1)
    return 0.5 * (a + b)


def landmarks_to_table(lm_px, Hs):
    T = len(lm_px)
    out = np.full((T, 21, 2), np.nan)
    for t in range(T):
        if not np.isnan(lm_px[t, 0, 0]):
            out[t] = to_table(Hs[t], lm_px[t])
    return out


def calib_hand_size(frames, fps, Hs):
    """Median table-plane hand size over a calibration clip (hand held flat and still)."""
    lm = landmarks_to_table(track_hand(frames, fps), Hs)
    s = hand_size_table(lm)
    s = s[~np.isnan(s)]
    if len(s) == 0:
        raise RuntimeError("no hand found in calibration clip")
    return float(np.median(s)), float(np.std(s)), len(s)


def height_from_size(s, s0, H):
    return H * (1.0 - s0 / s)


def parallax_correct(xy_proj, z, cam_xy, H):
    """Table-plane projection of a point at height z -> its true xy (pull toward the camera ground point)."""
    return cam_xy + (xy_proj - cam_xy) * ((H - z) / H)[..., None]


# ---------------------------------------------------------------------------- signal cleanup
def fill_gaps(x, max_gap):
    """Linearly interpolate NaN runs up to max_gap samples; returns (filled, long_gap_mask)."""
    x = np.array(x, dtype=float)
    bad = np.isnan(x) if x.ndim == 1 else np.isnan(x).any(-1)
    long_gap = np.zeros_like(bad)
    t = np.arange(len(bad))
    if bad.all():
        return x, ~long_gap
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if j - i > max_gap:
                long_gap[i:j] = True
            i = j
        else:
            i += 1
    cols = x.reshape(len(x), -1)
    for c in range(cols.shape[1]):
        cols[bad, c] = np.interp(t[bad], t[~bad], cols[~bad, c])
    return cols.reshape(x.shape), long_gap


def smooth(x, fps, window_s=0.3, order=2):
    n = max(order + 2, int(window_s * fps) | 1)
    return savgol_filter(x, n, order, axis=0) if len(x) > n else x


def hysteresis(sig, lo, hi, start_closed=False):
    """sig small = closed. Close when sig < lo, open when sig > hi."""
    state, out = start_closed, np.zeros(len(sig), dtype=bool)
    for t, v in enumerate(sig):
        if state and v > hi:
            state = False
        elif not state and v < lo:
            state = True
        out[t] = state
    return out


def min_duration(binary, n):
    """Remove runs (of either value) shorter than n samples by merging them into their neighbors."""
    b = binary.copy()
    changed = True
    while changed:
        changed = False
        edges = np.flatnonzero(np.diff(b.astype(int))) + 1
        bounds = [0, *edges, len(b)]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e - s < n and not (s == 0 or e == len(b)):
                b[s:e] = not b[s]
                changed = True
                break
    return b


# ---------------------------------------------------------------------------- objects
def detect_objects(frame, H, specs, camera_h=None, cam_xy=None, ppm=1000, exclude_xy=None, exclude_half=0.05,
                   hand_px=None, hand_margin=0.03):
    """Centers (table m) of the 3 objects and 3 zones via HSV thresholding of the top-down warp, using the
    per-session color specs (human.objects.session_specs). Returns {name: xy or None}. Object centroids are
    parallax-corrected to their mid-height (the blob covers the object's top and visible sides).
    exclude_xy: table points (the marker centers) around which a square of half-size exclude_half is ignored,
    so black/white items aren't confused with the ArUco squares.
    hand_px: hand landmarks [21, 2] (image px) in this frame; their convex hull plus hand_margin is ignored, so a
    pale hand isn't taken for a white patch nor the gaps between fingers for a dark object."""
    top, A = warp_topdown(frame, H, ppm=ppm)
    hsv = cv2.cvtColor(cv2.GaussianBlur(top, (5, 5), 0), cv2.COLOR_BGR2HSV)
    Ainv = np.linalg.inv(A)
    hand_mask = None
    if hand_px is not None and not np.isnan(hand_px).any():
        tab = to_table(H, hand_px)
        hull = cv2.convexHull((np.c_[tab, np.ones(len(tab))] @ A.T)[:, :2].astype(np.int32))
        hand_mask = np.zeros(hsv.shape[:2], np.uint8)
        cv2.fillConvexPoly(hand_mask, hull, 255)
        k = 2 * int(hand_margin * ppm) + 1
        hand_mask = cv2.dilate(hand_mask, np.ones((k, k), np.uint8))
    res = {}
    for name, sp in specs.items():
        min_area = (1.5e-4 if sp["kind"] == "object" else 2e-3) * ppm ** 2   # 1.2 cm / 4.5 cm squares
        mask = color_mask(hsv, sp["hsv"])
        if hand_mask is not None:
            mask[hand_mask > 0] = 0
        for x, y in (exclude_xy if exclude_xy is not None else []):
            u, v = (A @ [x, y, 1.0])[:2].astype(int)
            h = int(exclude_half * ppm)
            mask[max(0, v - h):max(0, v + h), max(0, u - h):max(0, u + h)] = 0
        c = largest_blob(mask, min_area)
        if c is None:
            res[name] = None
            continue
        xy = (Ainv @ [*c, 1.0])[:2]
        if sp["kind"] == "object" and camera_h is not None and cam_xy is not None:
            xy = parallax_correct(xy, np.array(sp["height_m"] / 2), cam_xy, camera_h)
        res[name] = xy
    return res


# ---------------------------------------------------------------------------- full extraction
def extract_clip(frames, fps, Hs, session, s0, grip_lo=0.65, grip_hi=0.9, min_dur_s=0.2, max_gap_s=0.5):
    """Returns dict with t [T], xy [T,2], z [T], grip [T] (bool closed), aperture [T], valid [T], lm_px, objects."""
    Hc = session["camera_height"]
    lm_px = track_hand(frames, fps, session.get("hand", "right"))
    lm_tab = landmarks_to_table(lm_px, Hs)
    T = len(frames)
    cam = np.stack([camera_ground_xy(Hs[t], frames[0].shape) for t in range(T)])
    pinch = 0.5 * (lm_tab[:, THUMB_TIP] + lm_tab[:, INDEX_TIP])
    size = hand_size_table(lm_tab)
    ap = np.linalg.norm(lm_tab[:, THUMB_TIP] - lm_tab[:, INDEX_TIP], axis=-1) / size
    max_gap = int(max_gap_s * fps)
    raw = np.column_stack([pinch, size, ap])
    filled, long_gap = fill_gaps(raw, max_gap)
    sm = smooth(filled, fps)
    size_s = sm[:, 2]
    z = np.clip(height_from_size(size_s, s0, Hc), 0.0, None)
    xy = parallax_correct(sm[:, :2], z, cam, Hc)
    closed = min_duration(hysteresis(sm[:, 3], grip_lo, grip_hi), max(1, int(min_dur_s * fps)))
    specs = session_specs(session)
    mk = session["marker_xy"]
    objs = detect_objects(frames[0], Hs[0], specs, camera_h=Hc, cam_xy=cam[0], exclude_xy=mk, hand_px=lm_px[0])
    objs_end = detect_objects(frames[-1], Hs[-1], specs, camera_h=Hc, cam_xy=cam[-1], exclude_xy=mk,
                              hand_px=lm_px[-1])
    return {"t": np.arange(T) / fps, "xy": xy, "z": z, "grip": closed, "aperture": sm[:, 3], "size": size_s,
            "valid": ~long_gap, "cam_xy": cam, "tracked": ~np.isnan(lm_px[:, 0, 0]), "lm_px": lm_px, "objects": objs,
            "objects_end": objs_end}


def overlay_frame(frame, lm_px, xy, z, closed, t_idx):
    """Debug overlay: landmarks, grasp point, z and gripper state."""
    f = frame.copy()
    if not np.isnan(lm_px[0, 0]):
        for p in lm_px:
            cv2.circle(f, tuple(int(v) for v in p), 4, (0, 255, 255), -1)
        g = 0.5 * (lm_px[THUMB_TIP] + lm_px[INDEX_TIP])
        cv2.circle(f, tuple(int(v) for v in g), 10, (0, 0, 255) if closed else (0, 255, 0), 3)
    txt = f"t={t_idx} x={xy[0]:.3f} y={xy[1]:.3f} z={z:.3f} {'CLOSED' if closed else 'open'}"
    cv2.putText(f, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(f, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return f


def plot_traj(ex, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
    for a, k, lab in zip(ax, ["x", "y", "z"], ["x (m)", "y (m)", "z (m)"]):
        v = ex["xy"][:, "xy".index(k)] if k in "xy" else ex["z"]
        a.plot(ex["t"], v)
        a.set_ylabel(lab)
    ax[3].plot(ex["t"], ex["aperture"], label="aperture / hand size")
    ax[3].plot(ex["t"], ex["grip"].astype(float), label="closed")
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].set_xlabel("time (s)")
    for a in ax:
        a.fill_between(ex["t"], *a.get_ylim(), where=~ex["valid"], color="r", alpha=0.15)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
