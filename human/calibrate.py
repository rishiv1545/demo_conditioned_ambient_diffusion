"""ArUco detection and image->table homographies (table frame == sim frame, meters).

Marker centers sit at the rectangle corners: ID0 (0, 0) near-left, ID1 near-right, ID2 far-right, ID3 far-left.
Their table coordinates come from the measured distances in session.json.
"""
import json
import os

import cv2
import numpy as np
from scipy.ndimage import median_filter

DICT = cv2.aruco.DICT_4X4_50


def load_session(session_dir):
    with open(os.path.join(session_dir, "session.json")) as f:
        s = json.load(f)
    s["camera_height"] = s["camera_height_cm"] / 100.0
    bh = s.get("calib_box_height_cm")
    s["box_height"] = bh / 100.0 if bh is not None else None   # unknown -> calibration reports the estimate only
    corners = marker_table_xy(s["rect_cm"])  # near-left, near-right, far-right, far-left
    # Which marker ID is taped at each corner (near-left, near-right, far-right, far-left). Default: as in
    # RECORDING.md. Lets a rotated sticker layout be fixed in software instead of re-taping.
    ids = s.get("corner_ids", [0, 1, 2, 3])
    if sorted(ids) != [0, 1, 2, 3]:
        raise ValueError(f"corner_ids must be a permutation of 0-3, got {ids}")
    s["marker_xy"] = np.zeros((4, 2))
    for corner, mid in enumerate(ids):
        s["marker_xy"][mid] = corners[corner]   # indexed by marker ID, as video_homographies expects
    return s


def _circle_intersect(p0, r0, p1, r1, want_positive_y=True):
    d = np.linalg.norm(p1 - p0)
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(r0 ** 2 - a ** 2, 0.0))
    base = p0 + a * (p1 - p0) / d
    perp = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / d
    c1, c2 = base + h * perp, base - h * perp
    return c1 if (c1[1] > c2[1]) == want_positive_y else c2


def marker_table_xy(rect_cm):
    """Table coordinates (m) of the 4 rectangle corners (near-left, near-right, far-right, far-left) from measured
    side lengths and diagonals. In rect_cm, "d01" etc. name corners in that order (= marker IDs by default)."""
    r = {k: v / 100.0 for k, v in rect_cm.items()}
    p0, p1 = np.array([0.0, 0.0]), np.array([r["d01"], 0.0])
    p3 = _circle_intersect(p0, r["d30"], p1, r["d13"])
    p2 = _circle_intersect(p0, r["d02"], p1, r["d12"])
    return np.stack([p0, p1, p2, p3])


def detect_markers(frame, detector=None):
    """Return {id: center_px [2]} for marker IDs 0-3 found in a BGR frame."""
    detector = detector or cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT),
                                                   cv2.aruco.DetectorParameters())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = {}
    # second pass with local contrast enhancement for markers washed out by glare (session 3, marker 3: never found
    # plain, found in 13-21% of frames with CLAHE; the gaps are interpolated)
    for g in (gray, _CLAHE.apply(gray)):
        corners, ids, _ = detector.detectMarkers(g)
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                if 0 <= i <= 3 and int(i) not in out:
                    out[int(i)] = c[0].mean(0)
        if len(out) == 4:
            break
    return out


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def video_homographies(frames, marker_xy, smooth=9):
    """Per-frame image->table homography. Marker centers are tracked per frame, gaps are filled by linear
    interpolation, jitter is removed with a temporal median filter, then H is fit per frame.
    Returns (H [T, 3, 3], detected_fraction [4])."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), cv2.aruco.DetectorParameters())
    T = len(frames)
    pts = np.full((T, 4, 2), np.nan)
    for t, f in enumerate(frames):
        for i, c in detect_markers(f, det).items():
            pts[t, i] = c
    found = ~np.isnan(pts[..., 0])
    if not found.any(0).all():
        missing = [i for i in range(4) if not found[:, i].any()]
        raise RuntimeError(f"ArUco markers {missing} never detected")
    tt = np.arange(T)
    for i in range(4):
        for k in range(2):
            pts[:, i, k] = np.interp(tt, tt[found[:, i]], pts[found[:, i], i, k])
    if smooth > 1 and T >= smooth:
        pts = median_filter(pts, size=(smooth, 1, 1), mode="nearest")
    Hs = np.stack([cv2.getPerspectiveTransform(pts[t].astype(np.float32), marker_xy.astype(np.float32))
                   for t in range(T)])
    return Hs, found.mean(0)


def to_table(H, px):
    """Map image points [N, 2] (or [2]) to the table plane with homography H."""
    p = np.asarray(px, dtype=np.float64).reshape(-1, 2)
    q = cv2.perspectiveTransform(p[None], H)[0]
    return q if np.ndim(px) > 1 else q[0]


def camera_ground_xy(H, frame_shape):
    """Table point under the optical center, approximated by mapping the image center (used for parallax)."""
    h, w = frame_shape[:2]
    return to_table(H, [w / 2.0, h / 2.0])


def warp_topdown(frame, H, ppm=1000, margin=0.06, extent=(0.5, 0.35)):
    """Top-down view of the table: `ppm` pixels per meter, y up. Returns (image, table->pixel affine A)."""
    W = int((extent[0] + 2 * margin) * ppm)
    Hh = int((extent[1] + 2 * margin) * ppm)
    A = np.array([[ppm, 0, margin * ppm], [0, -ppm, (extent[1] + margin) * ppm], [0, 0, 1]])
    return cv2.warpPerspective(frame, A @ H, (W, Hh)), A


def draw_grid(img, A, extent=(0.5, 0.35), step=0.05):
    out = img.copy()
    for x in np.arange(0, extent[0] + 1e-9, step):
        p0, p1 = A @ [x, 0, 1], A @ [x, extent[1], 1]
        cv2.line(out, tuple(int(v) for v in p0[:2]), tuple(int(v) for v in p1[:2]), (255, 255, 255), 1)
    for y in np.arange(0, extent[1] + 1e-9, step):
        p0, p1 = A @ [0, y, 1], A @ [extent[0], y, 1]
        cv2.line(out, tuple(int(v) for v in p0[:2]), tuple(int(v) for v in p1[:2]), (255, 255, 255), 1)
    return out


def read_video(path, max_frames=None, max_width=960):
    """Read BGR frames, downscaled to max_width (a 15 s 1080p clip would otherwise need ~3 GB of RAM)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        if max_width and f.shape[1] > max_width:
            f = cv2.resize(f, (max_width, int(round(f.shape[0] * max_width / f.shape[1]))), interpolation=cv2.INTER_AREA)
        frames.append(f)
    cap.release()
    return frames, fps


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    p = argparse.ArgumentParser(description="Debug: warped top-down view with the table grid for one clip")
    p.add_argument("session")
    p.add_argument("clip")
    p.add_argument("--out", default="outputs/m2")
    a = p.parse_args()
    s = load_session(a.session)
    frames, fps = read_video(os.path.join(a.session, a.clip))
    Hs, frac = video_homographies(frames, s["marker_xy"])
    print("marker detection rate per ID:", np.round(frac, 3))
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(a.clip)[0]
    vids = [cv2.cvtColor(draw_grid(*warp_topdown(f, H, ppm=1500)), cv2.COLOR_BGR2RGB) for f, H in zip(frames[::3], Hs[::3])]
    imageio.mimsave(os.path.join(a.out, f"{name}_topdown.mp4"), vids, fps=fps / 3, macro_block_size=1)
