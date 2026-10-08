"""Sim camera matching the phone's viewpoint, estimated from the ArUco markers.

The phone's intrinsics are unknown (and the 0.5x ultra-wide differs from the main lens), so we assume a pinhole
camera with the principal point at the image center and solve for the focal length that makes solvePnP's
camera height equal the measured camera height. solvePnP then gives the full pose. The result goes into the
"phone_match" camera of the MuJoCo scene (position, orientation, vertical field of view).
"""
import cv2
import mujoco
import numpy as np

from human.calibrate import detect_markers


def _pnp(obj, img, w, h, f):
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    R, _ = cv2.Rodrigues(rv)
    C = (-R.T @ tv).ravel()
    err = np.linalg.norm(cv2.projectPoints(obj, rv, tv, K, None)[0].reshape(-1, 2) - img, axis=1).mean()
    return R, C, err


def estimate_phone_camera(frames, marker_xy, camera_height):
    """Median marker centers over the frames -> pose + focal length. Returns a dict for apply_camera()."""
    pts = {i: [] for i in range(4)}
    for f in frames:
        for i, c in detect_markers(f).items():
            pts[i].append(c)
    ids = [i for i in range(4) if pts[i]]
    if len(ids) < 4:
        raise RuntimeError(f"need all 4 markers for the camera pose, saw {ids}")
    img = np.array([np.median(pts[i], 0) for i in ids], np.float64)
    obj = np.array([[*marker_xy[i], 0.0] for i in ids], np.float64)
    h, w = frames[0].shape[:2]
    lo, hi = 0.2 * max(w, h), 5.0 * max(w, h)
    for _ in range(60):                         # camera height grows monotonically with f
        f = 0.5 * (lo + hi)
        _, C, _ = _pnp(obj, img, w, h, f)
        lo, hi = (f, hi) if C[2] < camera_height else (lo, f)
    R, C, err = _pnp(obj, img, w, h, f)
    # OpenCV camera: x right, y down, z forward. MuJoCo camera: x right, y up, looks along -z.
    x_ax, y_down, z_fwd = R.T[:, 0], R.T[:, 1], R.T[:, 2]
    rot = np.column_stack([x_ax, -y_down, -z_fwd])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    fovy = float(np.degrees(2 * np.arctan(h / 2 / f)))
    tilt = float(np.degrees(np.arccos(-z_fwd[2])))
    return {"pos": C.tolist(), "quat": quat.tolist(), "fovy": fovy, "width": int(w), "height": int(h),
            "focal_px": float(f), "fovx": float(np.degrees(2 * np.arctan(w / 2 / f))), "tilt_deg": tilt,
            "reproj_err_px": float(err)}


def apply_camera(env, cam, name="phone_match"):
    cid = env.m.camera(name).id
    env.m.cam_pos[cid] = cam["pos"]
    env.m.cam_quat[cid] = cam["quat"]
    env.m.cam_fovy[cid] = cam["fovy"]
    env.m.cam_mode[cid] = 0                     # fixed in the world
    mujoco.mj_forward(env.m, env.d)
