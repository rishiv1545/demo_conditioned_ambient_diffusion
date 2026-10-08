"""Per-session object/zone specs, HSV ranges from clicked samples, color detection and manual layout overrides.

Task names in clip file names always use the canonical names (red/green/blue cubes, yellow/purple/orange zones).
session.json maps each canonical name to the real household object and its color:

    "objects": [{"name": "red",    "label": "red Lego brick", "height_cm": 2.0, "hsv": [[[0,120,70],[8,255,255]]]}, ...],
    "zones":   [{"name": "yellow", "label": "yellow sticky note",            "hsv": [[[21,100,120],[35,255,255]]]}, ...]

"hsv" is a list of [lo, hi] ranges in OpenCV units (H 0-179, S and V 0-255); several ranges handle hue wrap-around.
Missing entries fall back to the defaults below. A per-clip "<clip>.layout.json" (from scripts/click_layout.py)
overrides color detection for that clip.
"""
import json
import os
import re

import cv2
import numpy as np

from sim.env import CUBES, ZONES

DEFAULT_HSV = {
    "red": [((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((9, 140, 120), (20, 255, 255))],
    "yellow": [((21, 100, 120), (35, 255, 255))],
    "green": [((40, 80, 50), (85, 255, 255))],
    "blue": [((95, 120, 50), (125, 255, 255))],
    "purple": [((126, 50, 40), (165, 255, 255))],
}
DEFAULT_OBJ_HEIGHT_CM = 4.0
ORDER = list(CUBES) + list(ZONES)  # fixed click order: 3 objects, then 3 zones


def session_specs(session):
    """{canonical name: {"kind": "object"|"zone", "label", "height_m", "hsv": [(lo, hi), ...]}} for all 6 items."""
    given = {e["name"]: e for e in session.get("objects", []) + session.get("zones", [])}
    unknown = set(given) - set(ORDER)
    if unknown:
        raise ValueError(f"session.json names {sorted(unknown)} must be one of {ORDER}")
    specs = {}
    for name in ORDER:
        e = given.get(name, {})
        kind = "object" if name in CUBES else "zone"
        specs[name] = {
            "kind": kind,
            "label": e.get("label", name),
            "height_m": (e.get("height_cm", DEFAULT_OBJ_HEIGHT_CM) if kind == "object" else 0.0) / 100.0,
            "hsv": [tuple(map(tuple, r)) for r in e["hsv"]] if "hsv" in e else DEFAULT_HSV[name],
        }
    return specs


def hsv_ranges_from_samples(hsv_pixels, h_margin=6, sv_margin=40, achromatic_s=50):
    """Fit HSV ranges to sampled pixels [N, 3] (OpenCV HSV). S and V are bounded on both sides (percentiles 5/95
    plus a margin). Hue is circular (period 180): the center is a circular mean, and a range crossing 0/179 is
    split in two. Weakly saturated samples (white/gray/black items) have no meaningful hue, so they get the full
    hue range and are separated by S/V alone."""
    p = np.asarray(hsv_pixels, dtype=float).reshape(-1, 3)
    s_lo = max(0, int(np.percentile(p[:, 1], 5) - sv_margin))
    s_hi = min(255, int(np.percentile(p[:, 1], 95) + sv_margin))
    v_lo = max(0, int(np.percentile(p[:, 2], 5) - sv_margin))
    v_hi = min(255, int(np.percentile(p[:, 2], 95) + sv_margin))
    if np.median(p[:, 1]) < achromatic_s:
        return [((0, s_lo, v_lo), (179, s_hi, v_hi))]
    ang = p[:, 0] * (2 * np.pi / 180.0)
    c = (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * 180.0 / (2 * np.pi)
    dh = (p[:, 0] - c + 90) % 180 - 90                        # signed hue distance to the center
    half = min(float(np.percentile(np.abs(dh), 95)) + h_margin, 40.0)
    lo_h, hi_h = c - half, c + half
    if lo_h < 0:
        return [((0, s_lo, v_lo), (int(hi_h), s_hi, v_hi)), ((int(180 + lo_h), s_lo, v_lo), (179, s_hi, v_hi))]
    if hi_h > 179:
        return [((int(lo_h), s_lo, v_lo), (179, s_hi, v_hi)), ((0, s_lo, v_lo), (int(hi_h - 180), s_hi, v_hi))]
    return [((int(lo_h), s_lo, v_lo), (int(hi_h), s_hi, v_hi))]


def sample_patch(hsv_img, xy_px, radius=6):
    x, y = int(round(xy_px[0])), int(round(xy_px[1]))
    h, w = hsv_img.shape[:2]
    return hsv_img[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)].reshape(-1, 3)


def color_mask(hsv_img, ranges):
    mask = np.zeros(hsv_img.shape[:2], np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv_img, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def largest_blob(mask, min_area_px):
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cent[k] if stats[k, cv2.CC_STAT_AREA] >= min_area_px else None


# ---------------------------------------------------------------------------- clip names
CLIP_RE = re.compile(r"^(red|green|blue)(?:-|_to_)(yellow|purple|orange)(?:_(\d+))?\.(mp4|mov|m4v)$", re.IGNORECASE)


def parse_clip(fname):
    """Task name ("blue-orange") from a demo clip name, or None. Accepts the protocol form "blue-orange_03.mp4"
    and the short form "blue_to_orange.MOV" (number optional, any case of the extension)."""
    m = CLIP_RE.match(fname)
    return f"{m.group(1).lower()}-{m.group(2).lower()}" if m else None


def list_clips(session_dir):
    return sorted(f for f in os.listdir(session_dir) if parse_clip(f))


# ---------------------------------------------------------------------------- manual layout overrides
def layout_json_path(session_dir, clip):
    return os.path.join(session_dir, os.path.splitext(clip)[0] + ".layout.json")


def save_layout_json(path, table_xy_by_name, meta=None):
    with open(path, "w") as f:
        json.dump({"positions_m": {k: [float(v[0]), float(v[1])] for k, v in table_xy_by_name.items()},
                   "order": ORDER, **(meta or {})}, f, indent=2)


def load_layout_json(path):
    """-> {name: xy} for all 6 items (already parallax-corrected, in table meters), or None if absent."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)["positions_m"]
    missing = [n for n in ORDER if n not in d]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return {n: np.asarray(d[n], float) for n in ORDER}
