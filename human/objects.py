"""Per-session object/zone specs, HSV ranges from clicked samples, color detection and manual layout overrides.

Task names in clip file names always use the canonical names (red/green/blue cubes, yellow/purple/orange zones).
session.json maps each canonical name to the real household object and its color:

    "objects": [{"name": "red",    "label": "red Lego brick", "height_cm": 2.0, "hsv": [[[0,120,70],[8,255,255]]]}, ...],
    "zones":   [{"name": "yellow", "label": "yellow sticky note",            "hsv": [[[21,100,120],[35,255,255]]]}, ...]

"hsv" is a list of [lo, hi] ranges in OpenCV units (H 0-179, S and V 0-255); several ranges handle hue wrap-around.
Missing entries fall back to the defaults below. A per-clip "<clip>.layout.json" (from scripts/click_layout.py)
overrides color detection for that clip.

Single-object sessions (the constrained blue cube -> red zone task, env v3s) set "scene": "single" and list one
object ("blue") and one zone ("red"); clips are named blue_to_red_NN.MOV. Their items are keyed "blue" and
"red_zone" (the red cube's name is taken), and click_layout.py asks for 2 clicks per clip.
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
ORDER = list(CUBES) + list(ZONES[:3])  # fixed click order: 3 objects, then 3 zones
SINGLE_ORDER = ["blue", "red_zone"]       # single-object scene: the blue object, then the red zone


def is_single(session):
    return session.get("scene", "multi") == "single"


def session_order(session):
    return SINGLE_ORDER if is_single(session) else ORDER


def item_key(kind, name):
    """Spec/layout key of a session.json entry: the canonical name, except the single scene's red zone."""
    return "red_zone" if (kind, name) == ("zone", "red") else name


def item_name(key):
    return "red" if key == "red_zone" else key


def session_specs(session):
    """{key: {"kind": "object"|"zone", "label", "height_m", "hsv": [(lo, hi), ...]}} for the session's items in
    click order (6 for the multi-object scene, 2 for the single one)."""
    order = session_order(session)
    given = {item_key("object", e["name"]): e for e in session.get("objects", [])}
    given.update({item_key("zone", e["name"]): e for e in session.get("zones", [])})
    unknown = set(given) - set(order)
    if unknown:
        raise ValueError(f"session.json names {sorted(unknown)} must be one of {[item_name(k) for k in order]}")
    specs = {}
    for key in order:
        e = given.get(key, {})
        kind = "zone" if key in ZONES[:3] or key == "red_zone" else "object"
        specs[key] = {
            "kind": kind,
            "label": e.get("label", item_name(key)),
            "height_m": (e.get("height_cm", DEFAULT_OBJ_HEIGHT_CM) if kind == "object" else 0.0) / 100.0,
            "hsv": [tuple(map(tuple, r)) for r in e["hsv"]] if "hsv" in e else DEFAULT_HSV[item_name(key)],
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
CLIP_RE = re.compile(r"^(red|green|blue)(?:-|_to_)(yellow|purple|orange|red)(_rec)?(?:_(\d+))?\.(mp4|mov|m4v)$",
                     re.IGNORECASE)


def parse_clip(fname):
    """Task name ("blue-orange") from a demo clip name, or None. Accepts the protocol form "blue-orange_03.mp4"
    and the short form "blue_to_orange.MOV" (number optional, any case of the extension), and "miss and correct"
    clips with _rec after the task ("red_to_yellow_rec_01.MOV")."""
    m = CLIP_RE.match(fname)
    return f"{m.group(1).lower()}-{m.group(2).lower()}" if m else None


def is_recovery_clip(fname):
    """"Miss and correct" clips are named like the demos with _rec after the task: red_to_yellow_rec_01.MOV."""
    m = CLIP_RE.match(fname)
    return bool(m and m.group(3))


def list_clips(session_dir):
    return sorted(f for f in os.listdir(session_dir) if parse_clip(f))


# ---------------------------------------------------------------------------- manual layout overrides
def layout_json_path(session_dir, clip):
    return os.path.join(session_dir, os.path.splitext(clip)[0] + ".layout.json")


def save_layout_json(path, table_xy_by_name, meta=None, order=ORDER):
    with open(path, "w") as f:
        json.dump({"positions_m": {k: [float(v[0]), float(v[1])] for k, v in table_xy_by_name.items()},
                   "order": list(order), **(meta or {})}, f, indent=2)


def load_layout_json(path, order=ORDER):
    """-> {name: xy} for all items in order (already parallax-corrected, in table meters), or None if absent."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)["positions_m"]
    missing = [n for n in order if n not in d]
    if missing:
        raise ValueError(f"{path} is missing {missing}")
    return {n: np.asarray(d[n], float) for n in order}
