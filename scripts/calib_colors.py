"""Click each object and zone once in one frame; fit HSV ranges and save them into session.json.

    python scripts/calib_colors.py data/raw_phone/<session> [--clip red-yellow_01.mp4] [--check_all]

Click order: the 3 objects (red, green, blue = your stand-ins as labeled in session.json), then the 3 zones
(yellow, purple, orange). Click on a well-lit, uniformly colored part of each item.
--check_all then runs color detection on the first frame of every clip and reports what it misses, so you can
decide whether to use scripts/click_layout.py.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from human.calibrate import camera_ground_xy, load_session, read_video, video_homographies, warp_topdown  # noqa: E402
from human.clicker import collect_clicks  # noqa: E402
from human.extract import detect_objects  # noqa: E402
from human.objects import ORDER, hsv_ranges_from_samples, sample_patch, session_specs  # noqa: E402
from sim.env import CUBES  # noqa: E402

CLIP_RE = re.compile(r"^([a-z]+-[a-z]+)_(\d+)\.(mp4|mov|MP4|MOV)$")
PPM = 1600


def first_frame_topdown(path, marker_xy, n=15):
    frames, _ = read_video(path, max_frames=n)
    Hs, _ = video_homographies(frames, marker_xy)
    top, A = warp_topdown(frames[0], Hs[0], ppm=PPM)
    return frames[0], Hs[0], top, A


def update_session_json(sdir, ranges):
    path = os.path.join(sdir, "session.json")
    with open(path) as f:
        raw = json.load(f)
    for key, names in (("objects", list(CUBES)), ("zones", [n for n in ORDER if n not in CUBES])):
        entries = {e["name"]: e for e in raw.get(key, [])}
        raw[key] = []
        for n in names:
            e = entries.get(n, {"name": n, "label": n})
            e["hsv"] = [[list(map(int, lo)), list(map(int, hi))] for lo, hi in ranges[n]]
            raw[key].append(e)
    with open(path, "w") as f:
        json.dump(raw, f, indent=2)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--clip", default=None, help="clip to calibrate on (default: first demo clip)")
    p.add_argument("--radius", type=int, default=8, help="sampling patch radius in px (1 px = 0.6 mm)")
    p.add_argument("--check_all", action="store_true")
    p.add_argument("--out", default="outputs/m2")
    a = p.parse_args()
    s = load_session(a.session)
    clips = sorted(f for f in os.listdir(a.session) if CLIP_RE.match(f))
    clip = a.clip or clips[0]
    frame, H, top, A = first_frame_topdown(os.path.join(a.session, clip), s["marker_xy"])
    specs = session_specs(s)
    labels = {n: specs[n]["label"] for n in ORDER}
    clicks, status = collect_clicks(top, ORDER, title=f"colors: {clip}", labels=labels,
                                    hint="click a uniformly colored spot on each item")
    if status != "ok":
        sys.exit(f"aborted ({status}); session.json unchanged")
    hsv = cv2.cvtColor(cv2.GaussianBlur(top, (5, 5), 0), cv2.COLOR_BGR2HSV)
    ranges = {n: hsv_ranges_from_samples(sample_patch(hsv, clicks[n], a.radius)) for n in ORDER}
    for n in ORDER:
        print(f"{n:7s} ({labels[n]}): {ranges[n]}")
    path = update_session_json(a.session, ranges)
    print("saved HSV ranges to", path)

    # verify on the calibration frame
    s = load_session(a.session)
    specs = session_specs(s)
    det = detect_objects(frame, H, specs)  # no parallax here: compare with the clicks in the warped view
    out = os.path.join(a.out, os.path.basename(os.path.normpath(a.session)))
    os.makedirs(out, exist_ok=True)
    vis = top.copy()
    Ainv = np.linalg.inv(A)
    for n in ORDER:
        click_xy = (Ainv @ [*clicks[n], 1])[:2]
        if det[n] is None:
            print(f"  {n:7s}: NOT detected")
            continue
        px = A @ [*det[n], 1]
        cv2.drawMarker(vis, (int(px[0]), int(px[1])), (255, 255, 255), cv2.MARKER_CROSS, 24, 3)
        cv2.putText(vis, n, (int(px[0]) + 8, int(px[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        print(f"  {n:7s}: detected {np.linalg.norm(det[n] - click_xy) * 100:.1f} cm from your click")
    cv2.imwrite(os.path.join(out, "color_calib.png"), vis)
    print("debug image:", os.path.join(out, "color_calib.png"))

    if a.check_all:
        bad = 0
        for c in clips:
            fr, Hc, _, _ = first_frame_topdown(os.path.join(a.session, c), s["marker_xy"])
            d = detect_objects(fr, Hc, specs, camera_h=s["camera_height"], cam_xy=camera_ground_xy(Hc, fr.shape))
            miss = [n for n in ORDER if d[n] is None]
            bad += bool(miss)
            print(f"{c:22s} {'ok' if not miss else 'missing ' + ','.join(miss)}")
        print(f"{len(clips) - bad}/{len(clips)} clips fully detected; use scripts/click_layout.py for the rest")


if __name__ == "__main__":
    main()
