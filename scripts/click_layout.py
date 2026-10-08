"""Manual layout fallback: click the 3 objects and 3 zones on the top-down first frame of each clip.

    python scripts/click_layout.py data/raw_phone/<session> [--redo] [--clips red-yellow_01.mp4 ...]

Order: objects red, green, blue (your stand-ins), then zones yellow, purple, orange. Click the CENTER OF THE TOP
of each object and the center of each zone. Object clicks are parallax-corrected using height_cm from
session.json. Saves <clip>.layout.json next to the clip; process_phone.py then uses it instead of color detection.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from human.calibrate import camera_ground_xy, load_session, read_video, video_homographies, warp_topdown  # noqa: E402
from human.clicker import collect_clicks  # noqa: E402
from human.extract import parallax_correct  # noqa: E402
from human.objects import ORDER, layout_json_path, list_clips, parse_clip, save_layout_json, session_specs  # noqa: E402

PPM = 1600


def clicks_to_table(clicks, A, specs, cam_xy, camera_h):
    """Warped-view pixels -> table meters; object (top-surface) clicks are pulled back for parallax."""
    Ainv = np.linalg.inv(A)
    out = {}
    for n, px in clicks.items():
        xy = (Ainv @ [px[0], px[1], 1.0])[:2]
        if specs[n]["kind"] == "object":
            xy = parallax_correct(xy, np.array(specs[n]["height_m"]), cam_xy, camera_h)
        out[n] = xy
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--redo", action="store_true", help="re-click clips that already have a layout file")
    p.add_argument("--clips", nargs="*", default=None)
    a = p.parse_args()
    s = load_session(a.session)
    specs = session_specs(s)
    labels = {n: specs[n]["label"] for n in ORDER}
    clips = a.clips or list_clips(a.session)
    todo = [c for c in clips if a.redo or not os.path.exists(layout_json_path(a.session, c))]
    print(f"{len(todo)} clips to click ({len(clips) - len(todo)} already done)")
    for i, clip in enumerate(todo):
        frames, _ = read_video(os.path.join(a.session, clip), max_frames=15)
        Hs, _ = video_homographies(frames, s["marker_xy"])
        top, A = warp_topdown(frames[0], Hs[0], ppm=PPM)
        task = parse_clip(clip)
        clicks, status = collect_clicks(top, ORDER, title=f"layout {i + 1}/{len(todo)}: {clip}", labels=labels,
                                        hint=f"task {task}; click top-center of objects, center of zones")
        if status == "quit":
            print("quit; progress so far is saved")
            break
        if status == "skip":
            print(f"{clip}: skipped")
            continue
        xy = clicks_to_table(clicks, A, specs, camera_ground_xy(Hs[0], frames[0].shape), s["camera_height"])
        save_layout_json(layout_json_path(a.session, clip), xy, {"clip": clip, "source": "manual_click"})
        print(f"{clip}: saved")


if __name__ == "__main__":
    main()
