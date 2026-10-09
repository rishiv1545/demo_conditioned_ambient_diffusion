"""Systematic offset between the human pinch point and the object at grasp time, per phone session.

For each processed clip (data/human/<session>/*.npz): the retargeted gripper target at the moment the gripper first
closes on the object, minus the object's position from the layout (clicked or detected). A consistent mean offset
(large relative to the spread) is a pipeline bias, e.g. the hand stopping beside the object while the gripper must be
centered above it; put the negated residual into session.json "pinch_offset_cm" and re-run process_phone.py.

    python scripts/pinch_bias.py data/human/3 [--session data/raw_phone/3]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from policy.data import _read, layout_from_array  # noqa: E402
from sim.env import task_name  # noqa: E402


def grasp_offsets(paths):
    rows = []
    for f in sorted(paths):
        ep = _read(f)
        ee, g = ep["raw_traj"], ep["raw_gripper"]
        lay = layout_from_array(ep["layout"])
        c = ep["task"][0]
        closed = g > 0.5
        # first open -> closed switch after the gripper has been opened (it starts closed at rest)
        opened = np.flatnonzero(~closed)
        if not len(opened):
            continue
        sw = [t for t in range(opened[0] + 1, len(g)) if closed[t] and not closed[t - 1]]
        if not sw:
            continue
        t = sw[0]
        d = ee[t, :2] - lay["cubes"][c]
        rows.append({"clip": os.path.basename(f), "task": task_name(ep["task"]), "dx": d[0], "dy": d[1],
                     "success": bool(ep["success"]), "approach": ee[max(0, t - 10), :2] - ee[t, :2]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--session", default=None, help="raw session dir: print the corrected pinch_offset_cm")
    a = p.parse_args()
    rows = grasp_offsets(glob.glob(os.path.join(a.data, "*.npz")))
    if not rows:
        raise SystemExit("no clips with a grasp")
    d = np.array([[r["dx"], r["dy"]] for r in rows]) * 100
    for r, (dx, dy) in zip(rows, d):
        print(f"{r['clip']:24s} {r['task']:13s} dx {dx:+5.1f} dy {dy:+5.1f} cm  {'ok' if r['success'] else 'fail'}")
    m, sd = d.mean(0), d.std(0)
    med = np.median(d, 0)
    se = sd / np.sqrt(len(d))
    print(f"\n{len(d)} grasps: mean dx {m[0]:+.2f} dy {m[1]:+.2f} cm (std {sd[0]:.2f}, {sd[1]:.2f}; "
          f"s.e. {se[0]:.2f}, {se[1]:.2f}); median dx {med[0]:+.2f} dy {med[1]:+.2f} cm")
    by_cube = {}
    for r, v in zip(rows, d):
        by_cube.setdefault(r["task"].split("-")[0], []).append(v)
    for c, v in by_cube.items():
        v = np.array(v)
        print(f"  {c:6s} n={len(v):2d} median dx {np.median(v[:, 0]):+.2f} dy {np.median(v[:, 1]):+.2f} cm")
    if a.session:
        cur = np.array(json.load(open(os.path.join(a.session, "session.json"))).get("pinch_offset_cm", [0.0, 0.0]))
        print(f"current pinch_offset_cm {cur.tolist()} -> suggested {np.round(cur - med, 2).tolist()} (median-based)")


if __name__ == "__main__":
    main()
