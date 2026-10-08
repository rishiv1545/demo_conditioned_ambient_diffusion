"""Phone -> robot pipeline over one recording session: calibrate, extract, retarget, replay, save, summarize.

    python scripts/process_phone.py data/raw_phone/<session> --out_data data/human --side_by_side 3
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from human.calibrate import load_session, read_video, video_homographies  # noqa: E402
from human.extract import (calib_hand_size, extract_clip, height_from_size, overlay_frame,  # noqa: E402
                           plot_traj)
from human.objects import layout_json_path, list_clips, load_layout_json, parse_clip, session_specs  # noqa: E402
from human.replay import replay, save_human_episode  # noqa: E402
from human.retarget import RetargetConfig, retarget  # noqa: E402
from sim.env import CUBES, ZONES, PickPlaceEnv, parse_task, task_name  # noqa: E402



def calibrate_session(sdir, s, out, fit_camera_height=True):
    res = {}
    for name in ("calib_table", "calib_box"):
        path = next(iter(glob.glob(os.path.join(sdir, name + ".*"))), None)
        if path is None:
            raise FileNotFoundError(f"{name}.mp4 missing in {sdir}")
        frames, fps = read_video(path)
        Hs, _ = video_homographies(frames, s["marker_xy"])
        res[name] = calib_hand_size(frames, fps, Hs)
    s0 = res["calib_table"][0]
    z_box = height_from_size(res["calib_box"][0], s0, s["camera_height"])
    known = s["box_height"] is not None
    cal = {"s0_m": s0, "s0_std_m": res["calib_table"][1], "s_box_m": res["calib_box"][0],
           "camera_height_m": s["camera_height"], "z_box_est_m": float(z_box), "z_box_true_m": s["box_height"],
           "z_box_err_m": float(z_box - s["box_height"]) if known else None}
    if known and fit_camera_height:
        # two-point calibration: s = L*H/(H - z) at z=0 (s0) and z=box gives H = box / (1 - s0/s_box).
        # The box then no longer validates the height model; the fitted H is cross-checked against the
        # marker-based (solvePnP) estimate instead.
        h_fit = s["box_height"] / (1.0 - s0 / res["calib_box"][0])
        cal.update(camera_height_given_m=s["camera_height"], camera_height_fit_m=float(h_fit))
        print(f"camera height fitted from the box clip: {h_fit * 100:.1f} cm (session.json: "
              f"{s['camera_height'] * 100:.1f} cm; with that, the box was estimated at {z_box * 100:.1f} cm, "
              f"error {(z_box - s['box_height']) * 100:+.1f} cm)")
        s["camera_height"] = h_fit
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)
    if known:
        print(f"hand size on table {s0 * 100:.2f} cm; box height est {z_box * 100:.1f} cm vs true "
              f"{s['box_height'] * 100:.1f} cm (error {(z_box - s['box_height']) * 100:+.1f} cm)")
    else:
        print(f"hand size on table {s0 * 100:.2f} cm; box height est {z_box * 100:.1f} cm "
              f"(true box height not given in session.json: error unknown)")
    return cal


def layout_from_objects(objs):
    if any(objs.get(c) is None for c in CUBES + ZONES):
        return None
    return {"cubes": np.stack([objs[c] for c in CUBES]), "zones": np.stack([objs[z] for z in ZONES])}


def side_by_side(phone_frames, fps, ex, sim_frames, path, hz=10.0):
    out = []
    h = sim_frames[0].shape[0]
    for k, sf in enumerate(sim_frames):
        i = min(int(round(k / hz * fps)), len(phone_frames) - 1)
        pf = overlay_frame(phone_frames[i], ex["lm_px"][i], ex["xy"][i], ex["z"][i], ex["grip"][i], i)
        pf = cv2.cvtColor(pf, cv2.COLOR_BGR2RGB)
        pf = cv2.resize(pf, (2 * round(pf.shape[1] * h / pf.shape[0] / 2), h))  # even width for H.264
        out.append(np.concatenate([pf, sf], 1))
    imageio.mimsave(path, out, fps=hz, macro_block_size=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--out_data", default="data/human")
    p.add_argument("--out", default="outputs/m2")
    p.add_argument("--side_by_side", type=int, default=3, help="number of clips to render side-by-side videos for")
    p.add_argument("--z_offset", type=float, default=0.0)
    p.add_argument("--fit_camera_height", type=int, default=0,
                   help="1: replace camera_height_cm by a fit from the box clip (when the height wasn't measured); "
                        "0: use the measured height and report the box-height error as validation")
    p.add_argument("--contact_z", type=int, default=1,
                   help="re-anchor z so the pinch point is at the object's half height when the grip closes/opens")
    p.add_argument("--grip_lo", type=float, default=0.65)
    p.add_argument("--grip_hi", type=float, default=0.9)
    p.add_argument("--min_tracked", type=float, default=0.8, help="min fraction of frames with a hand")
    a = p.parse_args()
    sname = os.path.basename(os.path.normpath(a.session))
    out = os.path.join(a.out, sname)
    os.makedirs(out, exist_ok=True)
    s = load_session(a.session)
    cal = calibrate_session(a.session, s, out, bool(a.fit_camera_height))
    env = PickPlaceEnv()
    clips = list_clips(a.session)
    specs = session_specs(s)
    print("objects:", ", ".join(f"{n} = {specs[n]['label']}" for n in specs))
    rows = []
    for ci, clip in enumerate(clips):
        tname = parse_clip(clip)
        task = parse_task(tname)
        name = os.path.splitext(clip)[0]
        row = {"clip": clip, "task": tname, "layout_source": "", "extracted": 0, "replay_success": 0, "track_err": np.nan, "reason": ""}
        try:
            frames, fps = read_video(os.path.join(a.session, clip))
            Hs, _ = video_homographies(frames, s["marker_xy"])
            ex = extract_clip(frames, fps, Hs, s, cal["s0_m"], a.grip_lo, a.grip_hi)
            plot_traj(ex, os.path.join(out, f"{name}_traj.png"), title=clip)
            manual = load_layout_json(layout_json_path(a.session, clip))
            lay = layout_from_objects(manual if manual is not None else ex["objects"])
            row["layout_source"] = "manual_click" if manual is not None else "color"
            tracked = float(ex["tracked"].mean())
            if lay is None:
                row["reason"] = ("color detection missed " + ",".join(k for k, v in ex["objects"].items() if v is None)
                                 + " (fix: scripts/calib_colors.py or scripts/click_layout.py)")
            elif tracked < a.min_tracked:
                row["reason"] = f"hand tracked in {tracked:.0%} of frames"
            elif not ex["grip"].any():
                row["reason"] = "no grasp detected"
            if row["reason"]:
                rows.append(row)
                print(f"{clip}: extraction failed ({row['reason']})")
                continue
            row["extracted"] = 1
            # the human pinches the real object at about half its height; the sim cube's center is at 2 cm
            z_off = a.z_offset - (specs[CUBES[task[0]]]["height_m"] / 2 - 0.02)
            h_obj = specs[CUBES[task[0]]]["height_m"]
            ee, g, _ = retarget(ex, RetargetConfig(z_offset=z_off, contact_z=h_obj / 2 if a.contact_z else None))
            render = "front" if ci < a.side_by_side else None
            r = replay(env, task, lay, ee, g, render=render)
            # did the human complete the task? (target cube center inside target zone in the last frame)
            end = ex["objects_end"]
            human_done = (end.get(CUBES[task[0]]) is not None and
                          bool(np.all(np.abs(end[CUBES[task[0]]] - lay["zones"][task[1]]) <= 0.05)))
            meta = {"clip": clip, "session": sname, "fps": fps, "tracked_frac": tracked,
                    "human_completed": human_done, "long_gaps": int((~ex["valid"]).sum()),
                    "z_offset": z_off, "layout_source": row["layout_source"]}
            save_human_episode(os.path.join(a.out_data, sname, f"{name}.npz"), r, task, lay,
                               raw_traj=ee, raw_gripper=g, meta=meta)
            row.update(replay_success=int(r["success"]), track_err=r["track_err"], human_completed=int(human_done))
            if render:
                side_by_side(frames, fps, ex, r["frames"], os.path.join(out, f"{name}_side_by_side.mp4"))
            print(f"{clip}: replay {'SUCCESS' if r['success'] else 'fail'}  track err {r['track_err'] * 1000:.1f} mm")
        except Exception as e:  # keep going; record the failure
            row["reason"] = f"error: {e}"
            print(f"{clip}: {row['reason']}")
        rows.append(row)

    with open(os.path.join(out, "clips.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "task", "layout_source", "extracted", "replay_success",
                                          "human_completed", "track_err", "reason"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'task':14s} {'clips':>5s} {'extract':>8s} {'replay':>7s} {'track err (mm)':>15s}")
    summ = []
    for t in sorted({r["task"] for r in rows}, key=lambda n: parse_task(n)):
        rs = [r for r in rows if r["task"] == t]
        ext = np.mean([r["extracted"] for r in rs])
        rep = np.mean([r["replay_success"] for r in rs])
        te = np.nanmean([r["track_err"] for r in rs]) if any(r["extracted"] for r in rs) else np.nan
        summ.append([t, len(rs), ext, rep, te * 1000])
        print(f"{t:14s} {len(rs):5d} {ext:8.2f} {rep:7.2f} {te * 1000:15.1f}")
    ext_all = np.mean([r["extracted"] for r in rows]) if rows else 0
    rep_all = np.mean([r["replay_success"] for r in rows]) if rows else 0
    print(f"{'all':14s} {len(rows):5d} {ext_all:8.2f} {rep_all:7.2f}")
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "clips", "extraction_rate", "replay_success_rate", "mean_track_err_mm"])
        w.writerows(summ)
        w.writerow(["all", len(rows), f"{ext_all:.3f}", f"{rep_all:.3f}", ""])


if __name__ == "__main__":
    main()
