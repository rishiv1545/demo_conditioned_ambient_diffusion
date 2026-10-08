"""Phone -> robot pipeline over one recording session: calibrate, extract, retarget, replay, save, summarize.

    python scripts/process_phone.py data/raw_phone/<session> --out_data data/human --side_by_side 3
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from human.calibrate import load_session, read_video, video_homographies  # noqa: E402
from human.extract import (calib_hand_size, extract_clip, height_from_size, overlay_frame,  # noqa: E402
                           plot_traj)
from human.objects import layout_json_path, load_layout_json, session_specs  # noqa: E402
from human.replay import replay, save_human_episode  # noqa: E402
from human.retarget import RetargetConfig, retarget  # noqa: E402
from sim.env import CUBES, ZONES, PickPlaceEnv, parse_task, task_name  # noqa: E402

CLIP_RE = re.compile(r"^([a-z]+-[a-z]+)_(\d+)\.(mp4|mov|MP4|MOV)$")


def calibrate_session(sdir, s, out):
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
    cal = {"s0_m": s0, "s0_std_m": res["calib_table"][1], "s_box_m": res["calib_box"][0],
           "z_box_est_m": float(z_box), "z_box_true_m": s["box_height"],
           "z_box_err_m": float(z_box - s["box_height"])}
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)
    print(f"hand size on table {s0 * 100:.2f} cm; box height est {z_box * 100:.1f} cm vs true "
          f"{s['box_height'] * 100:.1f} cm (error {(z_box - s['box_height']) * 100:+.1f} cm)")
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
        pf = cv2.resize(pf, (int(pf.shape[1] * h / pf.shape[0]), h))
        out.append(np.concatenate([pf, sf], 1))
    imageio.mimsave(path, out, fps=hz, macro_block_size=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("session")
    p.add_argument("--out_data", default="data/human")
    p.add_argument("--out", default="outputs/m2")
    p.add_argument("--side_by_side", type=int, default=3, help="number of clips to render side-by-side videos for")
    p.add_argument("--z_offset", type=float, default=0.0)
    p.add_argument("--grip_lo", type=float, default=0.65)
    p.add_argument("--grip_hi", type=float, default=0.9)
    p.add_argument("--min_tracked", type=float, default=0.8, help="min fraction of frames with a hand")
    a = p.parse_args()
    sname = os.path.basename(os.path.normpath(a.session))
    out = os.path.join(a.out, sname)
    os.makedirs(out, exist_ok=True)
    s = load_session(a.session)
    cal = calibrate_session(a.session, s, out)
    env = PickPlaceEnv()
    clips = sorted(f for f in os.listdir(a.session) if CLIP_RE.match(f))
    specs = session_specs(s)
    print("objects:", ", ".join(f"{n} = {specs[n]['label']}" for n in specs))
    rows = []
    for ci, clip in enumerate(clips):
        tname = CLIP_RE.match(clip).group(1)
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
            ee, g, _ = retarget(ex, RetargetConfig(z_offset=z_off))
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
