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

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from human.calibrate import load_session, read_video, video_homographies  # noqa: E402
from human.extract import (calib_hand_size, extract_clip, height_from_size, overlay_frame, track_hand,  # noqa: E402
                           plot_traj)
from human.objects import layout_json_path, list_clips, load_layout_json, parse_clip, session_specs  # noqa: E402
from human.camera_match import apply_camera, estimate_phone_camera  # noqa: E402
from human.replay import replay, save_human_episode  # noqa: E402
from human.retarget import RetargetConfig, retarget  # noqa: E402
from sim.env import CUBES, ZONES, PickPlaceEnv, env_cfg_dict, parse_task, task_name  # noqa: E402



def calibrate_session(sdir, s, out, fit_camera_height=True):
    res = {}
    # "calib_from": another session folder (relative to this one) whose calibration clips to reuse, for sessions
    # recorded with the same camera mount and the same person
    csrc = os.path.normpath(os.path.join(sdir, s["calib_from"])) if s.get("calib_from") else sdir
    for name in ("calib_table", "calib_box"):
        path = next(iter(glob.glob(os.path.join(csrc, name + ".*"))), None)
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


def track_cache_path(cache_dir, session_dir, clip):
    """Cache file for a clip's hand landmarks + homographies, keyed by the video file's size and mtime."""
    st = os.stat(os.path.join(session_dir, clip))
    return os.path.join(cache_dir, f"{os.path.splitext(clip)[0]}_{st.st_size}_{int(st.st_mtime)}.npz")


def track_clip(args):
    """Worker: hand landmarks and per-frame homographies of one clip (the slow part), saved to the cache."""
    session_dir, clip, marker_xy, hand, path = args
    if os.path.exists(path):
        return clip, "cached"
    try:
        frames, fps = read_video(os.path.join(session_dir, clip))
        Hs, _ = video_homographies(frames, marker_xy)
        lm = track_hand(frames, fps, hand)
        np.savez(path + ".tmp.npz", Hs=Hs, lm_px=lm, fps=fps)
        os.replace(path + ".tmp.npz", path)
        return clip, "tracked"
    except Exception as e:  # recorded again (with the reason) by the main loop
        return clip, f"error: {e}"


def side_by_side(phone_frames, fps, ex, sim_frames, path, times, hz=10.0, caption=""):
    """[phone with overlays | sim from the phone's viewpoint | sim front view], aligned by the source time of each
    replay step (the phone panel freezes during the inserted dwells)."""
    out = []
    h = sim_frames[0].shape[0]
    for k, sf in enumerate(sim_frames):
        tk = times[min(k, len(times) - 1)]
        i = min(int(round(tk * fps)), len(phone_frames) - 1)
        pf = overlay_frame(phone_frames[i], ex["lm_px"][i], ex["xy"][i], ex["z"][i], ex["grip"][i], i)
        pf = cv2.cvtColor(pf, cv2.COLOR_BGR2RGB)
        pf = cv2.resize(pf, (2 * round(pf.shape[1] * h / pf.shape[0] / 2), h))  # even width for H.264
        fr = np.concatenate([pf, sf], 1)
        if caption:
            bar = np.full((28, fr.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            fr = np.concatenate([bar, fr], 0)
        out.append(fr)
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
    p.add_argument("--grip_mode", default="relative", choices=["relative", "fixed"],
                   help="relative: per-clip thresholds (spread level / grasp opening); fixed: --grip_lo/--grip_hi")
    p.add_argument("--dwell_s", type=float, default=0.5, help="EE hold before/after each gripper switch (0 = off)")
    p.add_argument("--grip_lo", type=float, default=0.65)
    p.add_argument("--grip_hi", type=float, default=0.9)
    p.add_argument("--min_tracked", type=float, default=0.8, help="min fraction of frames with a hand")
    p.add_argument("--approach_clear", type=float, default=0.05,
                   help="approach the grasp from this high above, descend vertically; rise as high after release "
                        "(0 = follow the hand path exactly)")
    p.add_argument("--workers", type=int, default=4, help="parallel hand tracking (each holds one clip in RAM)")
    a = p.parse_args()
    sname = os.path.basename(os.path.normpath(a.session))
    out = os.path.join(a.out, sname)
    os.makedirs(out, exist_ok=True)
    s = load_session(a.session)
    cal = calibrate_session(a.session, s, out, bool(a.fit_camera_height))
    env = PickPlaceEnv()
    # sim camera matching the phone's viewpoint, from the markers in the (table) calibration clip
    # from this session's own footage (its table clip if it has one, else the demo clips), so a session that reuses
    # another's calibration (calib_from) still gets its own camera pose; for sessions 1 and 3 the two agree
    own = glob.glob(os.path.join(a.session, "calib_table.*"))
    cam = None
    for src in own + [os.path.join(a.session, c) for c in list_clips(a.session)]:
        cframes, _ = read_video(src, max_frames=30)
        try:
            cam = estimate_phone_camera(cframes, s["marker_xy"], s["camera_height"])
            cam["source"] = os.path.basename(src)
            break
        except RuntimeError:   # a marker hidden in the first frames: try the next clip
            continue
    if cam is None:
        raise RuntimeError("no clip shows all 4 markers in its first 30 frames")
    apply_camera(env, cam)
    cal["phone_camera"] = cam
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(cal, f, indent=2)
    print(f"phone camera from markers: height {cam['pos'][2] * 100:.1f} cm, tilt {cam['tilt_deg']:.1f} deg, "
          f"fov {cam['fovx']:.0f}x{cam['fovy']:.0f} deg, marker reprojection error {cam['reproj_err_px']:.1f} px "
          f"(from {cam['source']})")
    vh = 480
    vw = 2 * round(cam["width"] * vh / cam["height"] / 2)

    def render_pair(e):
        return np.concatenate([e.render_camera("phone_match", vh, vw), e.render_camera("front", vh, 640)], 1)
    clips = list_clips(a.session)
    specs = session_specs(s)
    print("objects:", ", ".join(f"{n} = {specs[n]['label']}" for n in specs))
    # pinch point -> gripper target correction (table frame, cm), e.g. the hand stops beside the object while the
    # gripper must be centered above it; estimated from the clicked object positions (scripts/pinch_bias.py)
    pinch_off = np.array(s.get("pinch_offset_cm", [0.0, 0.0])) / 100.0
    if np.any(pinch_off):
        print(f"pinch offset correction: {pinch_off * 100} cm (session.json pinch_offset_cm)")
    cache_dir = os.path.join(out, "track_cache")
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(a.session, c, s["marker_xy"], s.get("hand", "right"), track_cache_path(cache_dir, a.session, c))
            for c in clips]
    with ProcessPoolExecutor(a.workers) as pool:
        for clip, st in pool.map(track_clip, jobs):
            print(f"track {clip}: {st}", flush=True)
    rows = []
    for ci, clip in enumerate(clips):
        tname = parse_clip(clip)
        task = parse_task(tname)
        name = os.path.splitext(clip)[0]
        row = {"clip": clip, "task": tname, "layout_source": "", "extracted": 0, "replay_success": 0, "track_err": np.nan, "reason": ""}
        try:
            frames, fps = read_video(os.path.join(a.session, clip))
            tc = np.load(track_cache_path(cache_dir, a.session, clip))
            Hs = tc["Hs"]
            ex = extract_clip(frames, fps, Hs, s, cal["s0_m"], a.grip_lo, a.grip_hi, grip_mode=a.grip_mode,
                              rest_closed=env.cfg.start_gripper_closed, lm_px=tc["lm_px"])
            ex["xy"] = ex["xy"] + pinch_off
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
                if ci < a.side_by_side:   # still show the recording with the hand overlay, to see what went wrong
                    step = max(1, int(round(fps / 10)))
                    vid = []
                    for i in range(0, len(frames), step):
                        pf = overlay_frame(frames[i], ex["lm_px"][i], ex["xy"][i], ex["z"][i], ex["grip"][i], i)
                        pf = cv2.resize(pf, (2 * round(pf.shape[1] * 480 / pf.shape[0] / 2), 480))
                        bar = np.full((28, pf.shape[1], 3), 255, np.uint8)
                        cv2.putText(bar, f"{clip}: {row['reason']}"[:60], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 0, 0), 1, cv2.LINE_AA)
                        vid.append(cv2.cvtColor(np.concatenate([bar, pf], 0), cv2.COLOR_BGR2RGB))
                    imageio.mimsave(os.path.join(out, f"{name}_phone_only.mp4"), vid, fps=10, macro_block_size=1)
                continue
            row["extracted"] = 1
            # the human pinches the real object at about half its height; the sim cube's center is at 2 cm
            z_off = a.z_offset - (specs[CUBES[task[0]]]["height_m"] / 2 - 0.02)
            h_obj = specs[CUBES[task[0]]]["height_m"]
            ee, g, tt = retarget(ex, RetargetConfig(z_offset=z_off, contact_z=h_obj / 2 if a.contact_z else None,
                                                   dwell_s=a.dwell_s, approach_clear=a.approach_clear))
            render = render_pair if ci < a.side_by_side else None
            r = replay(env, task, lay, ee, g, render=render)
            # did the human complete the task? (target cube center inside target zone in the last frame)
            end = ex["objects_end"]
            human_done = (end.get(CUBES[task[0]]) is not None and
                          bool(np.all(np.abs(end[CUBES[task[0]]] - lay["zones"][task[1]]) <= 0.05)))
            meta = {"clip": clip, "session": sname, "fps": fps, "tracked_frac": tracked,
                    "human_completed": human_done, "long_gaps": int((~ex["valid"]).sum()),
                    "z_offset": z_off, "layout_source": row["layout_source"], "approach_clear": a.approach_clear,
                    "env_cfg": env_cfg_dict(env.cfg), "grip_info": ex["grip_info"]}
            save_human_episode(os.path.join(a.out_data, sname, f"{name}.npz"), r, task, lay,
                               raw_traj=ee, raw_gripper=g, meta=meta)
            row.update(replay_success=int(r["success"]), track_err=r["track_err"], human_completed=int(human_done))
            if render:
                cap = (f"{clip}  task {tname}  replay {'SUCCESS' if r['success'] else 'FAIL'}  |  phone (hand: "
                       f"red = pinch point) | sim, phone viewpoint | sim, front")
                side_by_side(frames, fps, ex, r["frames"], os.path.join(out, f"{name}_side_by_side.mp4"), tt,
                             caption=cap)
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
