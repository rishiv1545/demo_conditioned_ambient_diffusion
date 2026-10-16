"""README media for the single-object phone task (session 5): side-by-side videos (GIF + MP4) and stills.

    python scripts/make_media.py --out docs/media

  phone_vs_sim.{gif,mp4}            phone clip | its sim replay from the phone's viewpoint (time-aligned)
  phone_frame.jpg, sim_frame.jpg    one instant of that pair (the approach)
  rollout.{gif,mp4}                 the all-100 policy (P5_naive_all step 15000) on an evaluation layout
  phone_replay_vs_clean_{a,b}       a phone-derived sim demo | a clean sim-expert demo with the nearest cube start
  quality_tags                      highest vs lowest quality tag among the first 15 clips
  ambient_clean_vs_noisy            a clean clip (t_min 0) | a noisy clip (t_min near 0.9) of the 15-clip split
  vn_before_after                   the same clip replayed without | with velocity normalization
Sim panels are the front camera unless noted; the demos are replayed from their stored actions.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from policy.data import layout_from_array  # noqa: E402
from sim.env import EnvConfig, PickPlaceEnv  # noqa: E402

SESSION = "data/raw_phone/5"
CROP = (0.172, 0.657)  # rows kept, as fractions of the portrait frame height: the marker rectangle


def rows(h):
    """Even row range [r0, r1) of CROP for a frame of height h."""
    r0 = int(round(CROP[0] * h / 2)) * 2
    return r0, r0 + int(round((CROP[1] - CROP[0]) * h / 2)) * 2
SIM = 480              # sim panel size (front camera)
HZ = 10


def env_for(meta_cfg):
    return PickPlaceEnv(EnvConfig(**meta_cfg))


def replay_npz(path, camera="front", size=SIM):
    """Replay a stored episode (phone replay or sim expert) from its actions; returns frames, success."""
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"])) if "meta" in d.files and str(d["meta"]) else {}
    env = env_for(meta.get("env_cfg", {"scene": "single"}))
    kw = {"ee_start": d["raw_traj"][0]} if "raw_traj" in d.files else {}
    env.reset(task=tuple(int(v) for v in d["task"]), layout=layout_from_array(d["layout"]), **kw)
    frames, ok = [env.render_camera(camera, size, size)], False
    for a in d["action"]:
        _, ok = env.step(a)
        frames.append(env.render_camera(camera, size, size))
    env.close()
    return frames, bool(ok), bool(d["success"])


def label(frame, top, bottom=None):
    """White bar with a title above the panel (and an optional second line)."""
    lines = [top] + ([bottom] if bottom else [])
    bar = np.full((26 * len(lines) + 6, frame.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return np.concatenate([bar, frame], 0)


def pair(left, right, gap=8):
    """Concatenate two frame lists side by side; the shorter one holds its last frame."""
    n = max(len(left), len(right))
    out = []
    for k in range(n):
        a, b = left[min(k, len(left) - 1)], right[min(k, len(right) - 1)]
        h = max(a.shape[0], b.shape[0])
        pad = lambda f: np.concatenate([f, np.full((h - f.shape[0], f.shape[1], 3), 255, np.uint8)], 0)
        out.append(np.concatenate([pad(a), np.full((h, gap, 3), 255, np.uint8), pad(b)], 1))
    return out


def save(frames, path, fps=HZ, gif_width=720):
    """MP4 (H.264) and a palette GIF at gif_width."""
    imageio.mimsave(path + ".mp4", frames, fps=fps, macro_block_size=1)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path + ".mp4", "-vf",
                    f"fps={fps},scale={gif_width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
                    "[b][p]paletteuse=dither=bayer:bayer_scale=4", path + ".gif"], check=True)
    print(f"wrote {path}.gif ({os.path.getsize(path + '.gif') / 1e6:.1f} MB), .mp4, {len(frames)} frames")


def phone_vs_sim(clip, out):
    """Re-run the phone pipeline for one clip (cached hand tracking; data/human is not touched) and render the
    phone video next to the sim replay from the matched phone camera, aligned by each replay step's source time."""
    import process_phone as pp
    from human.calibrate import load_session, read_video
    from human.camera_match import apply_camera, estimate_phone_camera
    from human.extract import extract_clip
    from human.objects import layout_json_path, load_layout_json, session_order, session_specs
    from human.replay import replay
    from human.retarget import RetargetConfig, retarget
    from sim.env import CUBES, parse_task
    from human.objects import parse_clip
    s = load_session(SESSION)
    scratch = os.path.join(out, "_pipeline")
    os.makedirs(scratch, exist_ok=True)
    cal = pp.calibrate_session(SESSION, s, scratch, False)
    env = PickPlaceEnv(EnvConfig(scene="single"))
    cam = None   # as process_phone.py: the session's table clip if it has one, else the first clip showing all markers
    for src in glob.glob(os.path.join(SESSION, "calib_table.*")) + [os.path.join(SESSION, c) for c in pp.list_clips(SESSION)]:
        try:
            cam = estimate_phone_camera(read_video(src, max_frames=30)[0], s["marker_xy"], s["camera_height"])
            break
        except RuntimeError:
            continue
    apply_camera(env, cam)
    d = np.load(f"data/human/5/{os.path.splitext(clip)[0]}.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    frames, fps = read_video(os.path.join(SESSION, clip))
    tc = np.load(pp.track_cache_path("outputs/m2/5/track_cache", SESSION, clip))
    ex = extract_clip(frames, fps, tc["Hs"], s, cal["s0_m"], 0.65, 0.9, grip_mode="relative",
                      rest_closed=env.cfg.start_gripper_closed, lm_px=tc["lm_px"])
    ex["xy"] = ex["xy"] + np.array(s.get("pinch_offset_cm", [0.0, 0.0])) / 100.0
    task = parse_task(parse_clip(clip))
    manual = load_layout_json(layout_json_path(SESSION, clip), session_order(s))
    lay = pp.layout_from_objects(manual if manual is not None else ex["objects"], env)
    h_obj = session_specs(s)[CUBES[task[0]]]["height_m"]
    ee, g, tt = retarget(ex, RetargetConfig(z_offset=meta["z_offset"], contact_z=h_obj / 2, dwell_s=0.5,
                                           approach_clear=meta["approach_clear"], carry_lift=meta["carry_lift"],
                                           retreat_keep_radius=meta["retreat_keep_radius"]))
    ph, pw = 960, 540   # the phone camera's full portrait frame at half resolution; crop like the phone video
    c0, c1 = rows(ph)
    p0, p1 = rows(frames[0].shape[0])
    r = replay(env, task, lay, ee, g, render=lambda e: e.render_camera("phone_match", ph, pw)[c0:c1])
    assert abs(len(r["action"]) - len(d["action"])) <= 1 and r["success"] == bool(d["success"]), \
        "re-run replay differs from the stored episode"
    left, right = [], []
    for k, sf in enumerate(r["frames"]):
        i = min(int(round(tt[min(k, len(tt) - 1)] * fps)), len(frames) - 1)
        pf = cv2.cvtColor(frames[i][p0:p1], cv2.COLOR_BGR2RGB)
        pf = cv2.resize(pf, (sf.shape[1], sf.shape[0]), interpolation=cv2.INTER_AREA)
        left.append(label(pf, f"phone recording ({clip})"))
        right.append(label(sf, "its sim replay, phone viewpoint"))
    save(pair(left, right), os.path.join(out, "phone_vs_sim"))
    g = np.asarray(g)
    k0 = int(np.argmax(g < 0.5))   # the gripper starts closed: the grasp is the first close after it opens
    k = k0   # the still: the gripper opens on the approach (the cube is still visible in both)
    i = min(int(round(tt[min(k, len(tt) - 1)] * fps)), len(frames) - 1)
    cv2.imwrite(os.path.join(out, "phone_frame.jpg"), frames[i][p0:p1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    imageio.imwrite(os.path.join(out, "sim_frame.jpg"), r["frames"][k], quality=90)


def rollout(out):
    src = sorted(glob.glob("outputs/colab/P5_naive_all/hero/success_1090000.mp4"))[0]
    frames = [label(f, "policy trained on 100 phone demos, unseen layout (sim)") for f in imageio.mimread(src, memtest=False)]
    save(frames, os.path.join(out, "rollout"), gif_width=480)


def nearest_sim(phone_npz, sim_dir="data/sim_v3s2", n=200):
    c = np.load(phone_npz)["layout"][2]
    fs = sorted(glob.glob(os.path.join(sim_dir, "blue-red_*.npz")))[:n]
    return min(fs, key=lambda f: np.linalg.norm(np.load(f)["layout"][2] - c))


def phone_replay_vs_clean(clip, tag, out):
    p = f"data/human/5/{clip}.npz"
    q = nearest_sim(p)
    a, oka, _ = replay_npz(p)
    b, okb, _ = replay_npz(q)
    left = [label(f, "sim demo from a phone clip", f"{clip}, {len(a) - 1} steps") for f in a]
    right = [label(f, "clean sim-expert demo", f"{os.path.basename(q)[:-4]}, nearest start, {len(b) - 1} steps")
             for f in b]
    save(pair(left, right), os.path.join(out, f"phone_replay_vs_clean_{tag}"))


def quality_rows():
    return {os.path.splitext(os.path.basename(r["file"]))[0]: r for r in csv.DictReader(open("quality/phone_s5_first15.csv"))}


def two_clips(c1, t1, c2, t2, name, out):
    a, _, _ = replay_npz(f"data/human/5/{c1}.npz")
    b, _, _ = replay_npz(f"data/human/5/{c2}.npz")
    save(pair([label(f, *t1) for f in a], [label(f, *t2) for f in b]), os.path.join(out, name))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/media")
    p.add_argument("--only", default="", help="comma list of: pair,rollout,clean,quality,ambient,vn")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else {"pair", "rollout", "clean", "quality", "ambient", "vn"}
    q = quality_rows()
    if "pair" in only:
        phone_vs_sim("blue_to_red_001.MOV", a.out)
    if "rollout" in only:
        rollout(a.out)
    if "clean" in only:
        phone_replay_vs_clean("blue_to_red_001", "a", a.out)
        phone_replay_vs_clean("blue_to_red_015", "b", a.out)
    if "quality" in only:
        hi, lo = max(q.values(), key=lambda r: float(r["quality"])), min(q.values(), key=lambda r: float(r["quality"]))
        name = lambda r: os.path.splitext(os.path.basename(r["file"]))[0]
        desc = lambda r: (f"{name(r)}: quality tag {float(r['quality']):+.2f}",
                          f"grasp error {float(r['grasp_cm']):.1f} cm, replay {'ok' if r['success'] == 'True' else 'FAILED'}")
        two_clips(name(hi), desc(hi), name(lo), desc(lo), "quality_tags", a.out)
    if "ambient" in only:
        clean = [r for r in q.values() if r["clean"] == "True"]
        noisy = [r for r in q.values() if r["clean"] == "False" and r["success"] == "True"]
        c = max(clean, key=lambda r: float(r["score"]))
        n = min(noisy, key=lambda r: float(r["score"]))
        name = lambda r: os.path.splitext(os.path.basename(r["file"]))[0]
        two_clips(name(c), (f"clean: {name(c)}", f"score {float(c['score']):.2f}, all noise levels (t_min 0)"),
                  name(n), (f"noisy: {name(n)}", f"score {float(n['score']):.2f}, only t >= {float(n['t_min']):.2f}"),
                  "ambient_clean_vs_noisy", a.out)
    if "vn" in only:
        c = "blue_to_red_001"
        f = json.load(open("quality/vn_s5.json"))["f"][c]
        x, _, _ = replay_npz(f"data/human/5/{c}.npz")
        y, _, _ = replay_npz(f"data/human_vn/5/{c}.npz")
        save(pair([label(fr, f"{c}: as recorded", f"{len(x) - 1} steps") for fr in x],
                  [label(fr, f"{c}: velocity-normalized", f"speed x{f:.2f}, {len(y) - 1} steps") for fr in y]),
             os.path.join(a.out, "vn_before_after"))


if __name__ == "__main__":
    main()
