"""Quality score per phone replay (one task), for clean/noisy splits, per-episode ambient t_min and a quality tag.

    python scripts/score_phone.py --datasets phone_s5 --task blue-red --sim_dir data/sim_v3s2 --first 15 \
        --out quality/phone_s5_first15.json

Per episode (from the sim replay stored in each episode's .npz: obs = ee, gripper width, 3 cubes, 3 zones at 10 Hz;
action = ee target + gripper command, 1 = closed):
  success      the sim replay put the target cube in the target zone
  grasp_cm     xy offset of the gripper from the target cube at the grasp close (first open -> closed command)
  dwell_s      time from arriving at the grasp point (within 1 cm of the close position) to the close command;
               the error is |dwell - the sim expert's median|. For phone episodes dwell, jerk and speed come from the
               retargeted hand track before retarget.py inserts its fixed dwell at gripper switches (raw_traj,
               raw_gripper): the commanded path always shows that 0.5 s dwell
  lift_s       time from the close to the target cube rising 1 cm (never: the episode length)
  jerk         mean |third difference| of the commanded ee path (m / step^3 x 1e4)
  speed        95th percentile of the commanded ee speed (m/s)
  moved_cm     largest displacement of a non-target cube
Each metric is divided by a reference (the 90th percentile over clean sim expert demos of the same task, or a
fixed tolerance for grasp_cm 1.0 and moved_cm 1.0) and turned into a sub-score exp(-max(0, x / ref - 1)), so
expert-like values score 1. score = 0.4 * success + 0.6 * mean(sub-scores), in [0, 1].

Split (within the scored episodes: --first N restricts them to a training subset): the top --clean_frac by score (successful replays only) are clean (t_min 0: full supervision); the others
are noisy with t_min from t_lo (best noisy) to t_hi (worst) by score rank. quality = 2 * score - 1 in [-1, 1]
(appended to the normalized state with train_smolvla.py --quality_tag 1; inference requests +1).
The output maps dataset -> episode_index -> {score, t_min, quality, clean, metrics}; train_smolvla.py
--episode_info reads it (episodes not listed keep sigma_n semantics: clean sim t_min 0, quality +1).
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human.correction import first_grasp_close  # noqa: E402
from sim.env import CUBES, parse_task  # noqa: E402

FPS = 10
FIXED_REF = {"grasp_cm": 1.0, "moved_cm": 1.0}
METRICS = ["grasp_cm", "dwell_err_s", "lift_s", "jerk", "speed", "moved_cm"]


def dwell(path, closed):
    """Seconds from the start of the last run within 1 cm of the grasp-close position to the close (None: no close)."""
    k = first_grasp_close(closed)
    if k is None:
        return None
    near = np.linalg.norm(path[:k + 1] - path[k], axis=1) < 0.01
    return float((np.flatnonzero(~near[::-1])[0] if (~near).any() else k) / FPS)


def episode_metrics(obs, action, task, dwell_ref=None, raw_traj=None, raw_gripper=None):
    c, _ = task
    ee, cubes = obs[:, :3], obs[:, 4:13].reshape(-1, 3, 3)
    closed = action[:, 3] > 0.5
    k = first_grasp_close(closed)
    n = len(obs)
    path, grip = (raw_traj, raw_gripper > 0.5) if raw_traj is not None else (action[:, :3], closed)
    m = {}
    if k is None:
        m.update(grasp_cm=10.0, lift_s=n / FPS)
    else:
        m["grasp_cm"] = float(np.linalg.norm(ee[k, :2] - cubes[k, c, :2]) * 100)
        rise = np.flatnonzero(cubes[k:, c, 2] > cubes[k, c, 2] + 0.01)
        m["lift_s"] = float(rise[0] / FPS) if len(rise) else float((n - k) / FPS)
    dw = dwell(np.asarray(path, np.float64), grip)
    m["dwell_s"] = dw if dw is not None else 0.0
    m["dwell_err_s"] = abs(m["dwell_s"] - dwell_ref) if dwell_ref is not None else 0.0
    p = np.asarray(path, np.float64)
    m["jerk"] = float(np.abs(np.diff(p, 3, axis=0)).sum(1).mean() * 1e4) if n > 4 else 0.0
    m["speed"] = float(np.percentile(np.linalg.norm(np.diff(p, axis=0), axis=1) * FPS, 95)) if n > 1 else 0.0
    others = [i for i in range(len(CUBES)) if i != c]
    m["moved_cm"] = float(max(np.linalg.norm(cubes[:, i, :2] - cubes[0, i, :2], axis=1).max() for i in others) * 100)
    return m


def expert_refs(sim_dir, task, n=50):
    from sim.env import task_name
    files = sorted(glob.glob(os.path.join(sim_dir, f"{task_name(task)}_*.npz")))[:n]
    assert files, f"no expert demos for {task_name(task)} in {sim_dir}"
    ms = [episode_metrics(d["obs"], d["action"], task) for d in (np.load(f, allow_pickle=True) for f in files)]
    dwell_ref = float(np.median([m["dwell_s"] for m in ms]))
    ms = [episode_metrics(d["obs"], d["action"], task, dwell_ref) for d in (np.load(f, allow_pickle=True) for f in files)]
    ref = {k: max(float(np.percentile([m[k] for m in ms], 90)), 1e-3) for k in METRICS}
    ref["dwell_err_s"] = max(ref["dwell_err_s"], 0.2)   # expert dwell is nearly constant: allow 0.2 s
    ref.update(FIXED_REF)
    return ref, dwell_ref


def score(m, success, ref):
    sub = {k: float(np.exp(-max(0.0, m[k] / ref[k] - 1.0))) for k in METRICS}
    return 0.4 * float(success) + 0.6 * float(np.mean(list(sub.values()))), sub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True, help="exported phone datasets under --root")
    p.add_argument("--root", default="data/lerobot")
    p.add_argument("--task", default="blue-orange")
    p.add_argument("--sim_dir", default="data/sim_v2_bo", help="clean expert demos for the references")
    p.add_argument("--clean_frac", type=float, default=0.3)
    p.add_argument("--first", type=int, default=None,
                   help="score only each dataset's first N episodes (the per_task=N training subset), so the clean "
                        "split is the top clean_frac of the clips actually used, not of the whole pool")
    p.add_argument("--t_lo", type=float, default=0.5)
    p.add_argument("--t_hi", type=float, default=0.9)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    task = parse_task(a.task)
    ref, dwell_ref = expert_refs(a.sim_dir, task)
    rows = []
    for name in a.datasets:
        side = json.load(open(os.path.join(a.root, name, "episodes.json")))["episodes"]
        for e in side:
            if e["task"] != a.task or (a.first is not None and e["episode_index"] >= a.first):
                continue
            d = np.load(e["file"], allow_pickle=True)
            raw = (d["raw_traj"], d["raw_gripper"]) if "raw_traj" in d.files else (None, None)
            m = episode_metrics(d["obs"], d["action"], task, dwell_ref, *raw)
            s, sub = score(m, e["success"], ref)
            rows.append({"dataset": name, "episode_index": e["episode_index"], "file": e["file"],
                         "success": bool(e["success"]), "score": s, **m, **{f"sub_{k}": v for k, v in sub.items()}})
    assert rows, f"no {a.task} episodes in {a.datasets}"
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["score"])
    n_clean = max(1, int(np.ceil(a.clean_frac * len(rows) - 1e-9)))   # 15 clips -> 5, 30 -> 9, 100 -> 30
    clean = [i for i in order if rows[i]["success"]][:n_clean]
    noisy = [i for i in order if i not in clean]
    for i in clean:
        rows[i].update(clean=True, t_min=0.0)
    for r, i in enumerate(noisy):
        frac = r / max(1, len(noisy) - 1)
        rows[i].update(clean=False, t_min=round(a.t_lo + (a.t_hi - a.t_lo) * frac, 3))
    for r in rows:
        r["quality"] = round(2 * r["score"] - 1, 4)
    out = {"task": a.task, "ref": ref, "dwell_ref_s": dwell_ref, "clean_frac": a.clean_frac, "first": a.first,
           "t_range": [a.t_lo, a.t_hi], "datasets": {}}
    for r in rows:
        out["datasets"].setdefault(r["dataset"], {})[str(r["episode_index"])] = {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("dataset", "episode_index")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    with open(os.path.splitext(a.out)[0] + ".csv", "w", newline="") as f:
        cols = ["dataset", "episode_index", "file", "success", "score", "clean", "t_min", "quality"] + METRICS
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for i in order:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[i].items()})
    print(f"{len(rows)} {a.task} episodes: {len(clean)} clean (scores {rows[clean[-1]]['score']:.2f}-"
          f"{rows[clean[0]]['score']:.2f}), {len(noisy)} noisy; refs {json.dumps({k: round(v, 3) for k, v in ref.items()})}")
    for i in order:
        r = rows[i]
        print(f"  {r['dataset'] + '/' + os.path.basename(r['file']):40s} {'ok ' if r['success'] else 'FAIL'} score {r['score']:.2f} "
              f"{'clean' if r['clean'] else 't_min %.2f' % r['t_min']}  grasp {r['grasp_cm']:.1f} cm, dwell "
              f"{r['dwell_s']:.1f} s, lift {r['lift_s']:.1f} s, jerk {r['jerk']:.2f}, speed {r['speed']:.2f}, "
              f"moved {r['moved_cm']:.1f} cm")


if __name__ == "__main__":
    main()
