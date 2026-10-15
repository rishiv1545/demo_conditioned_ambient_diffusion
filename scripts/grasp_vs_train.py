"""Do the policy's grasps copy the grasp of the nearest training demo? (memorization check for the phone runs)

    python scripts/grasp_vs_train.py --eval outputs/colab/P5_naive_15/eval_step010000_k20_br_zn_go/episodes.csv \
        --train data/human/5 --n 15

Grasp offset = EE - target cube position at the step the gripper is first commanded closed after being open, the
same definition as vla/eval_smolvla.py (training demos: from the stored replay obs/actions). For every eval
episode: the training clip whose cube start is nearest (the first --n clips in episode order, as per_task=N), its
offset, and the distance of the eval offset to that clip's offset vs to the mean training offset. Memorized noise
shows as eval offsets close to the nearest clip's (unusually large) offset rather than to the mean.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def train_grasps(src, n):
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "*.npz")))[:n]:
        d = np.load(f, allow_pickle=True)
        c = int(d["task"][0])
        obs, act = d["obs"], d["action"]
        opened = np.flatnonzero(act[:, 3] < 0.5)
        close = [k for k in range(len(act)) if act[k, 3] >= 0.5 and len(opened) and k > opened[0]]
        off = obs[close[0], :3] - obs[close[0], 4 + 3 * c: 7 + 3 * c] if close else np.full(3, np.nan)
        rows.append({"clip": os.path.splitext(os.path.basename(f))[0], "start": d["layout"][c], "off": off.astype(float),
                     "replay_ok": bool(d["success"])})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True, help="episodes.csv with cube_x/cube_y/grasp_d* columns")
    p.add_argument("--train", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    tr = train_grasps(a.train, a.n)
    starts = np.array([r["start"] for r in tr])
    offs = np.array([r["off"] for r in tr])
    mean_off = np.nanmean(offs, axis=0)
    print(f"training grasps ({len(tr)} clips): xy offset median {np.nanmedian(np.linalg.norm(offs[:, :2], axis=1)) * 100:.1f} cm,"
          f" mean offset ({mean_off[0] * 100:+.1f}, {mean_off[1] * 100:+.1f}, {mean_off[2] * 100:+.1f}) cm")
    res = []
    for e in csv.DictReader(open(a.eval)):
        if not e.get("grasp_dx"):
            off = None
        else:
            off = np.array([float(e[k]) for k in ("grasp_dx", "grasp_dy", "grasp_dz")])
        st = np.array([float(e["cube_x"]), float(e["cube_y"])])
        j = int(np.argmin(np.linalg.norm(starts - st, axis=1)))
        lifted = e["target_lifted"] == "1"
        r = {"seed": int(e["seed"]), "success": e["success"] == "1",
             "outcome": "success" if e["success"] == "1" else ("missed_grasp" if not lifted else "other_fail"),
             "start_cm": (st * 100).round(1).tolist(), "nn_clip": tr[j]["clip"],
             "nn_start_dist_cm": float(np.linalg.norm(starts[j] - st) * 100),
             "nn_off_cm": (tr[j]["off"] * 100).round(1).tolist(), "nn_replay_ok": tr[j]["replay_ok"]}
        if off is not None:
            r.update(off_cm=(off * 100).round(1).tolist(),
                     d_nn_cm=float(np.linalg.norm((off - tr[j]["off"])[:2]) * 100),
                     d_mean_cm=float(np.linalg.norm((off - mean_off)[:2]) * 100))
        res.append(r)
    print(f"{'seed':>8} {'outcome':12} {'eval offset cm':>20} {'nearest clip':>16} {'start d':>7} "
          f"{'clip offset cm':>20} {'d_nn':>5} {'d_mean':>6}")
    for r in sorted(res, key=lambda r: r["outcome"]):
        o = r.get("off_cm")
        print(f"{r['seed']:>8} {r['outcome']:12} {str(o) if o else '(never closed)':>20} {r['nn_clip']:>16} "
              f"{r['nn_start_dist_cm']:6.1f} {str(r['nn_off_cm']):>20} "
              f"{r.get('d_nn_cm', float('nan')):5.1f} {r.get('d_mean_cm', float('nan')):6.1f}")
    for k in ("success", "missed_grasp", "other_fail"):
        g = [r for r in res if r["outcome"] == k and "d_nn_cm" in r]
        if g:
            print(f"{k:12s} n={len(g):2d}  median xy dist to nearest clip's offset {np.median([r['d_nn_cm'] for r in g]):.1f} cm,"
                  f" to mean offset {np.median([r['d_mean_cm'] for r in g]):.1f} cm; nearest clip's own |xy offset| "
                  f"{np.median([np.linalg.norm(r['nn_off_cm'][:2]) for r in g]):.1f} cm")
    if a.out:
        json.dump({"train": [{**r, "start": r["start"].tolist(), "off": r["off"].tolist()} for r in tr],
                   "eval": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
