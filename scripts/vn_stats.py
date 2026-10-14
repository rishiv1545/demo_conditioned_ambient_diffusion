"""Velocity normalization stats for one task's phone episodes, and the before/after speed comparison.

    python scripts/vn_stats.py stats --src data/human/5 --out quality/vn_s5.json          # from the non-VN processing
    python scripts/process_phone.py data/raw_phone/5 --vn quality/vn_s5.json --out_data data/human_vn ...
    python scripts/vn_stats.py compare --before data/human/5 --after data/human_vn/5

Speeds come from each episode's retargeted path (raw_traj; the inserted dwells are idle frames and drop out of the
moving-frame percentiles). See human/velocity_norm.py.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from human.velocity_norm import cruise_speed, frame_speed, task_stats  # noqa: E402


def load(src):
    fs = sorted(glob.glob(os.path.join(src, "*.npz")))
    return {os.path.splitext(os.path.basename(f))[0]: np.load(f, allow_pickle=True) for f in fs}


def summary(eps, idle_v=0.02):
    cruise = np.array([cruise_speed(d["raw_traj"], idle_v=idle_v) for d in eps.values()])
    med = np.array([np.median(v[v >= idle_v]) for v in (frame_speed(d["raw_traj"]) for d in eps.values())])
    p95 = np.array([np.percentile(v, 95) for v in (frame_speed(d["raw_traj"]) for d in eps.values())])
    dur = np.array([len(d["action"]) / 10.0 for d in eps.values()])
    ok = np.array([bool(d["success"]) for d in eps.values()])
    s = lambda x: {"mean": float(x.mean()), "std": float(x.std()), "cv": float(x.std() / x.mean()), "var": float(x.var())}
    return {"n": len(eps), "replay_success": float(ok.mean()), "cruise_mps": s(cruise), "median_moving_mps": s(med),
            "p95_mps": s(p95), "duration_s": s(dur)}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--src", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--out", default=None)
    a = p.parse_args()
    if a.cmd == "stats":
        eps = load(a.src)
        st = task_stats([d["raw_traj"] for d in eps.values()])
        st["f"] = dict(zip(eps, st["f"]))
        st["cruise"] = dict(zip(eps, st["cruise"]))
        json.dump(st, open(a.out, "w"), indent=1)
        f = np.array(list(st["f"].values()))
        print(f"{len(eps)} episodes: m {st['m']:.3f} m/s, M {st['M']:.3f} m/s; f {f.min():.2f}-{f.max():.2f} "
              f"(clamped: {int((f <= 0.75).sum())} at 0.75, {int((f >= 1.5).sum())} at 1.5)")
        return
    before, after = load(a.before), load(a.after)
    common = sorted(set(before) & set(after))
    res = {"before": summary({k: before[k] for k in common}), "after": summary({k: after[k] for k in common})}
    for k in ("cruise_mps", "median_moving_mps", "p95_mps", "duration_s"):
        b, f = res["before"][k], res["after"][k]
        print(f"{k:18s} mean {b['mean']:.3f} -> {f['mean']:.3f}   std {b['std']:.3f} -> {f['std']:.3f}   "
              f"CV {b['cv']:.2f} -> {f['cv']:.2f}")
    print(f"replay success     {res['before']['replay_success']:.0%} -> {res['after']['replay_success']:.0%} "
          f"({len(common)} episodes)")
    flips = [(k, bool(before[k]["success"]), bool(after[k]["success"])) for k in common
             if bool(before[k]["success"]) != bool(after[k]["success"])]
    print("changed:", ", ".join(f"{k} {'ok' if x else 'FAIL'}->{'ok' if y else 'FAIL'}" for k, x, y in flips) or "none")
    res["changed"] = flips
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
