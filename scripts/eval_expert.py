"""Milestone 1 acceptance: expert success over N random episodes per task, example videos, and rollout speed.

    python scripts/eval_expert.py --n 100 --out outputs/m1
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from sim.env import ALL_TASKS, PickPlaceEnv, task_name  # noqa: E402
from sim.expert import run_expert_episode  # noqa: E402

SEED_BASE = 500_000
_env = None


def _work(args):
    global _env
    task, seeds = args
    _env = _env or PickPlaceEnv()
    return task, [run_expert_episode(_env, task, s)["success"] for s in seeds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", default="outputs/m1")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--videos", default="red-yellow,green-orange,blue-purple")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # speed: headless 200-step rollout (expert actions, then holding), single process
    env = PickPlaceEnv()
    env.reset((0, 0), seed=0)
    t0 = time.time()
    for _ in range(200):
        env.step([0.25, 0.2, 0.1, 0.0])
    sps = 200 / (time.time() - t0)
    print(f"headless speed: {sps:.0f} control steps/s (200-step rollout in {200 / sps:.2f}s)")

    jobs = []
    for ti, task in enumerate(ALL_TASKS):
        seeds = [SEED_BASE + ti * 10_000 + k for k in range(a.n)]
        jobs += [(task, seeds[i::4]) for i in range(4)]
    res = {t: [] for t in ALL_TASKS}
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        for task, succ in pool.imap_unordered(_work, jobs):
            res[task] += succ
    print(f"{len(ALL_TASKS) * a.n} episodes in {time.time() - t0:.0f}s with {a.workers} workers")
    with open(os.path.join(a.out, "expert_success.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n", "success_rate"])
        print(f"{'task':14s} {'n':>4s} success")
        for t in ALL_TASKS:
            r = float(np.mean(res[t]))
            w.writerow([task_name(t), len(res[t]), f"{r:.3f}"])
            print(f"{task_name(t):14s} {len(res[t]):4d} {r:.2f}")
        overall = float(np.mean(sum(res.values(), [])))
        w.writerow(["all", sum(len(v) for v in res.values()), f"{overall:.3f}"])
        print(f"{'all':14s} {'':4s} {overall:.3f}")

    for name in a.videos.split(","):
        task = tuple(ALL_TASKS[[task_name(t) for t in ALL_TASKS].index(name)])
        r = run_expert_episode(env, task, 7, render="front")
        top = run_expert_episode(env, task, 7, render="top")["frames"]
        frames = [np.concatenate([f, g], 1) for f, g in zip(r["frames"], top)]
        imageio.mimsave(os.path.join(a.out, f"expert_{name}.mp4"), frames, fps=10, macro_block_size=1)
        print(f"video expert_{name}.mp4 success={r['success']}")


if __name__ == "__main__":
    main()
