"""Generate clean scripted-expert demos: N successful episodes per task for the chosen task split.

    python scripts/gen_sim_data.py --out data/sim --n_per_task 20 --tasks seen
"""
import argparse
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.data import save_episode  # noqa: E402
from sim.env import ALL_TASKS, DEFAULT_HELDOUT, PickPlaceEnv, parse_task, task_name  # noqa: E402
from sim.expert import run_expert_episode  # noqa: E402

DATA_SEED_BASE = 0  # data seeds: task_index * 100_000 + k; eval seeds live at >= 1_000_000

_env = None


def _work(args):
    global _env
    task, n, out = args
    _env = _env or PickPlaceEnv()
    ti = ALL_TASKS.index(task)
    kept, tried = 0, 0
    while kept < n:
        seed = DATA_SEED_BASE + ti * 100_000 + tried
        tried += 1
        r = run_expert_episode(_env, task, seed)
        if not r["success"]:
            continue
        save_episode(os.path.join(out, f"{task_name(task)}_{kept:03d}.npz"), r["obs"], r["action"], task, "sim",
                     _env.layout, True, meta={"seed": seed})
        kept += 1
    return task, kept, tried


def select_tasks(which, heldout):
    if which == "all":
        return list(ALL_TASKS)
    if which == "seen":
        return [t for t in ALL_TASKS if t not in heldout]
    if which == "heldout":
        return list(heldout)
    return [parse_task(s) for s in which.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sim")
    p.add_argument("--n_per_task", type=int, default=20)
    p.add_argument("--tasks", default="seen", help="seen | heldout | all | comma list like red-yellow,blue-orange")
    p.add_argument("--heldout", default=",".join(task_name(t) for t in DEFAULT_HELDOUT))
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = p.parse_args()
    heldout = [parse_task(s) for s in a.heldout.split(",")]
    tasks = select_tasks(a.tasks, heldout)
    os.makedirs(a.out, exist_ok=True)
    with get_context("spawn").Pool(min(a.workers, len(tasks))) as pool:
        for task, kept, tried in pool.imap_unordered(_work, [(t, a.n_per_task, a.out) for t in tasks]):
            print(f"{task_name(task):14s} kept {kept}/{tried} expert episodes")


if __name__ == "__main__":
    main()
