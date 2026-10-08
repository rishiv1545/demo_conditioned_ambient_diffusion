"""Synthetic "phone-like" demos for the ambient-loss validation (mechanism study, small diffusion policy).

Expert action trajectories are corrupted with the error types measured in the real phone pipeline (NOTES.md,
Milestone 2): a per-episode xy offset (grasp misplacement, ~2-4 cm in the real replays), a per-episode z offset,
per-step jitter, and shifted gripper switch times. Then they are replayed open-loop in sim from the episode's
layout, exactly like phone replays, so some grasps genuinely fail. Saved as source "synthetic" with
sigma_n = the xy-offset scale, and success = the replay's success.

    python scripts/make_synthetic_noisy.py --src data/sim_v2 --out data/synthetic_v2 --tasks heldout --per_task 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from policy.data import layout_from_array, load_episodes, save_episode  # noqa: E402
from sim.env import (ALL_TASKS, DEFAULT_SPLIT, PickPlaceEnv, env_cfg_dict, env_config, resolve_heldout,  # noqa: E402
                     task_name)


def corrupt(actions, rng, xy_sigma, z_sigma, jitter, grip_shift_steps):
    a = actions.copy()
    a[:, :2] += rng.normal(0, xy_sigma, 2)                 # per-episode misplacement
    a[:, 2] += rng.normal(0, z_sigma)
    a[:, :3] += rng.normal(0, jitter, (len(a), 3))
    g = a[:, 3] > 0.5
    sw = np.flatnonzero(np.diff(g.astype(int)) != 0) + 1
    for k in sw:                                           # shift each gripper switch by a random number of steps
        s = int(rng.integers(-grip_shift_steps, grip_shift_steps + 1))
        lo, hi = sorted((k, k + s))
        a[lo:hi, 3] = a[k - 1, 3] if s > 0 else a[k, 3]
    a[:, 2] = np.maximum(a[:, 2], 0.008)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/sim_v2")
    p.add_argument("--out", default="data/synthetic_v2")
    p.add_argument("--tasks", default="heldout")
    p.add_argument("--heldout", default=DEFAULT_SPLIT)
    p.add_argument("--per_task", type=int, default=8)
    p.add_argument("--skip", type=int, default=150, help="use source episodes from this index on (not used by sim runs)")
    p.add_argument("--xy_sigma", type=float, default=0.025)
    p.add_argument("--z_sigma", type=float, default=0.01)
    p.add_argument("--jitter", type=float, default=0.003)
    p.add_argument("--grip_shift", type=int, default=3, help="max gripper switch shift in steps (0.1 s each)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    heldout = resolve_heldout(a.heldout)
    tasks = {"seen": [t for t in ALL_TASKS if t not in heldout], "heldout": heldout, "all": ALL_TASKS}[a.tasks]
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    for t in tasks:
        src = load_episodes(a.src, {"sources": ["sim"], "tasks": [t]})[a.skip:a.skip + a.per_task]
        n_ok = 0
        for k, ep in enumerate(src):
            env = PickPlaceEnv(env_config(ep["meta"].get("env_cfg")))
            act = corrupt(ep["action"], rng, a.xy_sigma, a.z_sigma, a.jitter, a.grip_shift)
            obs = env.reset(task=t, layout=layout_from_array(ep["layout"]))
            O, ok = [], False
            for x in act:
                O.append(obs)
                obs, ok = env.step(x)
            n_ok += ok
            save_episode(os.path.join(a.out, f"{task_name(t)}_{k:03d}.npz"), np.stack(O), act, t, "synthetic",
                         layout_from_array(ep["layout"]), ok, sigma_n=a.xy_sigma,
                         meta={"from": ep["path"], "env_cfg": env_cfg_dict(env.cfg), "xy_sigma": a.xy_sigma})
        print(f"{task_name(t):14s} {len(src)} corrupted demos, replay success {n_ok}/{len(src)}")


if __name__ == "__main__":
    main()
