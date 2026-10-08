"""Render example videos: training demos (replayed from their stored layout + actions) and policy rollouts.

    python scripts/render_videos.py --ckpt checkpoints/A --out outputs/m3/videos
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.data import layout_from_array, load_episodes  # noqa: E402
from policy.evaluate import EVAL_SEED_BASE, PolicyRunner, run_episode  # noqa: E402
from sim.env import ALL_TASKS, PickPlaceEnv, resolve_heldout, task_name  # noqa: E402


def caption(frame, text, ok=None):
    f = frame.copy()
    cv2.rectangle(f, (0, 0), (f.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(f, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if ok is not None:
        lab, col = ("SUCCESS", (60, 200, 60)) if ok else ("FAIL", (230, 60, 60))
        cv2.putText(f, lab, (f.shape[1] - 130, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return f


def both_views(env):
    return np.concatenate([env.render("front"), env.render("top")], 1)


def save(path, frames):
    imageio.mimsave(path, frames, fps=10, macro_block_size=1)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/A")
    p.add_argument("--data", default="data/sim")
    p.add_argument("--out", default="outputs/m3/videos")
    p.add_argument("--n", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.ckpt, "ckpt.pt") if os.path.isdir(a.ckpt) else a.ckpt
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    heldout = resolve_heldout(ck["args"]["heldout"])
    seen = [t for t in ALL_TASKS if t not in heldout]
    env = PickPlaceEnv()

    # training demos: the first episode of n different tasks the checkpoint was trained on
    per_task = ck["args"].get("sim_per_task", 20)
    for t in seen[::max(1, len(seen) // a.n)][:a.n]:
        ep = load_episodes(a.data, {"sources": ["sim"], "tasks": [t], "max_per_task": {"sim": 1}})[0]
        env.reset(task=t, layout=layout_from_array(ep["layout"]))
        frames = [both_views(env)]
        ok = False
        for act in ep["action"]:
            _, ok = env.step(act)
            frames.append(both_views(env))
        name = os.path.splitext(os.path.basename(ep["path"]))[0]
        frames = [caption(f, f"training demo (sim expert): {task_name(t)}  [{name}, 1 of {per_task}/task]", ok)
                  for f in frames]
        save(os.path.join(a.out, f"train_{name}.mp4"), frames)

    # held-out rollouts: one per held-out task, eval seeds
    runner = PolicyRunner(ckpt, "cpu")
    for t in heldout[:a.n]:
        seed = EVAL_SEED_BASE + ALL_TASKS.index(t) * 10_000
        runner.gen.manual_seed(seed)
        ok, steps, _ = run_episode(env, runner, t, seed)
        runner.gen.manual_seed(seed)   # re-run identically, rendering both views
        obs = env.reset(task=t, seed=seed)
        runner.reset(t)
        frames, streak = [both_views(env)], 0
        for _ in range(steps):
            obs, s = env.step(runner.act(obs))
            frames.append(both_views(env))
        frames = [caption(f, f"policy {os.path.basename(os.path.dirname(ckpt))} rollout, HELD-OUT task "
                             f"{task_name(t)} (seed {seed})", ok) for f in frames]
        save(os.path.join(a.out, f"rollout_heldout_{task_name(t)}_{seed}.mp4"), frames)


if __name__ == "__main__":
    main()
