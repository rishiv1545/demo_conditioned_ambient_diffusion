"""Evaluate a SmolVLA checkpoint on all 9 tasks (seen vs held-out), same protocol as policy/evaluate.py:
fixed eval seeds, success = env.success() held for 5 consecutive steps, else failure at 200 steps.

Episodes run in batches of environments stepping in lockstep, with one batched policy call per re-plan.

    python vla/eval_smolvla.py --run checkpoints/vla/V0 --step latest --k 20 --out outputs/vla/V0
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import build_policy, list_checkpoints, load_run_config, load_trainable, pick_device  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.evaluate import EVAL_SEED_BASE, wilson  # noqa: E402
from sim.env import (ALL_TASKS, DEFAULT_SPLIT, IMAGE_CAMERAS, PickPlaceEnv, instruction,  # noqa: E402
                     resolve_heldout, task_name)


def to_batch(envs, tasks, dev):
    imgs = {c: [] for c in IMAGE_CAMERAS}
    for e in envs:
        for c, im in e.images().items():
            imgs[c].append(torch.from_numpy(im).permute(2, 0, 1).float() / 255.0)
    b = {f"observation.images.{c}": torch.stack(v) for c, v in imgs.items()}
    b["observation.state"] = torch.from_numpy(np.stack([e.state() for e in envs]))
    b["task"] = [instruction(t) for t in tasks]
    return b


def run_batch(policy, pre, post, jobs, dev, hold=5, record=None):
    """jobs: list of (task, seed). Returns list of (success, steps) and, for indices in `record`, frame lists."""
    envs = [PickPlaceEnv() for _ in jobs]
    for e, (t, s) in zip(envs, jobs):
        e.reset(task=t, seed=s)
    policy.reset()
    tasks = [t for t, _ in jobs]
    streak = np.zeros(len(jobs), int)
    done = np.zeros(len(jobs), bool)
    steps = np.zeros(len(jobs), int)
    frames = {i: [] for i in (record or [])}
    max_steps = envs[0].cfg.max_steps
    for _ in range(max_steps):
        for i in frames:
            frames[i].append(envs[i].render("front"))
        with torch.no_grad():
            act = post(policy.select_action(pre(to_batch(envs, tasks, dev)))).cpu().numpy()
        for i, e in enumerate(envs):
            if done[i]:
                continue                     # finished envs are frozen (their actions are ignored)
            _, s = e.step(act[i])
            steps[i] += 1
            streak[i] = streak[i] + 1 if s else 0
            done[i] = streak[i] >= hold
        if done.all():
            break
    for e in envs:
        e.close()
    return [(bool(d), int(n)) for d, n in zip(done, steps)], frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run dir with run.json and step_* checkpoints")
    p.add_argument("--step", default="latest", help="checkpoint step, 'latest', or 'base' (no fine-tuning)")
    p.add_argument("--k", type=int, default=20, help="episodes per task (20 intermediate, 50 final)")
    p.add_argument("--tasks", default="all", help="all | seen | heldout")
    p.add_argument("--heldout", default=None, help="default: the split stored with the training data")
    p.add_argument("--n_action_steps", type=int, default=10, help="actions executed per 50-step chunk")
    p.add_argument("--batch_envs", type=int, default=18)
    p.add_argument("--videos", type=int, default=2, help="episodes per task-split to save as video")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    dev = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    rc = load_run_config(a.run)
    spec = a.heldout
    if spec is None:  # the split the training data was exported with
        side = os.path.join(rc["args"]["data"][0], "episodes.json")
        spec = ",".join(json.load(open(side))["heldout"]) if os.path.exists(side) else DEFAULT_SPLIT
    heldout = resolve_heldout(spec)
    policy, pre, post, _ = build_policy(rc["features"], rc["stats"], dev, n_action_steps=a.n_action_steps)
    step = "base"
    if a.step != "base":
        cks = dict(list_checkpoints(a.run))
        step = max(cks) if a.step == "latest" else int(a.step)
        load_trainable(policy, cks[step])
    policy.eval()
    tasks = {"all": ALL_TASKS, "seen": [t for t in ALL_TASKS if t not in heldout], "heldout": heldout}[a.tasks]
    jobs = [(t, EVAL_SEED_BASE + ALL_TASKS.index(t) * 10_000 + k) for t in tasks for k in range(a.k)]
    print(f"{a.run} step {step} on {dev}: {len(jobs)} episodes, held-out = {[task_name(t) for t in heldout]}")
    rows, t0 = [], time.time()
    vid_left = {"seen": a.videos, "heldout": a.videos}
    import imageio.v2 as imageio
    for i in range(0, len(jobs), a.batch_envs):
        chunk = jobs[i:i + a.batch_envs]
        rec = []
        for j, (t, _) in enumerate(chunk):
            sp = "heldout" if t in heldout else "seen"
            if vid_left[sp] > 0 and j % 6 == 0:
                rec.append(j)
                vid_left[sp] -= 1
        res, frames = run_batch(policy, pre, post, chunk, dev, record=rec)
        for (t, s), (ok, n) in zip(chunk, res):
            rows.append((t, s, ok, n))
        for j, fr in frames.items():
            t, s = chunk[j]
            ok = res[j][0]
            imageio.mimsave(os.path.join(a.out, f"{'success' if ok else 'fail'}_{task_name(t)}_{s}.mp4"),
                            fr[:res[j][1] + 1], fps=10, macro_block_size=1)
        print(f"{len(rows)}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seed", "success", "steps"])
        for t, s, ok, n in rows:
            w.writerow([task_name(t), s, int(ok), n])
    per = {t: [r[2] for r in rows if r[0] == t] for t in tasks}
    summary = {"run": a.run, "step": step, "k": a.k, "n_action_steps": a.n_action_steps,
               "heldout_tasks": [task_name(t) for t in heldout], "eval_seconds": time.time() - t0,
               "per_task": {task_name(t): float(np.mean(v)) for t, v in per.items()}}
    for split, ts in (("seen", [t for t in tasks if t not in heldout]), ("heldout", [t for t in tasks if t in heldout])):
        v = sum((per[t] for t in ts), [])
        if v:
            m, lo, hi = wilson(int(sum(v)), len(v))
            summary[split] = {"success": m, "ci95": [lo, hi], "n": len(v)}
            print(f"{split:8s} {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(v)})")
    for t in tasks:
        print(f"  {task_name(t):14s} {np.mean(per[t]):.2f}")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
