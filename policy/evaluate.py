"""Evaluate a checkpoint on all 9 tasks with fixed held-out seeds, in parallel CPU workers.

    python policy/evaluate.py --ckpt checkpoints/A/ckpt.pt --k 50 --out outputs/m3/A

An episode counts as a success once env.success() holds for `--hold` consecutive control steps (cube in the zone,
on the table, gripper open), or fails at max_steps (200 = 20 s).
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.data import Normalizer  # noqa: E402
from policy.diffusion import Diffusion  # noqa: E402
from sim.env import ALL_TASKS, DEFAULT_SPLIT, PickPlaceEnv, resolve_heldout, task_name, task_onehot  # noqa: E402

EVAL_SEED_BASE = 1_000_000


class PolicyRunner:
    """Receding-horizon execution: sample a 16-step chunk, execute the first n_exec actions, re-plan."""

    def __init__(self, ckpt, device="cpu", n_exec=8, ddim_steps=10, seed=0):
        from policy.train import load_policy
        self.pol, self.cfg, ck = load_policy(ckpt, device)
        self.on = Normalizer.from_state_dict(ck["norm"]["obs"])
        self.an = Normalizer.from_state_dict(ck["norm"]["act"])
        self.diff = Diffusion(self.cfg.T, device)
        self.dev, self.n_exec, self.ddim_steps = device, n_exec, ddim_steps
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def reset(self, task):
        self.task, self.hist, self.queue = task, [], []

    def act(self, obs):
        self.hist = (self.hist + [obs])[-self.cfg.n_obs:]
        if not self.queue:
            h = [self.hist[0]] * (self.cfg.n_obs - len(self.hist)) + self.hist
            o = np.concatenate([self.on.norm(x) for x in h]).astype(np.float32)  # per-frame stats
            batch = {"obs": torch.as_tensor(o, device=self.dev)[None],
                     "task": torch.as_tensor(task_onehot(self.task), device=self.dev)[None]}
            with torch.no_grad():
                cond = self.pol.encode(batch)
                x = self.diff.sample(self.pol, cond, (1, self.cfg.horizon, 4), self.ddim_steps, self.gen)
            chunk = self.an.denorm(x[0].cpu().numpy())
            self.queue = list(chunk[:self.n_exec])
        return self.queue.pop(0)


def run_episode(env, runner, task, seed, hold=5, render=None):
    obs = env.reset(task=task, seed=seed)
    runner.reset(task)
    streak, frames = 0, []
    for _ in range(env.cfg.max_steps):
        if render:
            frames.append(env.render(render))
        obs, s = env.step(runner.act(obs))
        streak = streak + 1 if s else 0
        if streak >= hold:
            break
    if render:
        frames.append(env.render(render))
    return streak >= hold, env.t, frames


_env, _runner = None, None


def _init(ckpt, n_exec, ddim_steps):
    global _env, _runner
    torch.set_num_threads(1)
    _env = PickPlaceEnv()
    _runner = PolicyRunner(ckpt, "cpu", n_exec, ddim_steps, seed=os.getpid())


def _work(job):
    task, seed, hold = job
    _runner.gen.manual_seed(seed)
    s, T, _ = run_episode(_env, _runner, task, seed, hold)
    return task, seed, s, T


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--n_exec", type=int, default=8)
    p.add_argument("--ddim_steps", type=int, default=10)
    p.add_argument("--hold", type=int, default=5)
    p.add_argument("--heldout", default=None, help="split spec; default: the split the checkpoint was trained with")
    p.add_argument("--videos", type=int, default=3, help="successes and failures to render (each)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    spec = a.heldout or torch.load(a.ckpt, map_location="cpu", weights_only=False)["args"].get("heldout", DEFAULT_SPLIT)
    heldout = resolve_heldout(spec)
    print("held-out tasks:", [task_name(t) for t in heldout])
    jobs = [(t, EVAL_SEED_BASE + ti * 10_000 + k, a.hold) for ti, t in enumerate(ALL_TASKS) for k in range(a.k)]
    t0 = time.time()
    rows = []
    with get_context("spawn").Pool(a.workers, _init, (a.ckpt, a.n_exec, a.ddim_steps)) as pool:
        for r in pool.imap_unordered(_work, jobs, chunksize=2):
            rows.append(r)
    print(f"{len(rows)} episodes in {time.time() - t0:.0f}s")
    rows.sort(key=lambda r: (ALL_TASKS.index(r[0]), r[1]))
    with open(os.path.join(a.out, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seed", "success", "steps"])
        for t, s, ok, T in rows:
            w.writerow([task_name(t), s, int(ok), T])
    per = {t: [r[2] for r in rows if r[0] == t] for t in ALL_TASKS}
    with open(os.path.join(a.out, "per_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "split", "n", "success_rate"])
        for t in ALL_TASKS:
            w.writerow([task_name(t), "heldout" if t in heldout else "seen", len(per[t]), f"{np.mean(per[t]):.3f}"])
            print(f"{task_name(t):14s} {'heldout' if t in heldout else 'seen':8s} {np.mean(per[t]):.2f}")
    summary = {}
    for split, ts in (("seen", [t for t in ALL_TASKS if t not in heldout]), ("heldout", heldout)):
        v = sum((per[t] for t in ts), [])
        m, lo, hi = wilson(int(sum(v)), len(v))
        summary[split] = {"success": m, "ci95": [lo, hi], "n": len(v)}
        print(f"{split:8s} {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(v)})")
    summary["ckpt"] = a.ckpt
    summary["heldout_tasks"] = [task_name(t) for t in heldout]
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # videos: a few successes and failures, re-run deterministically in this process
    if a.videos:
        import imageio.v2 as imageio
        _init(a.ckpt, a.n_exec, a.ddim_steps)
        succ = [r for r in rows if r[2]][:a.videos]
        fail = [r for r in rows if not r[2]]
        fail = [fail[i] for i in np.linspace(0, len(fail) - 1, min(a.videos, len(fail))).astype(int)] if fail else []
        for t, seed, ok, _ in succ + fail:
            _runner.gen.manual_seed(seed)
            s, _, frames = run_episode(_env, _runner, t, seed, a.hold, render="front")
            name = f"{'success' if s else 'fail'}_{task_name(t)}_{seed}.mp4"
            imageio.mimsave(os.path.join(a.out, name), frames, fps=10, macro_block_size=1)


if __name__ == "__main__":
    main()
