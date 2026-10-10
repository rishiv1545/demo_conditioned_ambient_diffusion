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

from vla.common import (build_policy, image_spec, list_checkpoints, load_run_config, load_trainable,  # noqa: E402
                        pick_device, training_side_info)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.evaluate import EVAL_SEED_BASE, wilson  # noqa: E402
from sim.env import (ALL_TASKS, CUBES, DEFAULT_SPLIT, ZONES, PickPlaceEnv, env_config,  # noqa: E402
                     instruction, resolve_heldout, task_name)


def to_batch(envs, tasks, dev, cams, size):
    imgs = {c: [] for c in cams}
    for e in envs:
        for c, im in e.images(size, cams).items():
            imgs[c].append(torch.from_numpy(im).permute(2, 0, 1).float() / 255.0)
    b = {f"observation.images.{c}": torch.stack(v) for c, v in imgs.items()}
    b["observation.state"] = torch.from_numpy(np.stack([e.state() for e in envs]))
    b["task"] = [instruction(t) for t in tasks]
    return b


def run_batch(policy, pre, post, jobs, dev, hold=5, record=None, env_cfg=None, cams=("phone",), size=256,
              no_distractors=False):
    """jobs: list of (task, seed). Returns list of (success, steps) and, for indices in `record`, frame lists."""
    envs = [PickPlaceEnv(env_cfg) for _ in jobs]
    for e, (t, s) in zip(envs, jobs):
        e.reset(task=t, seed=s)
        if no_distractors:
            e.remove_distractors()
    policy.reset()
    tasks = [t for t, _ in jobs]
    streak = np.zeros(len(jobs), int)
    lifted = np.zeros(len(jobs))                       # max height of the target cube (failure analysis)
    grasp_err = [None] * len(jobs)                     # EE - target cube offset when the gripper first closes
    was_open = np.zeros(len(jobs), bool)
    done = np.zeros(len(jobs), bool)
    steps = np.zeros(len(jobs), int)
    frames = {i: [] for i in (record or [])}
    max_steps = envs[0].cfg.max_steps
    obs = None
    for _ in range(max_steps):
        for i in frames:
            frames[i].append(envs[i].render("front"))
        # select_action only looks at the observation when its action queue is empty (once per n_action_steps);
        # otherwise it pops a queued action. Rendering (~0.1 s per 512 px image on Colab's software EGL) dominates
        # eval time, so render only when a new chunk will be planned. Same actions as rendering every step.
        if obs is None or len(policy._queues["action"]) == 0:
            obs = pre(to_batch(envs, tasks, dev, cams, size))
        with torch.no_grad():
            act = post(policy.select_action(obs)).cpu().numpy()
        for i, e in enumerate(envs):
            if done[i]:
                continue                     # finished envs are frozen (their actions are ignored)
            if act[i][3] < 0.5:
                was_open[i] = True
            elif was_open[i] and grasp_err[i] is None:
                grasp_err[i] = (e.ee_pos() - e.cube_pos(tasks[i][0])).tolist()
            _, s = e.step(act[i])
            lifted[i] = max(lifted[i], e.cube_pos(tasks[i][0])[2])
            steps[i] += 1
            streak[i] = streak[i] + 1 if s else 0
            done[i] = streak[i] >= hold
        if done.all():
            break
    diag = []
    for i, e in enumerate(envs):
        c, z = tasks[i]
        p = e.cube_pos(c)
        dz = [float(np.abs(p[:2] - zc).max()) for zc in e.layout["zones"]]
        others = [] if no_distractors else \
            [j for j in range(3) if j != c and np.linalg.norm(e.cube_pos(j)[:2] - e.layout["cubes"][j]) > 0.03]
        diag.append({"target_lifted": bool(lifted[i] > 0.035), "nearest_zone": int(np.argmin(dz)),
                     "in_target_zone": bool(dz[z] <= e.cfg.zone_half), "cube_z": float(p[2]),
                     "held_at_end": bool(np.linalg.norm(e.ee_pos() - p) < e.cfg.release_dist),
                     "other_cubes_moved": others, "grasp_offset": grasp_err[i]})
        e.close()
    return [(bool(d), int(n), g) for d, n, g in zip(done, steps, diag)], frames


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
    p.add_argument("--train_seeds", action="store_true",
                   help="diagnostic: use the layouts of the training demos (scripts/gen_sim_data.py seeds)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no_distractors", action="store_true",
                   help="diagnostic: only the named cube and zone in the scene (others moved out of view)")
    a = p.parse_args()
    dev = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    rc = load_run_config(a.run)
    side = training_side_info(rc)
    spec = a.heldout or (",".join(side["heldout"]) if "heldout" in side else DEFAULT_SPLIT)
    env_cfg = env_config(side.get("env_cfg"))   # the env version of the training data (absent: v1)
    cams, size = image_spec(rc["features"])
    print(f"cameras {cams} at {size} px; env: gripper yaw {env_cfg.gripper_yaw_deg:.0f} deg, starts {'closed' if env_cfg.start_gripper_closed else 'open'}")
    heldout = resolve_heldout(spec)
    policy, pre, post, _ = build_policy(rc["features"], rc["stats"], dev, n_action_steps=a.n_action_steps)
    step = "base"
    if a.step != "base":
        cks = dict(list_checkpoints(a.run))
        step = max(cks) if a.step == "latest" else int(a.step)
        load_trainable(policy, cks[step])
    policy.eval()
    tasks = {"all": ALL_TASKS, "seen": [t for t in ALL_TASKS if t not in heldout], "heldout": heldout}[a.tasks]
    base = (lambda t: ALL_TASKS.index(t) * 100_000) if a.train_seeds else \
        (lambda t: EVAL_SEED_BASE + ALL_TASKS.index(t) * 10_000)
    jobs = [(t, base(t) + k) for t in tasks for k in range(a.k)]
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
        res, frames = run_batch(policy, pre, post, chunk, dev, record=rec, env_cfg=env_cfg, cams=cams, size=size,
                                no_distractors=a.no_distractors)
        for (t, s), (ok, n, g) in zip(chunk, res):
            rows.append((t, s, ok, n, g))
        for j, fr in frames.items():
            t, s = chunk[j]
            ok = res[j][0]
            imageio.mimsave(os.path.join(a.out, f"{'success' if ok else 'fail'}_{task_name(t)}_{s}.mp4"),
                            fr[:res[j][1] + 1], fps=10, macro_block_size=1)
        print(f"{len(rows)}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seed", "success", "steps", "target_lifted", "in_target_zone", "nearest_zone",
                    "held_at_end", "other_cubes_moved"])
        for t, s, ok, n, g in rows:
            w.writerow([task_name(t), s, int(ok), n, int(g["target_lifted"]), int(g["in_target_zone"]),
                        ZONES[g["nearest_zone"]], int(g["held_at_end"]), "|".join(CUBES[j] for j in g["other_cubes_moved"])])
    per = {t: [r[2] for r in rows if r[0] == t] for t in tasks}
    summary = {"run": a.run, "step": step, "k": a.k, "n_action_steps": a.n_action_steps,
               "no_distractors": a.no_distractors,
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
    # failure analysis
    fails = [r for r in rows if not r[2]]
    if fails:
        g = [r[4] for r in fails]
        fa = {"n_fail": len(g), "target_lifted": float(np.mean([x["target_lifted"] for x in g])),
              "wrong_cube_moved": float(np.mean([bool(x["other_cubes_moved"]) for x in g])),
              "ended_in_target_zone": float(np.mean([x["in_target_zone"] for x in g])),
              "still_held_at_end": float(np.mean([x["held_at_end"] for x in g]))}
        go = np.array([x["grasp_offset"] for x in g if x["grasp_offset"] is not None])
        if len(go):
            fa.update(grasp_xy_err_cm_median=float(np.median(np.linalg.norm(go[:, :2], axis=1)) * 100),
                      grasp_z_above_cube_cm_median=float(np.median(go[:, 2]) * 100), n_grasps=len(go))
        summary["failure_analysis"] = fa
        print("failures:", {k: round(v, 2) for k, v in fa.items()})
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
