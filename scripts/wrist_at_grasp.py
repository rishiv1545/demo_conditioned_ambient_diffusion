"""Wrist-camera view at the grasp moment: scripted expert vs a SmolVLA checkpoint, same seen-task eval episodes.

For each episode, the frames 1 s before and at the step where the gripper command first switches to closed (after
having been open), from the wrist and scene cameras, plus the EE-to-target-cube offset at that moment. Writes one
contact sheet (rows = episodes; columns = expert wrist -1 s, expert wrist @close, policy wrist -1 s, policy wrist
@close, policy scene @close).

    python scripts/wrist_at_grasp.py --run checkpoints/vla/V0_2cam_unfrozen_30k --step 30000 --out outputs/wrist_check
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.evaluate import EVAL_SEED_BASE  # noqa: E402
from sim.env import ALL_TASKS, PickPlaceEnv, env_config, task_name  # noqa: E402
from sim.expert import ScriptedExpert  # noqa: E402
from vla.common import (build_policy, image_spec, list_checkpoints, load_run_config, load_trainable,  # noqa: E402
                        pick_device, training_side_info)
from vla.eval_smolvla import to_batch  # noqa: E402

SIZE = 512


def label(im, txt):
    im = cv2.resize(im, (256, 256))
    cv2.rectangle(im, (0, 0), (256, 18), (255, 255, 255), -1)
    cv2.putText(im, txt, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return im


def rollout(env, task, seed, act_fn, cams):
    """Run until the first open->closed switch (or the step limit). Returns frames at close-10 and close, the EE-cube
    offset at close, and whether the episode later succeeds is not needed here."""
    env.reset(task=task, seed=seed)
    hist, was_open = [], False
    for t in range(env.cfg.max_steps):
        hist.append({c: im for c, im in env.images(SIZE, cams).items()})
        a = act_fn(env, t)
        if a[3] < 0.5:
            was_open = True
        elif was_open:
            off = env.ee_pos() - env.cube_pos(task[0])
            return hist[max(0, t - 10)], hist[t], off, t
        env.step(a)
    return hist[-11], hist[-1], None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", default="latest")
    p.add_argument("--out", default="outputs/wrist_check")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    dev = pick_device(a.device)
    os.makedirs(a.out, exist_ok=True)
    rc = load_run_config(a.run)
    env_cfg = env_config(training_side_info(rc).get("env_cfg"))
    cams, _ = image_spec(rc["features"])
    policy, pre, post, _ = build_policy(rc["features"], rc["stats"], dev, n_action_steps=10)
    cks = dict(list_checkpoints(a.run))
    load_trainable(policy, cks[max(cks) if a.step == "latest" else int(a.step)])
    policy.eval()
    episodes = [("red-yellow", 0), ("red-purple", 1), ("green-yellow", 2), ("green-orange", 3), ("red-orange", 4)]
    rows = []
    for tname, k in episodes:
        task = next(t for t in ALL_TASKS if task_name(t) == tname)
        seed = EVAL_SEED_BASE + ALL_TASKS.index(task) * 10_000 + k
        env = PickPlaceEnv(env_cfg)
        env.reset(task=task, seed=seed)
        ex = ScriptedExpert(env, seed)
        ex.reset()
        e_pre, e_at, e_off, e_t = rollout(env, task, seed, lambda e, t: ex.act(t), cams)
        policy.reset()

        def pol_act(e, t):
            with torch.no_grad():
                return post(policy.select_action(pre(to_batch([e], [task], dev, cams, SIZE)))).cpu().numpy()[0]
        p_pre, p_at, p_off, p_t = rollout(env, task, seed, pol_act, cams)
        f = lambda o: "no close" if o is None else f"off {o[0] * 100:+.1f},{o[1] * 100:+.1f},{o[2] * 100:+.1f}cm"
        print(f"{tname} seed {seed}: expert close t={e_t} {f(e_off)} | policy close t={p_t} {f(p_off)}", flush=True)
        rows.append(np.hstack([label(e_pre["wrist"], f"{tname} expert wrist -1s"),
                               label(e_at["wrist"], f"expert @close {f(e_off)}"),
                               label(p_pre["wrist"], "V0 wrist -1s"),
                               label(p_at["wrist"], f"V0 @close {f(p_off)}"),
                               label(p_at["scene"], "V0 scene @close")]))
    path = os.path.join(a.out, "wrist_at_grasp.jpg")
    cv2.imwrite(path, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print("wrote", path)


if __name__ == "__main__":
    main()
