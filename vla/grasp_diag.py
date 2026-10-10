"""Grasp-precision diagnostics for a SmolVLA checkpoint: bias vs scatter, and offline vs closed-loop error.

1. Closed loop: roll out the seen-task eval episodes (same seeds as eval_smolvla.py) until the gripper first closes;
   record the signed EE - target-cube offset there (world x, y, z, and along / across the finger closing axis).
   Normal and zero-noise sampling. The scripted expert on the same layouts gives the reference offset.
2. Offline: on N grasp-moment frames from the training episodes, and on N from fresh layouts (scripted expert,
   seeds disjoint from training and eval) (the frame where the recorded gripper command first
   switches to closed, and the frame `h` steps before it), predict an action chunk and compare the predicted EE target
   at the recorded close index with the recorded one.

    python vla/grasp_diag.py --run checkpoints/vla/V0_2cam_unfrozen_30k --step 30000 --out grasp_diag.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import (base_config, build_policy, image_spec, list_checkpoints, load_run_config,  # noqa: E402
                        load_trainable, open_datasets, parse_data_spec, pick_device, training_side_info)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.evaluate import EVAL_SEED_BASE  # noqa: E402
from sim.env import ALL_TASKS, DEFAULT_SPLIT, PickPlaceEnv, env_config, resolve_heldout, task_name  # noqa: E402
from sim.expert import ScriptedExpert  # noqa: E402
from vla.eval_smolvla import to_batch  # noqa: E402


def finger_axis(e):
    """Unit xy vector of the finger closing axis (left -> right finger)."""
    v = e.d.xpos[e.m.body("right_finger").id][:2] - e.d.xpos[e.m.body("left_finger").id][:2]
    return v / (np.linalg.norm(v) + 1e-9)


def grasp_record(e, task):
    off = e.ee_pos() - e.cube_pos(task[0])
    ax = finger_axis(e)
    d = [np.linalg.norm(e.ee_pos()[:2] - e.cube_pos(j)[:2]) for j in range(3)]
    return {"off": off.tolist(), "along": float(off[:2] @ ax), "across": float(off[0] * -ax[1] + off[1] * ax[0]),
            "axis": ax.tolist(), "nearest_cube_is_target": bool(int(np.argmin(d)) == task[0])}


def rollout_to_grasp(policy, pre, post, jobs, dev, env_cfg, cams, size, noise_scale):
    envs = [PickPlaceEnv(env_cfg) for _ in jobs]
    for e, (t, s) in zip(envs, jobs):
        e.reset(task=t, seed=s)
    policy.reset()
    tasks = [t for t, _ in jobs]
    rec, was_open, obs = [None] * len(jobs), np.zeros(len(jobs), bool), None
    for step in range(envs[0].cfg.max_steps):
        if obs is None or len(policy._queues["action"]) == 0:
            obs = pre(to_batch(envs, tasks, dev, cams, size))
        noise = None
        if noise_scale != 1.0 and len(policy._queues["action"]) == 0:
            c = policy.config
            noise = noise_scale * torch.randn(len(envs), c.chunk_size, c.max_action_dim, device=dev)
        with torch.no_grad():
            act = post(policy.select_action(obs, noise=noise)).cpu().numpy()
        for i, e in enumerate(envs):
            if rec[i] is not None:
                continue
            if act[i][3] < 0.5:
                was_open[i] = True
            elif was_open[i]:
                rec[i] = dict(grasp_record(e, tasks[i]), step=step)
                continue
            e.step(act[i])
        if all(r is not None for r in rec):
            break
    for e in envs:
        e.close()
    return rec


def expert_grasps(jobs, env_cfg):
    out = []
    e = PickPlaceEnv(env_cfg)
    for t, s in jobs:
        e.reset(task=t, seed=s)
        ex = ScriptedExpert(e, s)
        ex.reset()
        was_open, r = False, None
        for k in range(e.cfg.max_steps):
            a = ex.act(k)
            if a[3] < 0.5:
                was_open = True
            elif was_open:
                r = grasp_record(e, t)
                break
            e.step(a)
        out.append(r)
    return out


def stats(recs):
    """Bias (mean vector) vs scatter (std) of the xy offset, in cm; over grasps aimed at the target cube."""
    r = [x for x in recs if x is not None]
    aimed = [x for x in r if x["nearest_cube_is_target"]]
    s = {"n_grasps": len(r), "frac_aimed_at_target": len(aimed) / max(1, len(r))}
    if aimed:
        o = np.array([x["off"] for x in aimed]) * 100
        al = np.array([x["along"] for x in aimed]) * 100
        ac = np.array([x["across"] for x in aimed]) * 100
        mean_xy = o[:, :2].mean(0)
        s.update(mean_xy_cm=mean_xy.round(2).tolist(), std_xy_cm=o[:, :2].std(0).round(2).tolist(),
                 bias_norm_cm=round(float(np.linalg.norm(mean_xy)), 2),
                 rms_xy_cm=round(float(np.sqrt((o[:, :2] ** 2).sum(1).mean())), 2),
                 median_xy_err_cm=round(float(np.median(np.linalg.norm(o[:, :2], axis=1))), 2),
                 along_mean_std_cm=[round(float(al.mean()), 2), round(float(al.std()), 2)],
                 across_mean_std_cm=[round(float(ac.mean()), 2), round(float(ac.std()), 2)],
                 z_above_cube_mean_std_cm=[round(float(o[:, 2].mean()), 2), round(float(o[:, 2].std()), 2)],
                 finger_axis_xy=np.mean([x["axis"] for x in aimed], 0).round(3).tolist())
    return s


def offline(policy, pre, post, cfg, root, n, hs, dev, seed=0):
    (ds,), _ = open_datasets([root], cfg.chunk_size)
    cols = ds.hf_dataset.select_columns(["action", "episode_index"])
    act = np.stack([np.asarray(a, np.float32) for a in cols["action"]])
    ep = np.asarray(cols["episode_index"]).reshape(-1)
    closes = []                                       # global frame index of each episode's first open -> closed switch
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        g = act[idx, 3] > 0.5
        opened = np.flatnonzero(~g)
        if len(opened):
            c = next((k for k in range(opened[0] + 1, len(g)) if g[k]), None)
            if c is not None:
                closes.append(int(idx[c]))
    rng = np.random.default_rng(seed)
    pick = rng.choice(closes, min(n, len(closes)), replace=False)
    out = {}
    for h in hs:
        for mode, scale in (("normal", 1.0), ("zeronoise", 0.0)):
            errs = []
            for s in range(0, len(pick), 20):
                ids = [int(c) for c in pick[s:s + 20] if c - h >= 0 and ep[c - h] == ep[c]]
                b = torch.utils.data.default_collate([ds[c - h] for c in ids])
                noise = scale * torch.randn(len(ids), cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(s)).to(dev)
                with torch.no_grad():
                    pred = post(policy.predict_action_chunk(pre({k: v for k, v in b.items() if k != "action"}),
                                                            noise=noise)).float().cpu().numpy()
                policy.reset()
                errs.append(pred[:, h, :2] - act[ids, :2])
            e = np.concatenate(errs) * 100
            out[f"h{h}_{mode}"] = {"n": len(e), "mean_xy_cm": [round(float(x), 2) for x in e.mean(0)],
                                   "std_xy_cm": [round(float(x), 2) for x in e.std(0)],
                                   "median_err_cm": round(float(np.median(np.linalg.norm(e, axis=1))), 2),
                                   "rms_err_cm": round(float(np.sqrt((e ** 2).sum(1).mean())), 2)}
    return out


def offline_heldout(policy, pre, post, cfg, tasks, n, hs, dev, env_cfg, cams, size, batch=20):
    """Offline check on fresh layouts never used for training or eval: the scripted expert runs to its grasp close
    (index tc); the policy, given the observation at tc - h, predicts a chunk; its entry h is compared with the
    expert's action at tc. Same comparison as offline() on training frames."""
    per = -(-n // len(tasks))
    jobs = [(t, EVAL_SEED_BASE + ALL_TASKS.index(t) * 10_000 + 5_000 + k) for t in tasks for k in range(per)][:n]
    plans = []
    e = PickPlaceEnv(env_cfg)
    for t, s in jobs:                                    # expert actions and close index per layout
        e.reset(task=t, seed=s)
        ex = ScriptedExpert(e, s)
        ex.reset()
        acts = np.stack([ex.act(k) for k in range(min(len(ex), e.cfg.max_steps))])
        g = acts[:, 3] > 0.5
        opened = np.flatnonzero(~g)
        tc = next((k for k in range(opened[0] + 1, len(g)) if g[k]), None) if len(opened) else None
        if tc is not None:
            plans.append((t, s, acts, tc))
    out = {}
    for h in hs:
        for mode, scale in (("normal", 1.0), ("zeronoise", 0.0)):
            errs = []
            for b0 in range(0, len(plans), batch):
                envs, tasks_b, tgt = [], [], []
                for t, s, acts, tc in plans[b0:b0 + batch]:
                    if tc - h < 0:
                        continue
                    env = PickPlaceEnv(env_cfg)
                    env.reset(task=t, seed=s)
                    for k in range(tc - h):
                        env.step(acts[k])
                    envs.append(env)
                    tasks_b.append(t)
                    tgt.append(acts[tc, :2])
                g = torch.Generator().manual_seed(b0)
                noise = scale * torch.randn(len(envs), cfg.chunk_size, cfg.max_action_dim, generator=g).to(dev)
                with torch.no_grad():
                    pred = post(policy.predict_action_chunk(pre(to_batch(envs, tasks_b, dev, cams, size)),
                                                            noise=noise)).float().cpu().numpy()
                policy.reset()
                errs.append(pred[:, h, :2] - np.array(tgt))
                for env in envs:
                    env.close()
            er = np.concatenate(errs) * 100
            out[f"h{h}_{mode}"] = {"n": len(er), "mean_xy_cm": [round(float(x), 2) for x in er.mean(0)],
                                   "std_xy_cm": [round(float(x), 2) for x in er.std(0)],
                                   "median_err_cm": round(float(np.median(np.linalg.norm(er, axis=1))), 2),
                                   "rms_err_cm": round(float(np.sqrt((er ** 2).sum(1).mean())), 2)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", default="latest")
    p.add_argument("--k", type=int, default=10, help="closed-loop episodes per seen task")
    p.add_argument("--n_offline", type=int, default=100)
    p.add_argument("--hs", default="0,10", help="offline: query frames this many steps before the recorded close")
    p.add_argument("--batch_envs", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = pick_device(a.device)
    rc = load_run_config(a.run)
    side = training_side_info(rc)
    env_cfg = env_config(side.get("env_cfg"))
    cams, size = image_spec(rc["features"])
    heldout = resolve_heldout(",".join(side["heldout"]) if "heldout" in side else DEFAULT_SPLIT)
    policy, pre, post, cfg = build_policy(rc["features"], rc["stats"], dev, n_action_steps=10)
    cks = dict(list_checkpoints(a.run))
    step = max(cks) if a.step == "latest" else int(a.step)
    load_trainable(policy, cks[step])
    policy.eval()
    tasks = [t for t in ALL_TASKS if t not in heldout]
    hs = [int(h) for h in a.hs.split(",")]
    if os.path.exists(a.out):   # earlier results: keep them, add only what is missing (the held-out offline check)
        with open(a.out) as f:
            res = json.load(f)
        if "offline_heldout" not in res:
            res["offline_heldout"] = offline_heldout(policy, pre, post, cfg, tasks, a.n_offline, hs, dev, env_cfg,
                                                     cams, size)
            for k, v in res["offline_heldout"].items():
                print("offline_heldout", k, v, flush=True)
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1)
        return
    jobs = [(t, EVAL_SEED_BASE + ALL_TASKS.index(t) * 10_000 + k) for t in tasks for k in range(a.k)]
    res = {"run": a.run, "step": step, "k": a.k, "closed_loop": {}}
    res["closed_loop"]["expert"] = stats(expert_grasps(jobs, env_cfg))
    print("expert", res["closed_loop"]["expert"], flush=True)
    per_ep = {}
    for mode, scale in (("normal", 1.0), ("zeronoise", 0.0)):
        recs = []
        for i in range(0, len(jobs), a.batch_envs):
            recs += rollout_to_grasp(policy, pre, post, jobs[i:i + a.batch_envs], dev, env_cfg, cams, size, scale)
        res["closed_loop"][mode] = stats(recs)
        res["closed_loop"][mode + "_per_task"] = {
            task_name(t): stats([r for (tt, _), r in zip(jobs, recs) if tt == t]).get("mean_xy_cm") for t in tasks}
        per_ep[mode] = recs
        print(mode, res["closed_loop"][mode], flush=True)
    root = parse_data_spec(rc["args"]["data"][0])[0]
    res["offline"] = offline(policy, pre, post, cfg, root, a.n_offline, hs, dev)
    for k, v in res["offline"].items():
        print("offline", k, v, flush=True)
    res["offline_heldout"] = offline_heldout(policy, pre, post, cfg, tasks, a.n_offline, hs, dev, env_cfg, cams, size)
    for k, v in res["offline_heldout"].items():
        print("offline_heldout", k, v, flush=True)
    res["episodes"] = {m: [dict(task=task_name(t), seed=s, **(r or {})) for (t, s), r in zip(jobs, recs)]
                       for m, recs in per_ep.items()}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
