"""Recovery demos for the grasp-precision problem (DART-style), from the scripted expert.

Each episode (fresh layout):
- with probability --p_start, the gripper starts away from HOME (xy +-8 cm, z +-5 cm; z kept >= 8 cm), and the
  expert plans from there;
- 1-2 perturbation windows during the approach (between the gripper opening and the grasp close): for 3-6 steps the
  EXECUTED target is shifted 1-3 cm sideways (at most 1 cm within 4 cm above the grasp height, so the fingers don't
  hit the cube), then returns to the plan.
Labels (`action`) are the expert's nominal plan: actions are absolute EE targets, so from a perturbed state the next
nominal targets point back to the path. The executed actions (`exec_action`) and the start (`ee_start`) are stored so
vla/export_lerobot.py re-simulates exactly what happened. Only successful episodes are kept; the success rate under
perturbation is reported.

    python scripts/gen_recovery_data.py --out data/sim_v2_recovery --n_per_task 60 --tasks seen
"""
import argparse
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from policy.data import save_episode  # noqa: E402
from sim.env import ALL_TASKS, DEFAULT_SPLIT, PickPlaceEnv, env_cfg_dict, resolve_heldout, task_name  # noqa: E402
from sim.expert import ScriptedExpert  # noqa: E402

SEED_OFFSET = 60_000   # seeds task_index * 100_000 + 60_000 + k: disjoint from gen_sim_data (k < 200) and eval (>= 1e6)
_env = None


def perturb(nominal, rng, n_windows, travel=(0.01, 0.03), descent_max=0.01):
    """Executed actions = nominal + 1-2 sideways offset windows during the approach. Returns (executed, windows)."""
    a = nominal.copy()
    g = nominal[:, 3] > 0.5
    opened = np.flatnonzero(~g)
    if not len(opened):
        return a, []
    t_open = opened[0]
    tc = next((t for t in range(t_open + 1, len(g)) if g[t]), None)
    if tc is None or tc - t_open < 8:
        return a, []
    z_grasp = nominal[tc, 2]
    wins = []
    for _ in range(n_windows):
        L = int(rng.integers(3, 7))
        s = int(rng.integers(t_open, max(t_open + 1, tc - L)))
        descent = nominal[s:s + L, 2].min() < z_grasp + 0.04      # fingers within 4 cm of the cube top
        mag = rng.uniform(0.003, descent_max) if descent else rng.uniform(*travel)
        ang = rng.uniform(0, 2 * np.pi)
        a[s:min(s + L, tc), :2] += mag * np.array([np.cos(ang), np.sin(ang)])
        wins.append({"start": s, "len": L, "offset_cm": round(mag * 100, 2), "descent": bool(descent)})
    return a, wins


def episode(env, task, seed, p_start, hold_steps=5):
    rng = np.random.default_rng(seed)
    ee_start = None
    if rng.random() < p_start:
        home = np.array(env.cfg.home_ee)
        ee_start = home + np.r_[rng.uniform(-0.08, 0.08, 2), rng.uniform(-0.05, 0.05)]
        ee_start[2] = max(ee_start[2], 0.08)
    obs = env.reset(task=task, seed=seed, ee_start=ee_start)
    start = env.ee_pos().copy()
    ex = ScriptedExpert(env, seed=seed + 1_000_003)
    ex.reset()
    T = min(len(ex) + hold_steps, env.cfg.max_steps)
    nominal = np.stack([ex.act(t) for t in range(T)]).astype(np.float32)
    executed, wins = perturb(nominal, rng, int(rng.integers(1, 3)))
    O, succ = [], False
    for t in range(T):
        O.append(obs)
        obs, succ = env.step(executed[t])
    return {"obs": np.stack(O), "action": nominal, "exec_action": executed.astype(np.float32), "success": succ,
            "ee_start": start if ee_start is not None else None, "windows": wins}


def _work(args):
    global _env
    task, n, out, p_start = args
    _env = _env or PickPlaceEnv()
    ti = ALL_TASKS.index(task)
    kept, tried, n_win = 0, 0, 0
    while kept < n:
        seed = ti * 100_000 + SEED_OFFSET + tried
        tried += 1
        r = episode(_env, task, seed, p_start)
        if not r["success"]:
            continue
        extra = {"exec_action": r["exec_action"]}
        if r["ee_start"] is not None:
            extra["ee_start"] = r["ee_start"]
        save_episode(os.path.join(out, f"{task_name(task)}_{kept:03d}.npz"), r["obs"], r["action"], task, "sim",
                     _env.layout, True, meta={"seed": seed, "env_cfg": env_cfg_dict(_env.cfg), "recovery": True,
                                              "perturbed_start": r["ee_start"] is not None, "windows": r["windows"]},
                     **extra)
        kept += 1
        n_win += len(r["windows"])
    return task, kept, tried, n_win


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/sim_v2_recovery")
    p.add_argument("--n_per_task", type=int, default=60)
    p.add_argument("--tasks", default="seen")
    p.add_argument("--heldout", default=DEFAULT_SPLIT)
    p.add_argument("--p_start", type=float, default=0.5, help="fraction of episodes with a perturbed start")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    heldout = resolve_heldout(a.heldout)
    tasks = {"seen": [t for t in ALL_TASKS if t not in heldout], "heldout": heldout, "all": ALL_TASKS}[a.tasks]
    os.makedirs(a.out, exist_ok=True)
    with get_context("spawn").Pool(min(a.workers, len(tasks))) as pool:
        res = pool.map(_work, [(t, a.n_per_task, a.out, a.p_start) for t in tasks])
    tot_k = tot_t = 0
    for task, kept, tried, n_win in res:
        print(f"{task_name(task):14s} kept {kept}/{tried} ({kept / tried:.0%} success under perturbation), "
              f"{n_win} perturbation windows")
        tot_k += kept
        tot_t += tried
    print(f"all: {tot_k}/{tot_t} = {tot_k / tot_t:.1%} expert success under perturbation")


if __name__ == "__main__":
    main()
