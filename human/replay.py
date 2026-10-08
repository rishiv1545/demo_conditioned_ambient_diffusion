"""Replay a retargeted human trajectory open-loop in sim and record it in the shared episode format."""
import numpy as np

from policy.data import save_episode

HOLD_STEPS = 5  # extra steps holding the last action, as in the expert data


def replay(env, task, layout, ee, grip, render=None):
    """Reset with the detected layout, start the arm at the first retargeted point and execute the trajectory.
    Returns dict(obs, action, success, track_err, frames)."""
    obs = env.reset(task=task, layout=layout, ee_start=ee[0])
    acts = np.column_stack([ee, grip]).astype(np.float32)
    acts = np.concatenate([acts, np.repeat(acts[-1:], HOLD_STEPS, 0)])
    O, errs, frames = [], [], []
    succ = False
    for a in acts:
        O.append(obs)
        if render:
            frames.append(render(env) if callable(render) else env.render(render))
        obs, succ = env.step(a)
        errs.append(np.linalg.norm(env.ee_pos() - a[:3]))
    if render:
        frames.append(render(env) if callable(render) else env.render(render))
    return {"obs": np.stack(O), "action": acts[:len(O)], "success": succ, "track_err": float(np.mean(errs)),
            "track_err_max": float(np.max(errs)), "frames": frames}


def save_human_episode(path, r, task, layout, raw_traj, raw_gripper, meta):
    meta = dict(meta, track_err=r["track_err"], track_err_max=r["track_err_max"])
    save_episode(path, r["obs"], r["action"], task, "human", layout, r["success"], sigma_n=float("nan"),
                 meta=meta, raw_traj=np.asarray(raw_traj, np.float32), raw_gripper=np.asarray(raw_gripper, np.float32))
