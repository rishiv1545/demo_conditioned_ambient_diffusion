"""Scripted waypoint expert with randomized hover heights, jitter, speed and pauses."""
import numpy as np

CUBE_Z = 0.02  # cube center height when resting on the table


def plan(env, rng):
    """Return a list of (ee_target[3], gripper, duration_s) segments for env.task, from env's current state."""
    c, z = env.task
    cube = env.layout["cubes"][c]
    zone = env.layout["zones"][z]
    jit = lambda s: rng.uniform(-s, s, size=2)
    v = rng.uniform(0.15, 0.30)                       # travel speed (m/s)
    v_down = v * rng.uniform(0.4, 0.7)                # slower approach
    h1, h2 = rng.uniform(0.07, 0.13), rng.uniform(0.09, 0.15)
    grasp = np.array([*(cube + jit(0.004)), CUBE_Z + rng.uniform(-0.004, 0.003)])
    place = np.array([*(zone + jit(0.012)), CUBE_Z + rng.uniform(0.004, 0.012)])
    pre_grasp = np.array([*grasp[:2], h1])
    pre_place = np.array([*place[:2], h2])
    retreat = np.array([*place[:2], rng.uniform(0.10, 0.16)])

    segs, pos = [], env.ee_pos().copy()

    def move(target, grip, speed, settle=0.0):
        nonlocal pos
        dur = np.linalg.norm(target - pos) / speed + settle
        segs.append((target, grip, max(dur, 0.1)))
        pos = target

    def wait(grip, dur):
        segs.append((pos.copy(), grip, dur))

    def pause(grip):
        if rng.random() < 0.5:
            wait(grip, rng.uniform(0.05, 0.3))

    if env.cfg.start_gripper_closed:          # v2: like the human's relaxed hand: rest closed, then open
        wait(1.0, rng.uniform(0.2, 0.6))
    move(pre_grasp, 0.0, v, settle=0.2); pause(0.0)
    move(grasp, 0.0, v_down, settle=0.2)
    wait(1.0, rng.uniform(0.5, 0.7))                 # close
    move(np.array([*grasp[:2], h2]), 1.0, v_down); pause(1.0)
    move(pre_place, 1.0, v, settle=0.2); pause(1.0)
    move(place, 1.0, v_down, settle=0.2)
    wait(0.0, rng.uniform(0.4, 0.6))                 # open
    move(retreat, 0.0, v_down)
    if env.cfg.start_gripper_closed:          # v2: return to HOME and relax (close), like the human demos
        home = np.array(env.cfg.home_ee) + np.r_[rng.uniform(-0.01, 0.01, 2), 0.0]
        move(home, 0.0, v)
        wait(1.0, rng.uniform(0.3, 0.6))
    return segs


def rollout_actions(segs, dt=0.1, start=None):
    """Turn segments into a 10 Hz action sequence (linear interpolation of the EE target)."""
    acts, pos = [], np.asarray(start, dtype=float)
    for target, grip, dur in segs:
        n = max(1, int(round(dur / dt)))
        for k in range(1, n + 1):
            p = pos + (target - pos) * (k / n)
            acts.append(np.array([*p, grip]))
        pos = np.asarray(target, dtype=float)
    return np.stack(acts)


class ScriptedExpert:
    """Open-loop plan made at reset time (objects don't move before the grasp), padded with a final hold."""

    def __init__(self, env, seed=None):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.actions = None

    def reset(self):
        segs = plan(self.env, self.rng)
        self.actions = rollout_actions(segs, 1.0 / self.env.cfg.control_hz, start=self.env.ee_pos().copy())

    def act(self, t):
        return self.actions[min(t, len(self.actions) - 1)]

    def __len__(self):
        return len(self.actions)


def run_expert_episode(env, task, seed, hold_steps=5, render=None):
    """Run one expert episode. Returns dict(obs, action, success, frames)."""
    obs = env.reset(task=task, seed=seed)
    ex = ScriptedExpert(env, seed=seed + 1_000_003 if seed is not None else None)
    ex.reset()
    T = min(len(ex) + hold_steps, env.cfg.max_steps)
    O, A, frames = [], [], []
    succ = False
    for t in range(T):
        a = ex.act(t)
        O.append(obs)
        A.append(a.astype(np.float32))
        if render:
            frames.append(env.render(render))
        obs, succ = env.step(a)
    if render:
        frames.append(env.render(render))
    return {"obs": np.stack(O), "action": np.stack(A), "success": succ, "frames": frames}
