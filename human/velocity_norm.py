"""Velocity normalization of retargeted phone trajectories (after Tutor Intelligence DF1 report, Sec. 4.2).

Human demos differ a lot in pace; the policy sees that as action noise. VN re-times each episode (the path itself is
unchanged) before the sim replay, in two stages:

  Stage 1, per episode: speed factor f = median task cruising speed / episode cruising speed, clamped to
           [0.75, 1.5]. Cruising speed = 30th percentile of the EE speed over the moving frames (speed >= idle_v;
           the resting start and end would otherwise put the 30th percentile at ~0).
  Stage 2, per frame: a smooth monotonic speed map h on the stage-1 speed u = f * v: identity below m (the median of
           the per-episode cruising speeds), compressing above, toward M (80th percentile of the stage-1 frame speeds
           over the task):  h(u) = u for u <= m,  m + (M - m) * tanh((u - m) / (M - m)) above  (C1 at m, h < M).

Frame selection by a cursor over the 10 Hz source steps: each output step advances the cursor by the time-warp rate
f * h(u) / u (u = 0: f). Frames within keep_s (0.5 s) of a gripper open/close are always kept at the original rate
(rate 1), and idle frames (speed < idle_v) are skipped, except in the last tail_s (2 s) of the episode. VN runs
before retarget.add_dwell, so the inserted grasp/release dwells are added afterwards and preserved exactly.

    stats = task_stats([ee_1, ee_2, ...])                    # from the retargeted paths of all episodes of a task
    ee, g, t = normalize(ee, g, t, stats, hz=10)             # one episode
"""
import numpy as np

CRUISE_PCT = 30
M_PCT = 80
F_CLAMP = (0.75, 1.5)


def frame_speed(ee, hz=10.0):
    """Speed (m/s) of each 10 Hz step: |ee[k+1] - ee[k]| * hz; the last frame repeats the previous one."""
    v = np.linalg.norm(np.diff(np.asarray(ee, float), axis=0), axis=1) * hz
    return np.append(v, v[-1] if len(v) else 0.0)


def cruise_speed(ee, hz=10.0, idle_v=0.02):
    v = frame_speed(ee, hz)
    moving = v[v >= idle_v]
    return float(np.percentile(moving, CRUISE_PCT)) if len(moving) else 0.0


def task_stats(paths, hz=10.0, idle_v=0.02):
    """{"m": median cruising speed, "M": 80th pct of stage-1 frame speeds, "cruise": [...], "f": [...]} for the
    episodes of one task (paths: retargeted EE paths [N, 3], dwells included or not: they are idle frames)."""
    cruise = np.array([cruise_speed(p, hz, idle_v) for p in paths])
    m = float(np.median(cruise))
    f = np.clip(m / np.maximum(cruise, 1e-6), *F_CLAMP)
    u = np.concatenate([fi * frame_speed(p, hz) for fi, p in zip(f, paths)])
    M = float(np.percentile(u[u >= idle_v], M_PCT))
    return {"m": m, "M": max(M, m * 1.05), "cruise": cruise.tolist(), "f": f.tolist(), "hz": hz, "idle_v": idle_v}


def speed_map(u, m, M):
    u = np.asarray(u, float)
    return np.where(u <= m, u, m + (M - m) * np.tanh((u - m) / (M - m)))


def normalize(ee, g, t, f, m, M, hz=10.0, idle_v=0.02, keep_s=0.5, tail_s=2.0):
    """Re-time one episode. ee [N, 3], g [N] (1 = closed), t [N] (source time per step); f: this episode's stage-1
    factor. Returns (ee, g, t) at hz with the same path, gripper switches and their surroundings."""
    ee, g, t = np.asarray(ee, float), np.asarray(g, np.float32), np.asarray(t, float)
    n = len(ee)
    if n < 3:
        return ee, g, t
    v = frame_speed(ee, hz)
    sw = np.flatnonzero(np.diff(g) != 0) + 1
    k = np.arange(n)
    keep = np.zeros(n, bool)
    for s in sw:
        keep |= np.abs(k - s) <= keep_s * hz
    tail = k >= n - 1 - tail_s * hz
    idle = (v < idle_v) & ~keep & ~tail
    u = f * v
    rate = np.where(u > 1e-9, f * speed_map(u, m, M) / np.maximum(u, 1e-9), f)
    rate[keep] = 1.0
    cur, c = [0.0], 0.0
    while c < n - 1:
        i = int(np.floor(c + 1e-9))
        if idle[i]:   # skip the idle stretch: next frame that is not idle
            nxt = np.flatnonzero(~idle[i + 1:])
            c = float(i + 1 + nxt[0]) if len(nxt) else float(n - 1)
        else:
            c = c + rate[i]
        cur.append(min(c, n - 1.0))
    c = np.array(cur)
    out = np.column_stack([np.interp(c, k, ee[:, j]) for j in range(3)])
    gi = g[np.floor(c + 1e-9).astype(int)]
    return out, gi, np.interp(c, k, t)
