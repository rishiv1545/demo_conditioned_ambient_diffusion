"""Episode storage, the single episode loader, normalization, chunked dataset and context sampler.

Episode format (one .npz per episode):
    obs [T, OBS_DIM] float32, action [T, 4] float32, task int[2] (cube, zone), source "sim"|"human",
    layout [6, 2] (cube xy rows 0-2, zone xy rows 3-5), success bool, sigma_n float (0 for sim, NaN for human
    until Phase 2), meta (JSON string), and for human data raw_traj [N, 3] and raw_gripper [N].
"""
import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------- episode I/O
def save_episode(path, obs, action, task, source, layout, success, sigma_n=None, meta=None, **extra):
    if sigma_n is None:
        sigma_n = 0.0 if source == "sim" else float("nan")
    lay = np.concatenate([np.asarray(layout["cubes"]), np.asarray(layout["zones"])]) if isinstance(layout, dict) else layout
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path, obs=np.asarray(obs, np.float32), action=np.asarray(action, np.float32), task=np.asarray(task, np.int64),
        source=np.array(source), layout=np.asarray(lay, np.float64), success=np.array(bool(success)),
        sigma_n=np.array(float(sigma_n)), meta=np.array(json.dumps(meta or {})),
        **{k: np.asarray(v) for k, v in extra.items()})


def layout_from_array(lay):
    lay = np.asarray(lay)
    return {"cubes": lay[:3].copy(), "zones": lay[3:6].copy()}


def _read(path):
    with np.load(path, allow_pickle=False) as f:
        ep = {k: f[k] for k in f.files}
    ep["task"] = tuple(int(v) for v in ep["task"])
    ep["source"] = str(ep["source"])
    ep["success"] = bool(ep["success"])
    ep["sigma_n"] = float(ep["sigma_n"])
    ep["meta"] = json.loads(str(ep["meta"]))
    ep["path"] = path
    return ep


def load_episodes(paths, filters=None):
    """Load episodes from files, directories or globs. filters (all optional):
    sources: list of "sim"/"human"; tasks: list of (cube, zone); include_failed: bool (default True for human,
    sim episodes are always successful); max_per_task: dict source -> int (keeps the first N per task)."""
    filters = filters or {}
    files = []
    for p in ([paths] if isinstance(paths, str) else paths):
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True))
        elif any(ch in p for ch in "*?["):
            files += sorted(glob.glob(p, recursive=True))
        else:
            files.append(p)
    sources = filters.get("sources")
    tasks = {tuple(t) for t in filters["tasks"]} if filters.get("tasks") is not None else None
    include_failed = filters.get("include_failed", True)
    max_per_task = filters.get("max_per_task") or {}
    eps, counts = [], {}
    for f in files:
        ep = _read(f)
        if sources and ep["source"] not in sources:
            continue
        if tasks is not None and ep["task"] not in tasks:
            continue
        if not include_failed and not ep["success"]:
            continue
        key = (ep["source"], ep["task"])
        if ep["source"] in max_per_task and counts.get(key, 0) >= max_per_task[ep["source"]]:
            continue
        counts[key] = counts.get(key, 0) + 1
        eps.append(ep)
    return eps


# ---------------------------------------------------------------------------- normalization
@dataclass
class Normalizer:
    """Affine map of each dimension to [-1, 1] from training-set min/max."""
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, x, eps=1e-4):
        lo, hi = x.min(0), x.max(0)
        span = np.maximum(hi - lo, eps)
        mid = (hi + lo) / 2
        return cls(lo=(mid - span / 2).astype(np.float32), hi=(mid + span / 2).astype(np.float32))

    def norm(self, x):
        return 2 * (x - self.lo) / (self.hi - self.lo) - 1

    def denorm(self, x):
        return (x + 1) / 2 * (self.hi - self.lo) + self.lo

    def state_dict(self):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(lo=np.asarray(d["lo"], np.float32), hi=np.asarray(d["hi"], np.float32))


# ---------------------------------------------------------------------------- chunked dataset
@dataclass
class DataConfig:
    horizon: int = 16       # action chunk length
    n_obs: int = 1          # number of stacked observation frames (1 or 2)


class ChunkDataset:
    """Every timestep of every episode is a sample: (obs history, task, action chunk, sigma_n, episode id).
    Chunks running past the episode end are padded with the last action (the policy learns to hold)."""

    def __init__(self, episodes, cfg: DataConfig = DataConfig(), obs_norm=None, act_norm=None):
        import torch  # noqa: F401  (keeps numpy-only users of this module light)
        self.eps, self.cfg = episodes, cfg
        all_obs = np.concatenate([e["obs"] for e in episodes])
        all_act = np.concatenate([e["action"] for e in episodes])
        self.obs_norm = obs_norm or Normalizer.fit(all_obs)
        self.act_norm = act_norm or Normalizer.fit(all_act)
        self.index = [(i, t) for i, e in enumerate(episodes) for t in range(len(e["action"]))]
        self.n_obs_ep = [self.obs_norm.norm(e["obs"]).astype(np.float32) for e in episodes]
        self.n_act_ep = [self.act_norm.norm(e["action"]).astype(np.float32) for e in episodes]
        self.by_task = {}
        for i, e in enumerate(episodes):
            self.by_task.setdefault(e["task"], []).append(i)

    def __len__(self):
        return len(self.index)

    def get(self, k):
        from sim.env import task_onehot
        i, t = self.index[k]
        e, O, A = self.eps[i], self.n_obs_ep[i], self.n_act_ep[i]
        T = len(A)
        oi = np.clip(np.arange(t - self.cfg.n_obs + 1, t + 1), 0, T - 1)
        ai = np.clip(np.arange(t, t + self.cfg.horizon), 0, T - 1)
        return {"obs": O[oi].reshape(-1), "task": task_onehot(e["task"]), "action": A[ai],
                "sigma_n": np.float32(e["sigma_n"]), "ep": np.int64(i)}

    def batch(self, ks):
        items = [self.get(k) for k in ks]
        return {key: np.stack([it[key] for it in items]) for key in items[0]}

    def arrays(self):
        """All samples stacked as numpy arrays (the datasets here are small enough to live on the GPU)."""
        return self.batch(range(len(self)))

    # ------------------------------------------------------------ Phase 2 hook: context sampling
    def sample_context(self, k, rng, source=None, n_keyframes=16):
        """For sample k, return another episode of the same task (never the sample's own episode), optionally
        restricted to `source`, as a subsampled trajectory of n_keyframes normalized (obs, action) rows.
        Returns None if no such episode exists."""
        i, _ = self.index[k]
        task = self.eps[i]["task"]
        cands = [j for j in self.by_task[task] if j != i and (source is None or self.eps[j]["source"] == source)]
        if not cands:
            return None
        j = cands[rng.integers(len(cands))]
        T = len(self.n_act_ep[j])
        idx = np.round(np.linspace(0, T - 1, n_keyframes)).astype(int)
        return {"obs": self.n_obs_ep[j][idx], "action": self.n_act_ep[j][idx], "ep": j}


def norm_state(ds):
    return {"obs": ds.obs_norm.state_dict(), "act": ds.act_norm.state_dict()}
