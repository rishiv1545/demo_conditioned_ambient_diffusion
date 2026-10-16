"""Episode storage and loading, evaluation seeds and the Wilson interval.

Episode format (one .npz per episode):
    obs [T, OBS_DIM] float32, action [T, 4] float32, task int[2] (cube, zone), source "sim"|"human",
    layout [7, 2] in the single-object scene (cube xy rows 0-2, zone xy rows 3-6; absent ones parked far away),
    success bool, sigma_n float (0 for sim, NaN for human), meta (JSON string), and for human data raw_traj [N, 3]
    and raw_gripper [N].
"""
import glob
import json
import os

import numpy as np

EVAL_SEED_BASE = 1_000_000   # evaluation layouts: seed base + task index * 10_000 + k (data seeds stay below)


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
    return {"cubes": lay[:3].copy(), "zones": lay[3:].copy()}   # 3 zones, or 4 in the single-object scene


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


def wilson(k, n, z=1.96):
    """(p, lo, hi): success rate and its Wilson 95% interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)
