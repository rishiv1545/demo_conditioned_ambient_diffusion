"""Train a task-conditioned diffusion policy.

    python policy/train.py --run_name A --data data/sim --sources sim --tasks seen
    python policy/train.py --run_name B --data data/sim data/human --tasks seen+human
"""
import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.data import ChunkDataset, DataConfig, load_episodes, norm_state  # noqa: E402
from policy.diffusion import Diffusion  # noqa: E402
from policy.model import ObsTaskEncoder, Policy  # noqa: E402
from sim.env import ALL_TASKS, DEFAULT_HELDOUT, OBS_DIM, parse_task, task_name  # noqa: E402


@dataclass
class TrainConfig:
    steps: int = 20000
    batch: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    ema: float = 0.999
    warmup: int = 500
    T: int = 100
    horizon: int = 16
    n_obs: int = 2
    dims: tuple = (64, 128, 256)
    seed: int = 0


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sim_task_list(spec, heldout):
    if spec == "seen":
        return [t for t in ALL_TASKS if t not in heldout]
    if spec == "all":
        return list(ALL_TASKS)
    return [parse_task(s) for s in spec.split(",")]


def build_policy(cfg: TrainConfig):
    enc = ObsTaskEncoder(OBS_DIM * cfg.n_obs)
    return Policy(enc, act_dim=4, dims=tuple(cfg.dims))


def load_policy(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ck["cfg"])
    pol = build_policy(cfg).to(device)
    pol.load_state_dict(ck["ema"])
    pol.eval()
    return pol, cfg, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--data", nargs="+", default=["data/sim"])
    p.add_argument("--sources", default="sim", help="comma list of sources: sim,human")
    p.add_argument("--sim_tasks", default="seen", help="tasks to use sim demos for: seen | all | comma list")
    p.add_argument("--human_tasks", default="all", help="tasks to use human demos for: seen | all | comma list")
    p.add_argument("--heldout", default=",".join(task_name(t) for t in DEFAULT_HELDOUT))
    p.add_argument("--include_failed", type=int, default=1, help="keep failed human replays (1) or drop them (0)")
    p.add_argument("--sim_per_task", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="checkpoints")
    for k, v in asdict(TrainConfig()).items():
        if not isinstance(v, tuple):
            p.add_argument(f"--{k}", type=type(v), default=v)
    a = p.parse_args()
    cfg = TrainConfig(**{k: getattr(a, k) for k in asdict(TrainConfig()) if hasattr(a, k)})
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = pick_device(a.device)

    heldout = [parse_task(s) for s in a.heldout.split(",")]
    sources = a.sources.split(",")
    eps = []
    if "sim" in sources:
        eps += load_episodes(a.data, {"sources": ["sim"], "tasks": sim_task_list(a.sim_tasks, heldout),
                                      "max_per_task": {"sim": a.sim_per_task}})
    if "human" in sources:
        eps += load_episodes(a.data, {"sources": ["human"], "tasks": sim_task_list(a.human_tasks, heldout),
                                      "include_failed": bool(a.include_failed)})
    if not eps:
        sys.exit("no episodes matched the filters")
    ds = ChunkDataset(eps, DataConfig(horizon=cfg.horizon, n_obs=cfg.n_obs))
    summary = {}
    for e in eps:
        k = f"{e['source']}:{task_name(e['task'])}"
        summary[k] = summary.get(k, 0) + 1
    print(f"{len(eps)} episodes, {len(ds)} samples on {dev}:", json.dumps(summary))

    arr = {k: torch.as_tensor(v, device=dev) for k, v in ds.arrays().items()}
    pol = build_policy(cfg).to(dev)
    ema = copy.deepcopy(pol).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    print(f"params: {sum(q.numel() for q in pol.parameters()) / 1e6:.2f}M")
    diff = Diffusion(cfg.T, dev)
    opt = torch.optim.AdamW(pol.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / cfg.steps))))

    run_dir = os.path.join(a.out, a.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "loss.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "loss", "lr", "elapsed_s"])
    N, t0, run = len(ds), time.time(), 0.0
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(0, N, (cfg.batch,), device=dev)
        batch = {k: v[idx] for k, v in arr.items()}
        cond = pol.encode(batch)
        loss = diff.loss(pol, batch["action"], cond)  # Phase 2: t_min from batch["sigma_n"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            d = min(cfg.ema, (1 + step) / (10 + step))
            for pe, pp in zip(ema.parameters(), pol.parameters()):
                pe.lerp_(pp, 1 - d)
            for be, bp in zip(ema.buffers(), pol.buffers()):
                be.copy_(bp)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            w.writerow([step, f"{run:.5f}", f"{sched.get_last_lr()[0]:.2e}", f"{time.time() - t0:.1f}"])
        if step % 1000 == 0 or step == cfg.steps:
            log.flush()
            print(f"step {step:6d}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)
    ck = {"ema": ema.state_dict(), "cfg": asdict(cfg), "norm": norm_state(ds), "args": vars(a),
          "train_episodes": summary, "train_time_s": time.time() - t0}
    torch.save(ck, os.path.join(run_dir, "ckpt.pt"))
    print("saved", os.path.join(run_dir, "ckpt.pt"))


if __name__ == "__main__":
    main()
