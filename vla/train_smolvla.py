"""Post-train SmolVLA (lerobot/smolvla_base) on our LeRobot datasets, optionally with ambient flow matching.

    # V0: sim only
    python vla/train_smolvla.py --run_name V0 --data data/lerobot/sim_seen --steps 3000
    # V1: + phone, naive;  V2: + phone, ambient
    python vla/train_smolvla.py --run_name V1 --data data/lerobot/sim_seen data/lerobot/phone --steps 3000
    python vla/train_smolvla.py --run_name V2 --data data/lerobot/sim_seen data/lerobot/phone --ambient_t_min 0.5

Checkpoints (trainable params + optimizer state) go to <out>/<run_name>/step_XXXXXX every --save_every steps;
--resume continues from the latest one. Point --out at Google Drive on Colab.
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import (base_config, build_policy, list_checkpoints, load_trainable, open_datasets, pick_device,  # noqa: E402
                        save_checkpoint, save_run_config)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vla.ambient import ambient_time  # noqa: E402
from vla.feature_cache import PREFIX, CachedChunkDataset, build_cache, has_cache, use_cached_features  # noqa: E402


def lr_lambda(step, warmup, total, peak, floor):
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return (floor + (peak - floor) * 0.5 * (1 + np.cos(np.pi * p))) / peak


def infinite(loader):
    while True:
        yield from loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--data", nargs="+", required=True, help="LeRobot dataset roots (exported by vla/export_lerobot.py)")
    p.add_argument("--out", default="checkpoints/vla")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16, help="micro-batch (16 fits an 18 GB Mac; 32 thrashes)")
    p.add_argument("--grad_accum", type=int, default=2, help="effective batch = batch * grad_accum")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_floor", type=float, default=2.5e-6)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--ambient_t_min", type=float, default=0.0, help="0 = off (V0/V1); >0 restricts sigma_n>0 samples")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", help="auto (bf16 where supported, else fp32) | keep | fp32")
    p.add_argument("--features", default="cache", choices=["cache", "raw"],
                   help="cache: train from precomputed frozen vision features (built on first use); raw: images")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_minutes", type=float, default=0, help="stop (after saving) after this long; 0 = no limit")
    a = p.parse_args()
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run_dir = os.path.join(a.out, a.run_name)
    os.makedirs(run_dir, exist_ok=True)

    chunk = base_config().chunk_size
    dss, stats = open_datasets(a.data, chunk)
    features = dss[0].meta.features
    policy, pre, _, cfg = build_policy(features, stats, dev, dtype=a.dtype)
    img_keys = list(cfg.image_features)
    if a.features == "cache":
        for root in a.data:
            if not has_cache(root, img_keys):
                t_c = time.time()
                build_cache(root, policy, dev)
                print(f"built vision cache for {root} in {time.time() - t_c:.0f}s", flush=True)
        dss = [CachedChunkDataset(root, chunk, img_keys) for root in a.data]
        policy = use_cached_features(policy)
    save_run_config(run_dir, vars(a), {k: dict(v) for k, v in features.items()}, stats)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in policy.parameters())
    print(f"device {dev}; {sum(len(d) for d in dss)} frames from {len(dss)} datasets; "
          f"params {n_all / 1e6:.0f}M total, {n_train / 1e6:.1f}M trainable")

    loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(dss), batch_size=a.batch, shuffle=True,
                                         num_workers=a.workers, drop_last=True,
                                         persistent_workers=a.workers > 0, pin_memory=dev == "cuda")
    params = [q for q in policy.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=cfg.optimizer_betas, eps=cfg.optimizer_eps,
                            weight_decay=cfg.optimizer_weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, a.warmup, a.steps, a.lr, a.lr_floor))
    start = 0
    if a.resume and list_checkpoints(run_dir):
        step0, d = list_checkpoints(run_dir)[-1]
        load_trainable(policy, d)
        st = torch.load(os.path.join(d, "train_state.pt"), map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng_torch"])
        np.random.set_state(st["rng_np"])
        start = st["step"]
        print(f"resumed from {d} (step {start})")

    log_path = os.path.join(run_dir, "loss.csv")
    new_log = not (a.resume and os.path.exists(log_path))
    log = open(log_path, "w" if new_log else "a", newline="")
    w = csv.writer(log)
    if new_log:
        w.writerow(["step", "loss", "loss_clean", "loss_phone", "lr", "sec_per_step", "elapsed_s"])
    policy.train()
    gen = torch.Generator().manual_seed(a.seed + start)
    it = infinite(loader)
    t0 = t_last = time.time()
    acc = {"loss": [], "clean": [], "phone": []}
    for step in range(start + 1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        for _ in range(a.grad_accum):
            raw = next(it)
            sigma = raw["sigma_n"].reshape(-1).float()
            batch = pre(raw)
            # LeRobot reads "actions_id_pad" (sic) but datasets provide "action_is_pad": without this, padded
            # chunk tails (copies of the last action) would count in the loss.
            batch["actions_id_pad"] = batch["action_is_pad"].to(dev)
            for k in img_keys:
                if PREFIX + k in raw:   # the preprocessor drops unknown keys; cached features go in directly
                    batch[PREFIX + k] = raw[PREFIX + k].to(dev)
            t = ambient_time(sigma, a.ambient_t_min, gen).to(dev)   # ambient flow matching (off when t_min=0)
            per, _ = policy.forward(batch, time=t, reduction="none")
            loss = per.mean()
            (loss / a.grad_accum).backward()
            per = per.detach().float().cpu()
            acc["loss"].append(loss.item())
            if (sigma == 0).any():
                acc["clean"].append(per[sigma == 0].mean().item())
            if (sigma > 0).any():
                acc["phone"].append(per[sigma > 0].mean().item())
        torch.nn.utils.clip_grad_norm_(params, cfg.optimizer_grad_clip_norm)
        opt.step()
        sched.step()
        if step % a.log_every == 0:
            now = time.time()
            sps = (now - t_last) / a.log_every
            t_last = now
            m = {k: (np.mean(v) if v else float("nan")) for k, v in acc.items()}
            acc = {k: [] for k in acc}
            w.writerow([step, f"{m['loss']:.5f}", f"{m['clean']:.5f}", f"{m['phone']:.5f}",
                        f"{sched.get_last_lr()[0]:.2e}", f"{sps:.3f}", f"{now - t0:.0f}"])
            log.flush()
            print(f"step {step:6d}  loss {m['loss']:.4f}  {sps:.2f}s/step  {(now - t0) / 60:.1f} min", flush=True)
        out_of_time = a.max_minutes and (time.time() - t0) / 60 > a.max_minutes
        if step % a.save_every == 0 or step == a.steps or out_of_time:
            d = save_checkpoint(run_dir, step, policy, opt, sched)
            print("saved", d, flush=True)
        if out_of_time:
            print(f"stopping after {a.max_minutes} min (resume with --resume)")
            break
    print(f"done: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
