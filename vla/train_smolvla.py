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

from vla.common import (base_config, build_policy, frames_of_episodes, list_checkpoints, load_trainable, open_datasets,
                        load_run_config, parse_data_spec, pick_device, selected_episodes,  # noqa: E402
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


def frame_tasks(ds):
    """Task (instruction) of every frame of a CachedChunkDataset or a Subset of one."""
    if isinstance(ds, torch.utils.data.Subset):
        return [ds.dataset.task[i] for i in ds.indices]
    return list(ds.task)


def make_sampler(dss, balance, mix, features):
    """None (uniform over all frames) or a WeightedRandomSampler: --balance tasks gives every task the same share,
    --mix gives every dataset a fixed share."""
    if balance == "none" and not mix:
        return None
    assert features == "cache", "--balance/--mix need --features cache"
    assert not (balance != "none" and mix), "--balance and --mix are exclusive"
    if mix:
        frac = [float(x) for x in mix.split(",")]
        assert len(frac) == len(dss), f"--mix needs {len(dss)} fractions"
        w = np.concatenate([np.full(len(d), f / len(d)) for d, f in zip(dss, frac)])
        print("sampling mix:", ", ".join(f"{f:.2f} ({len(d)} frames)" for d, f in zip(dss, frac)), flush=True)
    else:
        tasks = sum((frame_tasks(d) for d in dss), [])
        names, inv, counts = np.unique(tasks, return_inverse=True, return_counts=True)
        w = 1.0 / counts[inv]
        print("task-balanced sampling:", ", ".join(f"{n[:40]!r} {c}" for n, c in zip(names, counts)), flush=True)
    return torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(w),
                                                  replacement=True)


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
    p.add_argument("--lr_schedule", default="cosine", choices=["cosine", "constant", "cosine_from_resume"],
                   help="constant: hold lr after warmup (no decay); cosine_from_resume: hold lr until the resume step, "
                        "then cosine-decay from it to --lr_floor at --steps (e.g. finishing a constant-lr run)")
    p.add_argument("--resume_warmup", type=int, default=0,
                   help="cosine_from_resume: ramp the lr linearly over this many steps after the resume step")
    p.add_argument("--ambient_t_min", type=float, default=0.0, help="0 = off (V0/V1); >0 restricts sigma_n>0 samples")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--balance", default="none", choices=["none", "tasks"],
                   help="tasks: sample every task (language instruction) equally often, frames uniform within a task "
                        "(scarce held-out-task data otherwise is ~1%% of the samples)")
    p.add_argument("--mix", default=None,
                   help="comma list of sample fractions per --data entry (frames uniform within a dataset), "
                        "e.g. 0.63,0.37; exclusive with --balance tasks")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", help="auto (bf16 where supported, else fp32) | keep | fp32")
    p.add_argument("--unfreeze_vlm", type=int, default=0,
                   help="1: also train the VLM language layers (SigLIP + connector stay frozen); needs a big GPU")
    p.add_argument("--vlm_lr", type=float, default=1e-5, help="learning rate for the VLM layers with --unfreeze_vlm 1")
    p.add_argument("--features", default="cache", choices=["cache", "raw"],
                   help="cache: train from precomputed frozen vision features (built on first use); raw: images")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stats_from", default=None,
                   help="run dir whose normalization stats to reuse (continuing a run on a different data mix)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_minutes", type=float, default=0, help="stop (after saving) after this long; 0 = no limit")
    a = p.parse_args()
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run_dir = os.path.join(a.out, a.run_name)
    os.makedirs(run_dir, exist_ok=True)

    chunk = base_config().chunk_size
    specs = [parse_data_spec(d) for d in a.data]      # "root?per_task=4&cube=blue" selects episodes of a dataset
    roots = [r for r, _ in specs]
    dss, stats = open_datasets(roots, chunk)          # normalization stats from the full datasets
    if a.stats_from:   # keep the continued model's input/output normalization
        stats = load_run_config(a.stats_from)["stats"]
        print(f"normalization stats from {a.stats_from}", flush=True)
    features = dss[0].meta.features
    policy, pre, _, cfg = build_policy(features, stats, dev, dtype=a.dtype, unfreeze_vlm=bool(a.unfreeze_vlm))
    img_keys = list(cfg.image_features)
    if a.features == "raw":
        assert not any("loss_mask" in d.hf_dataset.column_names for d in dss), "loss_mask needs --features cache"
    if a.features == "cache":
        for root in roots:
            if not has_cache(root, img_keys):
                t_c = time.time()
                build_cache(root, policy, dev)
                print(f"built vision cache for {root} in {time.time() - t_c:.0f}s", flush=True)
        dss = [CachedChunkDataset(root, chunk, img_keys) for root in roots]
        policy = use_cached_features(policy)
    for i, (root, filt) in enumerate(specs):
        keep = selected_episodes(root, filt)
        if keep is not None:
            ep_col = dss[i].ep if a.features == "cache" else dss[i].hf_dataset["episode_index"]
            dss[i] = torch.utils.data.Subset(dss[i], frames_of_episodes(ep_col, keep))
            print(f"{root}: {len(keep)} episodes selected by {filt}", flush=True)
    save_run_config(run_dir, vars(a), {k: dict(v) for k, v in features.items()}, stats)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in policy.parameters())
    print(f"device {dev}; {sum(len(d) for d in dss)} frames from {len(dss)} datasets; "
          f"params {n_all / 1e6:.0f}M total, {n_train / 1e6:.1f}M trainable")

    sampler = make_sampler(dss, a.balance, a.mix, a.features)
    loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(dss), batch_size=a.batch,
                                         shuffle=sampler is None, sampler=sampler,
                                         num_workers=a.workers, drop_last=True,
                                         persistent_workers=a.workers > 0, pin_memory=dev == "cuda")
    params = [q for q in policy.parameters() if q.requires_grad]
    vlm_ids = {id(q) for q in policy.model.vlm_with_expert.vlm.parameters()}
    groups = [{"params": [q for q in params if id(q) not in vlm_ids], "lr": a.lr}]
    if a.unfreeze_vlm:
        groups.append({"params": [q for q in params if id(q) in vlm_ids], "lr": a.vlm_lr})
        print(f"unfrozen VLM: {sum(q.numel() for q in groups[1]['params']) / 1e6:.0f}M params at lr {a.vlm_lr}")
    opt = torch.optim.AdamW(groups, betas=cfg.optimizer_betas, eps=cfg.optimizer_eps,
                            weight_decay=cfg.optimizer_weight_decay)
    if a.lr_schedule == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    elif a.lr_schedule == "cosine_from_resume":
        decay_from = {"step": 0}   # set to the resume step below (the lambda reads it at call time)

        def from_resume(s):
            if s < decay_from["step"]:
                return min(1.0, (s + 1) / a.warmup)
            q = min(1.0, (s - decay_from["step"]) / max(1, a.steps - decay_from["step"]))
            ramp = min(1.0, (s - decay_from["step"] + 1) / a.resume_warmup) if a.resume_warmup else 1.0
            return ramp * (a.lr_floor + (a.lr - a.lr_floor) * 0.5 * (1 + np.cos(np.pi * q))) / a.lr
        sched = torch.optim.lr_scheduler.LambdaLR(opt, from_resume)
    else:
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
    if a.lr_schedule == "cosine_from_resume":
        decay_from["step"] = start
        sched.last_epoch = start          # LambdaLR's step counter = optimizer steps so far
        # the resumed optimizer/scheduler carry the source run's base lrs: use this run's --lr / --vlm_lr
        sched.base_lrs = [a.lr] + ([a.vlm_lr] if a.unfreeze_vlm else [])
        for gr, base in zip(opt.param_groups, sched.base_lrs):
            gr["initial_lr"] = base
        for gr, base in zip(opt.param_groups, sched.base_lrs):
            gr["lr"] = base * from_resume(start)
        print(f"lr: cosine from step {start} ({a.lr:g}, vlm {a.vlm_lr:g}, warmup {a.resume_warmup}) to {a.steps} "
              f"({a.lr_floor:g})", flush=True)

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
        stop = os.path.exists(os.path.join(run_dir, "STOP"))   # early stopping requested (colab/job.py)
        if step % a.save_every == 0 or step == a.steps or out_of_time or stop:
            d = save_checkpoint(run_dir, step, policy, opt, sched)
            print("saved", d, flush=True)
        if out_of_time:
            print(f"stopping after {a.max_minutes} min (resume with --resume)")
            break
        if stop:
            print(f"early stop at step {step} (STOP file)", flush=True)
            break
    print(f"done: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
