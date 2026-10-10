"""Raw-image vs cached-feature inference for a trained SmolVLA checkpoint (train/eval consistency check).

Training reads cached vision features (vla/feature_cache.py, fp16 storage); eval and the probe run the vision encoder
on raw images. On N training frames, predict action chunks both ways with the same flow-matching noise, in the
device/dtype setup used for training and eval (build_policy defaults), and compare. For scale, also compare two raw
predictions with different noise (the policy's own sampling spread) and the chunks against the dataset actions.

    python vla/check_cache.py --run <ckpt dir> --step 30000 --n 200 --device cuda
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import (base_config, build_policy, list_checkpoints, load_run_config, load_trainable,  # noqa: E402
                        open_datasets, parse_data_spec, pick_device)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vla.feature_cache import PREFIX, CachedChunkDataset, use_cached_features  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", default="latest")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="write the result JSON here")
    a = p.parse_args()
    dev = pick_device(a.device)
    rc = load_run_config(a.run)
    root = parse_data_spec(rc["args"]["data"][0])[0]
    cks = dict(list_checkpoints(a.run))
    step = max(cks) if a.step == "latest" else int(a.step)
    chunk = base_config().chunk_size
    (raw_ds,), _ = open_datasets([root], chunk)
    pols = []
    for cached in (False, True):
        pol, pre, post, cfg = build_policy(rc["features"], rc["stats"], dev)   # same call as eval/probe/train
        load_trainable(pol, cks[step])
        pol.eval()
        pols.append(use_cached_features(pol) if cached else pol)
    img_keys = list(cfg.image_features)
    cds = CachedChunkDataset(root, chunk, img_keys)
    idx = np.random.default_rng(0).choice(len(cds), a.n, replace=False)
    print(f"{a.run} step {step} on {dev}; {a.n} frames of {root}; vision dtype "
          f"{pols[0].model.vlm_with_expert.get_vlm_model().vision_model.dtype}", flush=True)
    d_cache, d_noise, err_raw, err_cache, gt_scale = [], [], [], [], []
    g = torch.Generator(device="cpu")
    for s in range(0, a.n, a.batch):
        ids = idx[s:s + a.batch]
        rb = torch.utils.data.default_collate([raw_ds[int(i)] for i in ids])
        cb = torch.utils.data.default_collate([cds[int(i)] for i in ids])
        assert torch.allclose(rb["observation.state"], cb["observation.state"]) and rb["task"] == cb["task"]
        g.manual_seed(1000 + s)
        noise = torch.randn((len(ids), chunk, cfg.max_action_dim), generator=g).to(dev)
        noise2 = torch.randn((len(ids), chunk, cfg.max_action_dim), generator=g).to(dev)
        with torch.no_grad():
            raw_in = pre({k: v for k, v in rb.items() if k != "action"})
            a_raw = post(pols[0].predict_action_chunk(raw_in, noise=noise)).float().cpu()
            pols[0].reset()
            a_raw2 = post(pols[0].predict_action_chunk(raw_in, noise=noise2)).float().cpu()
            pols[0].reset()
            cin = pre({k: v for k, v in cb.items() if k != "action" and not k.startswith(PREFIX)})
            for k in img_keys:
                cin[PREFIX + k] = cb[PREFIX + k].to(dev)
            a_c = post(pols[1].predict_action_chunk(cin, noise=noise)).float().cpu()
            pols[1].reset()
        gt = rb["action"].float()
        d_cache.append((a_raw - a_c).abs().mean((1, 2)))
        d_noise.append((a_raw - a_raw2).abs().mean((1, 2)))
        err_raw.append((a_raw - gt).abs().mean((1, 2)))
        err_cache.append((a_c - gt).abs().mean((1, 2)))
        gt_scale.append(gt.std((1,)).mean(1))
    cat = lambda x: torch.cat(x).numpy()
    res = {"run": a.run, "step": step, "n": a.n, "device": dev,
           "mean_abs_diff_raw_vs_cached": float(cat(d_cache).mean()),
           "max_frame_diff_raw_vs_cached": float(cat(d_cache).max()),
           "mean_abs_diff_raw_vs_raw_other_noise": float(cat(d_noise).mean()),
           "mean_abs_err_vs_dataset_raw": float(cat(err_raw).mean()),
           "mean_abs_err_vs_dataset_cached": float(cat(err_cache).mean()),
           "dataset_action_std_within_chunk": float(cat(gt_scale).mean())}
    for k, v in res.items():
        print(f"{k:40s} {v}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
