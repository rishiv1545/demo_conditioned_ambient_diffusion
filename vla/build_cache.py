"""Build the frozen-vision-feature cache for LeRobot datasets (see vla/feature_cache.py).

    python vla/build_cache.py data/lerobot/sim_seen_v2c_100 [--device cuda]

The cache is specific to the backbone dtype of the machine that builds it (bf16 on A100/MPS, fp32 on T4), so build
it on the machine that trains. Valid both for frozen-VLM training and for --unfreeze_vlm (SigLIP and the connector
stay frozen in both).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import base_config, build_policy, open_datasets, pick_device  # noqa: E402
from vla.feature_cache import build_cache, has_cache  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    a = p.parse_args()
    dev = pick_device(a.device)
    dss, stats = open_datasets(a.roots, base_config().chunk_size)
    policy, _, _, cfg = build_policy(dss[0].meta.features, stats, dev)
    for root in a.roots:
        if has_cache(root, list(cfg.image_features)):
            print(f"{root}: cache already present")
            continue
        t0 = time.time()
        build_cache(root, policy, dev, batch=a.batch, workers=a.workers)
        print(f"{root}: built vision cache in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
