"""Fast language-grounding probe for a SmolVLA checkpoint (about a minute; no rollouts).

For several eval layouts, keep the image and state fixed and swap the cube named in the instruction. A grounded
policy's predicted grasp point (the lowest point of the predicted chunk) should move to the named cube.

    python vla/probe_grounding.py --run checkpoints/vla/V0 --step latest

Reports, per checkpoint: grounding accuracy (the predicted grasp point is nearest the named cube; chance 1/3), the
median distance to the named cube, and the median distance to the nearest cube of ANY color. If the last one is
also large, the problem is localization (image resolution/camera), not only language grounding.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import (build_policy, image_spec, list_checkpoints, load_run_config, load_trainable,  # noqa: E402
                        pick_device)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from policy.evaluate import EVAL_SEED_BASE  # noqa: E402
from sim.env import CUBES, PickPlaceEnv, env_config, instruction  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", default="latest", help="a step, 'latest', or 'all'")
    p.add_argument("--layouts", type=int, default=8)
    p.add_argument("--samples", type=int, default=3, help="flow samples averaged per prediction")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    dev = pick_device(a.device)
    rc = load_run_config(a.run)
    side = os.path.join(rc["args"]["data"][0], "episodes.json")
    env = PickPlaceEnv(env_config(json.load(open(side)).get("env_cfg") if os.path.exists(side) else None))
    policy, pre, post, _ = build_policy(rc["features"], rc["stats"], dev)
    cams, size = image_spec(rc["features"])
    policy.eval()
    cks = dict(list_checkpoints(a.run))
    steps = sorted(cks) if a.step == "all" else [max(cks) if a.step == "latest" else int(a.step)]
    torch.manual_seed(0)
    for st in steps:
        load_trainable(policy, cks[st])
        hits, dists, nearest, nearest_each = [], [], [], []
        for k in range(a.layouts):
            env.reset((0, 0), seed=EVAL_SEED_BASE + 777_000 + k)
            ims = {f"observation.images.{c}": torch.from_numpy(im).permute(2, 0, 1).float()[None] / 255
                   for c, im in env.images(size, cams).items()}
            state = torch.from_numpy(env.state())[None]
            cubes = env.layout["cubes"]
            for c in range(3):
                pts = []
                for _ in range(a.samples):
                    b = {**ims, "observation.state": state, "task": [instruction((c, 0))]}
                    with torch.no_grad():
                        ch = post(policy.predict_action_chunk(pre(b)))[0].cpu().numpy()
                    pts.append(ch[np.argmin(ch[:, 2]), :2])
                    nearest_each.append(np.linalg.norm(cubes - pts[-1], axis=1).min())  # per sample: no averaging
                g = np.mean(pts, 0)
                d = np.linalg.norm(cubes - g, axis=1)
                hits.append(int(np.argmin(d) == c))
                dists.append(d[c])
                nearest.append(d.min())
        print(f"step {st:6d}: grounding accuracy {np.mean(hits):.2f} (chance 0.33), "
              f"median distance to the named cube {np.median(dists) * 100:.1f} cm, "
              f"to the nearest cube of any color {np.median(nearest) * 100:.1f} cm "
              f"(per single sample {np.median(nearest_each) * 100:.1f} cm, <2 cm in "
              f"{np.mean(np.array(nearest_each) < 0.02):.0%})  (n={len(hits)})", flush=True)


if __name__ == "__main__":
    main()
