"""Fast language-grounding probe for a SmolVLA checkpoint (about a minute; no rollouts).

For several eval layouts, keep the image and state fixed and swap the cube named in the instruction. A grounded
policy's predicted grasp point (the lowest point of the predicted chunk) should move to the named cube.

    python vla/probe_grounding.py --run checkpoints/vla/V0 --step latest

Reports, per checkpoint: grounding accuracy (the predicted grasp point is nearest the named cube) and the median
distance to the named cube. Chance accuracy is 1/3.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import build_policy, list_checkpoints, load_run_config, load_trainable, pick_device  # noqa: E402

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
    policy.eval()
    cks = dict(list_checkpoints(a.run))
    steps = sorted(cks) if a.step == "all" else [max(cks) if a.step == "latest" else int(a.step)]
    torch.manual_seed(0)
    for st in steps:
        load_trainable(policy, cks[st])
        hits, dists = [], []
        for k in range(a.layouts):
            env.reset((0, 0), seed=EVAL_SEED_BASE + 777_000 + k)
            img = torch.from_numpy(env.images()["phone"]).permute(2, 0, 1).float()[None] / 255
            state = torch.from_numpy(env.state())[None]
            cubes = env.layout["cubes"]
            for c in range(3):
                pts = []
                for _ in range(a.samples):
                    b = {"observation.images.phone": img, "observation.state": state, "task": [instruction((c, 0))]}
                    with torch.no_grad():
                        ch = post(policy.predict_action_chunk(pre(b)))[0].cpu().numpy()
                    pts.append(ch[np.argmin(ch[:, 2]), :2])
                g = np.mean(pts, 0)
                d = np.linalg.norm(cubes - g, axis=1)
                hits.append(int(np.argmin(d) == c))
                dists.append(d[c])
        print(f"step {st:6d}: grounding accuracy {np.mean(hits):.2f} (chance 0.33), "
              f"median distance to the named cube {np.median(dists) * 100:.1f} cm  (n={len(hits)})", flush=True)


if __name__ == "__main__":
    main()
