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
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import (build_policy, image_spec, list_checkpoints, load_run_config, load_trainable,  # noqa: E402
                        pick_device, training_side_info)

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
    p.add_argument("--watch", action="store_true",
                   help="keep polling the run dir and probe each new checkpoint as training saves it")
    p.add_argument("--until", type=int, default=None, help="with --watch: exit after probing this step")
    p.add_argument("--log", default=None, help="append each result line to this file")
    a = p.parse_args()
    dev = pick_device(a.device)
    while a.watch and not os.path.exists(os.path.join(a.run, "run.json")):
        time.sleep(30)                         # training hasn't started writing the run yet
    rc = load_run_config(a.run)
    env = PickPlaceEnv(env_config(training_side_info(rc).get("env_cfg")))
    policy, pre, post, _ = build_policy(rc["features"], rc["stats"], dev)
    cams, size = image_spec(rc["features"])
    policy.eval()

    def todo(done):
        cks = dict(list_checkpoints(a.run))
        if a.watch or a.step == "all":
            return {s: d for s, d in cks.items() if s not in done}
        st = max(cks) if a.step == "latest" else int(a.step)
        return {st: cks[st]}

    done = set()
    if a.log and os.path.exists(a.log):       # resuming after a disconnect: skip steps already probed
        import re
        done = {int(m.group(1)) for m in re.finditer(r"^step\s+(\d+):", open(a.log).read(), re.M)}
    while True:
        new = todo(done)
        for st in sorted(new):
            probe_one(a, policy, pre, post, env, cams, size, st, new[st])
            done.add(st)
            if a.until is not None and st >= a.until:
                return
        if not a.watch:
            break
        time.sleep(60)


def probe_one(a, policy, pre, post, env, cams, size, st, ck_dir):
    torch.manual_seed(0)
    load_trainable(policy, ck_dir)
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
    line = (f"step {st:6d}: grounding accuracy {np.mean(hits):.2f} (chance 0.33), "
            f"median distance to the named cube {np.median(dists) * 100:.1f} cm, "
            f"to the nearest cube of any color {np.median(nearest) * 100:.1f} cm "
            f"(per single sample {np.median(nearest_each) * 100:.1f} cm, <2 cm in "
            f"{np.mean(np.array(nearest_each) < 0.02):.0%})  (n={len(hits)})")
    print(line, flush=True)
    if a.log:
        with open(a.log, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
