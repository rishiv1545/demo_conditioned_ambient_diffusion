"""Why phone replays fail: re-run each processed episode's actions in sim and classify the outcome.

    python scripts/replay_diagnose.py data/human/3

Per clip: grasp offset in sim (EE - target cube when the gripper closes), cube height at the grasp, whether the cube
was lifted (> 3.5 cm), the max height, whether it was still held at the end, distance of the cube to the target
zone at the end, and a failure class: no_lift (grasp missed / slipped), dropped (lifted, not in the zone),
missed_zone (released outside), ok.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from policy.data import load_episodes, layout_from_array  # noqa: E402
from sim.env import PickPlaceEnv, env_config, task_name  # noqa: E402


def diagnose(ep):
    env = PickPlaceEnv(env_config(ep["meta"].get("env_cfg")))
    c, z = ep["task"]
    env.reset(task=ep["task"], layout=layout_from_array(ep["layout"]))
    zone = env.layout["zones"][z]
    act = ep["action"]
    grasp, max_z, lift_t, ok = None, 0.0, None, False
    for t, a in enumerate(act):
        if grasp is None and t > 0 and a[3] > 0.5 and act[t - 1][3] < 0.5 and (act[:t, 3] < 0.5).any():
            grasp = {"t": t, "off": (env.ee_pos() - env.cube_pos(c)), "ee_z": env.ee_pos()[2]}
        _, ok = env.step(a)
        cz = env.cube_pos(c)[2]
        if cz > 0.035 and lift_t is None:
            lift_t = t
        max_z = max(max_z, cz)
    p = env.cube_pos(c)
    dz = float(np.abs(p[:2] - zone).max())
    lifted = max_z > 0.035
    cls = "ok" if ok else ("no_lift" if not lifted else ("missed_zone" if dz < 0.10 else "dropped"))
    return {"cls": cls, "grasp": grasp, "max_z": max_z, "zone_dist": dz, "success": ok}


def main():
    eps = load_episodes(sys.argv[1])
    counts = {}
    for ep in sorted(eps, key=lambda e: e["path"]):
        d = diagnose(ep)
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
        g = d["grasp"]
        gs = (f"grasp off xy {g['off'][0] * 100:+5.1f} {g['off'][1] * 100:+5.1f} z {g['off'][2] * 100:+5.1f} cm "
              f"(ee z {g['ee_z'] * 100:4.1f})") if g else "no grasp"
        print(f"{os.path.basename(ep['path']):24s} {task_name(ep['task']):13s} {d['cls']:11s} {gs}  "
              f"max cube z {d['max_z'] * 100:4.1f} cm  end zone dist {d['zone_dist'] * 100:4.1f} cm")
    print(counts)


if __name__ == "__main__":
    main()
