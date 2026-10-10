"""Export stored episodes (sim demos and phone replays) as a LeRobot dataset with rendered camera images.

Each .npz episode is re-simulated from its stored layout by replaying its stored actions, rendering the "phone"
camera at every step. We check that the replay reproduces the stored observations.

    python vla/export_lerobot.py --src data/sim --sources sim --tasks seen --per_task 50 --name sim_seen
    python vla/export_lerobot.py --src data/human --sources human --tasks all --name phone --phone_sigma 1.0

Per frame: observation.images.phone (video), observation.state = [ee xyz, gripper width], action = [ee target
xyz, gripper], sigma_n (0 for sim, --phone_sigma for phone-derived frames), task = language instruction.
A sidecar episodes.json records each episode's origin file, task, source and success.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from policy.data import layout_from_array, load_episodes  # noqa: E402
from sim.env import (ALL_TASKS, DEFAULT_SPLIT, IMAGE_CAMERAS, IMAGE_SIZE, STATE_DIM, PickPlaceEnv,  # noqa: E402
                     env_cfg_dict, env_config, instruction, resolve_heldout, task_name)

FPS = 10


def features(size=IMAGE_SIZE, cameras=IMAGE_CAMERAS, loss_mask=False):
    f = {f"observation.images.{c}": {"dtype": "video", "shape": (size, size, 3),
                                     "names": ["height", "width", "channels"]} for c in cameras}
    f["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": ["x", "y", "z", "gripper"]}
    f["action"] = {"dtype": "float32", "shape": (4,), "names": ["x", "y", "z", "gripper"]}
    f["sigma_n"] = {"dtype": "float32", "shape": (1,), "names": None}
    if loss_mask:   # "miss and correct" phone clips: 0 = no loss on this frame's action (vla/feature_cache.py)
        f["loss_mask"] = {"dtype": "float32", "shape": (1,), "names": None}
    return f


def replay_render(env, ep, size=IMAGE_SIZE):
    """Re-simulate an episode from its layout + actions. Returns (images {cam: [T,H,W,3]}, states [T,4], max obs err)."""
    ee_start = ep["raw_traj"][0] if ep["source"] == "human" and "raw_traj" in ep else ep.get("ee_start")
    obs = env.reset(task=ep["task"], layout=layout_from_array(ep["layout"]), ee_start=ee_start)
    imgs = {c: [] for c in IMAGE_CAMERAS}
    states, err = [], 0.0
    # recovery demos (scripts/gen_recovery_data.py): replay the executed (perturbed) actions; the labels written to
    # the dataset stay ep["action"] (the expert's nominal, corrective targets)
    for t, a in enumerate(ep.get("exec_action", ep["action"])):
        err = max(err, float(np.abs(obs - ep["obs"][t]).max()))
        for c, im in env.images(size).items():
            imgs[c].append(im)
        states.append(env.state())
        obs, _ = env.step(a)
    return {c: np.stack(v) for c, v in imgs.items()}, np.stack(states), err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", default=["data/sim"])
    p.add_argument("--sources", default="sim")
    p.add_argument("--tasks", default="seen", help="seen | heldout | all (split from --heldout)")
    p.add_argument("--heldout", default=DEFAULT_SPLIT)
    p.add_argument("--per_task", type=int, default=50)
    p.add_argument("--include_failed", type=int, default=1)
    p.add_argument("--phone_sigma", type=float, default=1.0,
                   help="sigma_n written for phone-derived frames (any value > 0 marks them as corrupted)")
    p.add_argument("--name", required=True)
    p.add_argument("--out", default="data/lerobot")
    p.add_argument("--size", type=int, default=IMAGE_SIZE)
    p.add_argument("--max_replay_err", type=float, default=1e-3)
    p.add_argument("--vcodec", default="h264", help="h264 decodes much faster than LeRobot's default AV1")
    a = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    heldout = resolve_heldout(a.heldout)
    tasks = {"seen": [t for t in ALL_TASKS if t not in heldout], "heldout": heldout, "all": ALL_TASKS}[a.tasks]
    srcs = a.sources.split(",")
    eps = load_episodes(a.src, {"sources": srcs, "tasks": tasks, "include_failed": bool(a.include_failed),
                                "max_per_task": {s: a.per_task for s in srcs}})
    if not eps:
        sys.exit("no episodes matched")
    root = os.path.join(a.out, a.name)
    if os.path.exists(root):
        shutil.rmtree(root)
    has_mask = any("loss_mask" in ep for ep in eps)
    ds = LeRobotDataset.create(repo_id=f"local/{a.name}", fps=FPS, features=features(a.size, loss_mask=has_mask),
                               root=root,
                               robot_type="panda_mujoco", use_videos=True, vcodec=a.vcodec)
    # rebuild the env version the episodes were recorded in (gripper yaw, start state)
    cfgs = {json.dumps(ep["meta"].get("env_cfg"), sort_keys=True) for ep in eps}
    if len(cfgs) > 1:
        sys.exit(f"episodes come from different env versions: {cfgs}")
    env_cfg = eps[0]["meta"].get("env_cfg")
    env = PickPlaceEnv(env_config(env_cfg))
    side = []
    for i, ep in enumerate(eps):
        imgs, states, err = replay_render(env, ep, a.size)
        if err > a.max_replay_err:
            print(f"WARNING {ep['path']}: replay deviates from stored obs by {err:.2e}")
        sig = 0.0 if ep["source"] == "sim" else a.phone_sigma
        text = instruction(ep["task"])
        for t in range(len(ep["action"])):
            frame = {f"observation.images.{c}": imgs[c][t] for c in imgs}
            frame.update({"observation.state": states[t], "action": ep["action"][t].astype(np.float32),
                          "sigma_n": np.array([sig], np.float32), "task": text})
            if has_mask:
                frame["loss_mask"] = np.array([ep["loss_mask"][t] if "loss_mask" in ep else 1.0], np.float32)
            ds.add_frame(frame)
        ds.save_episode()
        side.append({"episode_index": i, "file": ep["path"], "task": task_name(ep["task"]), "source": ep["source"],
                     "success": ep["success"], "sigma_n": sig, "length": len(ep["action"]), "replay_err": err,
                     **({"loss_from_frame": int(np.argmax(ep["loss_mask"] > 0.5))} if "loss_mask" in ep else {})})
        if (i + 1) % 25 == 0 or i + 1 == len(eps):
            print(f"{i + 1}/{len(eps)} episodes")
    ds.finalize()
    with open(os.path.join(root, "episodes.json"), "w") as f:
        json.dump({"heldout": [task_name(t) for t in heldout], "env_cfg": env_cfg_dict(env.cfg), "episodes": side},
                  f, indent=1)
    print(f"wrote {root}: {len(eps)} episodes, {sum(s['length'] for s in side)} frames")


if __name__ == "__main__":
    main()
