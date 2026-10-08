import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.env import OBS_DIM, PickPlaceEnv  # noqa: E402
from sim.expert import run_expert_episode  # noqa: E402


def test_reset_step_shapes():
    env = PickPlaceEnv()
    o = env.reset(task=(1, 2), seed=3)
    assert o.shape == (OBS_DIM,)
    assert np.allclose(o[:3], env.cfg.home_ee, atol=0.01)
    o2, s = env.step([*env.cfg.home_ee, 0.0])
    assert o2.shape == (OBS_DIM,) and s is False


def test_layout_given_is_used():
    env = PickPlaceEnv()
    lay = {"cubes": np.array([[0.1, 0.1], [0.25, 0.1], [0.4, 0.1]]),
           "zones": np.array([[0.1, 0.27], [0.25, 0.27], [0.4, 0.27]])}
    o = env.reset(task=(0, 0), layout=lay)
    assert np.allclose(o[4:13].reshape(3, 3)[:, :2], lay["cubes"])
    assert np.allclose(o[13:].reshape(3, 2), lay["zones"])


def test_success_detection():
    env = PickPlaceEnv()
    lay = {"cubes": np.array([[0.1, 0.1], [0.25, 0.1], [0.4, 0.1]]),
           "zones": np.array([[0.1, 0.27], [0.25, 0.27], [0.4, 0.27]])}
    env.reset(task=(0, 1), layout=lay)
    assert not env.success()
    a = env.cube_qadr[0]
    env.d.qpos[a:a + 3] = [0.25, 0.27, 0.02]   # teleport red cube into the purple zone
    import mujoco
    mujoco.mj_forward(env.m, env.d)
    assert env.success()
    assert not env.success(task=(0, 0))        # wrong zone
    assert not env.success(task=(1, 1))        # wrong cube


def test_expert_fixed_seed():
    env = PickPlaceEnv()
    r = run_expert_episode(env, (2, 0), seed=123)
    assert r["success"]
    assert r["obs"].shape[1] == OBS_DIM and r["action"].shape[1] == 4


def test_object_split_blue_is_distractor():
    from sim.env import DEFAULT_SPLIT, SPLITS, resolve_heldout
    assert DEFAULT_SPLIT == "object" and resolve_heldout("object") == [(2, 0), (2, 1), (2, 2)]
    assert resolve_heldout("combo") == SPLITS["combo"]
    env = PickPlaceEnv()
    for task in [(0, 1), (1, 2)]:          # seen tasks: blue cube present, on the table, never moved
        r = run_expert_episode(env, task, seed=5)
        blue = r["obs"][:, 10:13]
        assert r["success"]
        assert np.allclose(blue[:, 2], 0.02, atol=2e-3)
        assert np.abs(blue[:, :2] - blue[0, :2]).max() < 2e-3
