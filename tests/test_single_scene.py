"""Single-object scene (env v3s): only the blue cube and the red zone; the multi-object scene is unchanged."""
import numpy as np

from policy.data import layout_from_array, save_episode
from sim.env import (SINGLE_TASK, EnvConfig, OBS_DIM, PickPlaceEnv, env_cfg_dict, env_config, instruction,
                     task_index)
from sim.expert import run_expert_episode


def test_single_scene(tmp_path):
    env = PickPlaceEnv(EnvConfig(scene="single"))
    assert instruction(SINGLE_TASK) == "put the blue cube in the red zone" and task_index(SINGLE_TASK) == 9
    assert env_config(env_cfg_dict(env.cfg)) == env.cfg
    r = run_expert_episode(env, SINGLE_TASK, 123)
    assert r["success"] and r["obs"].shape[1] == OBS_DIM + 2
    x0, x1, y0, y1 = env.cfg.single_cube_box
    cube = env.layout["cubes"][2]
    assert x0 <= cube[0] <= x1 and y0 <= cube[1] <= y1
    assert np.allclose(env.layout["zones"][3], env.cfg.single_zone_xy)
    for i in (0, 1):                                  # absent cubes stay parked below the table
        assert env.cube_pos(i)[2] == -0.5
    save_episode(tmp_path / "e.npz", r["obs"], r["action"], SINGLE_TASK, "sim", env.layout, True)
    lay = layout_from_array(np.load(tmp_path / "e.npz")["layout"])
    assert lay["zones"].shape == (4, 2) and np.allclose(lay["zones"], env.layout["zones"])


def test_multi_scene_unchanged():
    env = PickPlaceEnv()
    assert env_cfg_dict(env.cfg) == {"gripper_yaw_deg": 90.0, "start_gripper_closed": True}
    assert env.reset((2, 2), seed=5).shape == (OBS_DIM,)
    assert env.d.mocap_pos[env.zone_mocap[3]][2] == -0.5   # the red zone is parked
