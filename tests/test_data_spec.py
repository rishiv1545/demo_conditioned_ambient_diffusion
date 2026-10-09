"""Dataset selection specs for VLA training ("root?per_task=4&cube=blue")."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.common import frames_of_episodes, parse_data_spec, selected_episodes  # noqa: E402


def write_eps(tmp_path, tasks):
    eps = [{"episode_index": i, "task": t} for i, t in enumerate(tasks)]
    (tmp_path / "episodes.json").write_text(json.dumps({"episodes": eps}))
    return str(tmp_path)


def test_parse():
    assert parse_data_spec("data/x") == ("data/x", {})
    assert parse_data_spec("data/x?per_task=4&cube=blue,red") == ("data/x", {"per_task": 4, "cube": {"blue", "red"}})
    with pytest.raises(ValueError):
        parse_data_spec("data/x?foo=1")


def test_select_first_n_per_task_and_cube(tmp_path):
    root = write_eps(tmp_path, ["blue-yellow", "red-yellow", "blue-yellow", "blue-orange", "blue-yellow", "blue-orange"])
    assert selected_episodes(root, {}) is None
    assert selected_episodes(root, {"per_task": 2}) == [0, 1, 2, 3, 5]
    assert selected_episodes(root, {"cube": {"blue"}}) == [0, 2, 3, 4, 5]
    assert selected_episodes(root, {"cube": {"blue"}, "per_task": 1}) == [0, 3]
    with pytest.raises(ValueError):
        selected_episodes(root, {"cube": {"green"}})


def test_frames_of_episodes():
    assert frames_of_episodes(np.array([0, 0, 1, 1, 1, 3]), [1, 3]) == [2, 3, 4, 5]
