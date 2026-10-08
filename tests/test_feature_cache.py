"""Cached vision features must reproduce raw-image training exactly (up to fp16 storage of the features).

Integration test: needs the SmolVLA weights in the HF cache and data/lerobot/smoke_h264 (vla/export_lerobot.py
--per_task 3 --name smoke_h264). Skipped otherwise.
"""
import os
import shutil
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SMOKE = os.path.join(ROOT, "data", "lerobot", "smoke_h264")
HF = os.path.expanduser("~/.cache/huggingface/hub/models--lerobot--smolvla_base")

pytestmark = pytest.mark.skipif(not (os.path.isdir(SMOKE) and os.path.isdir(HF)),
                                reason="needs data/lerobot/smoke_h264 and cached lerobot/smolvla_base weights")


def test_cached_features_match_raw_images(tmp_path):
    from vla.common import build_policy, open_datasets
    from vla.feature_cache import PREFIX, CachedChunkDataset, build_cache, use_cached_features
    root = str(tmp_path / "smoke")
    shutil.copytree(SMOKE, root)
    chunk = 50
    dss, stats = open_datasets([root], chunk)
    policy, pre, _, cfg = build_policy(dss[0].meta.features, stats, "cpu", dtype="fp32")
    policy.eval()
    keys = list(cfg.image_features)
    build_cache(root, policy, "cpu", batch=64, workers=0)
    cds = CachedChunkDataset(root, chunk, keys)
    idx = [3, len(cds) - 2]                     # one mid-episode sample, one with a padded chunk tail
    raw = torch.utils.data.default_collate([dss[0][i] for i in idx])
    cac = torch.utils.data.default_collate([cds[i] for i in idx])
    assert torch.allclose(raw["observation.state"], cac["observation.state"])
    assert torch.allclose(raw["action"], cac["action"])
    assert torch.equal(raw["action_is_pad"], cac["action_is_pad"]) and cac["action_is_pad"][1].any()
    assert raw["task"] == cac["task"]

    g = torch.Generator().manual_seed(0)
    noise = torch.randn(2, chunk, cfg.max_action_dim, generator=g)
    t = torch.tensor([0.3, 0.8])
    b = pre(raw)
    b["actions_id_pad"] = b["action_is_pad"]
    with torch.no_grad():
        l_raw = policy.forward(b, noise=noise, time=t, reduction="none")[0]
    use_cached_features(policy)
    b2 = pre(cac)
    b2["actions_id_pad"] = b2["action_is_pad"]
    for k in keys:
        b2[PREFIX + k] = cac[PREFIX + k]
    with torch.no_grad():
        l_cac = policy.forward(b2, noise=noise, time=t, reduction="none")[0]
    assert torch.allclose(l_raw, l_cac, rtol=2e-3, atol=1e-4), (l_raw, l_cac)
