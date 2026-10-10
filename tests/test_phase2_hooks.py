"""Smoke tests for the four Phase 2 hooks."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policy.data import ChunkDataset, DataConfig, load_episodes, save_episode  # noqa: E402
from policy.diffusion import Diffusion  # noqa: E402
from policy.model import ConditionEncoder, ObsTaskEncoder, Policy, mlp  # noqa: E402
from sim.env import OBS_DIM  # noqa: E402


def _fake_episodes(tmp, n_per=(("sim", (0, 1), 3), ("human", (0, 1), 2), ("sim", (2, 2), 2))):
    rng = np.random.default_rng(0)
    lay = {"cubes": np.zeros((3, 2)), "zones": np.zeros((3, 2))}
    k = 0
    for source, task, n in n_per:
        for _ in range(n):
            T = int(rng.integers(20, 40))
            save_episode(os.path.join(tmp, f"ep{k:03d}.npz"), rng.normal(size=(T, OBS_DIM)), rng.normal(size=(T, 4)),
                         task, source, lay, success=bool(k % 2) or source == "sim")
            k += 1
    return load_episodes(str(tmp))


# 1. per-sample minimum timestep
def test_loss_t_min_per_sample():
    diff = Diffusion(T=100)
    seen = []

    def model(x, t, cond):
        seen.append(t.clone())
        return torch.zeros_like(x)

    x0, cond = torch.zeros(512, 16, 4), torch.zeros(512, 8)
    diff.loss(model, x0, cond)                                    # standard DDPM
    assert seen[-1].min() >= 0 and seen[-1].max() <= 99 and seen[-1].min() < 10
    t_min = torch.cat([torch.zeros(256), torch.full((256,), 70)]).long()
    per = diff.loss(model, x0, cond, t_min=t_min, reduce=False)
    t = seen[-1]
    assert per.shape == (512,)
    assert t[:256].min() < 30 and (t[256:] >= 70).all() and (t <= 99).all()


# 2. conditioning through a swappable ConditionEncoder
class ContextEncoder(ConditionEncoder):
    """Stand-in for Phase 2: observation + a context demo (keyframes) instead of the task one-hot."""

    def __init__(self, obs_dim, ctx_dim):
        super().__init__()
        self.obs_enc, self.ctx_enc = mlp(obs_dim, 64, 32), mlp(ctx_dim, 64, 32)
        self.cond_dim = 64

    def forward(self, batch):
        return torch.cat([self.obs_enc(batch["obs"]), self.ctx_enc(batch["context"].flatten(1))], -1)


def test_condition_encoder_swap():
    diff = Diffusion(T=100)
    x0 = torch.randn(4, 16, 4)
    obs = torch.randn(4, OBS_DIM)
    for enc, batch in [(ObsTaskEncoder(OBS_DIM), {"obs": obs, "task": torch.zeros(4, 6)}),
                       (ContextEncoder(OBS_DIM, 16 * (OBS_DIM + 4)), {"obs": obs, "context": torch.randn(4, 16, OBS_DIM + 4)})]:
        pol = Policy(enc, dims=(32, 64, 128))
        loss = diff.loss(pol, x0, pol.encode(batch))
        loss.backward()
        assert torch.isfinite(loss)
        assert diff.sample(pol, pol.encode(batch), (4, 16, 4), n_steps=3).shape == (4, 16, 4)


# 3. context sampling
def test_context_sampler(tmp_path):
    eps = _fake_episodes(tmp_path)
    ds = ChunkDataset(eps, DataConfig(horizon=16, n_obs=2))
    rng = np.random.default_rng(0)
    for k in rng.integers(0, len(ds), 50):
        i, _ = ds.index[k]
        ctx = ds.sample_context(int(k), rng)
        assert ctx is not None and ctx["ep"] != i and eps[ctx["ep"]]["task"] == eps[i]["task"]
        assert ctx["obs"].shape == (16, OBS_DIM) and ctx["action"].shape == (16, 4)
        h = ds.sample_context(int(k), rng, source="human")
        if eps[i]["task"] == (0, 1):
            assert h is not None and eps[h["ep"]]["source"] == "human" and h["ep"] != i
        else:
            assert h is None  # no human demos of (2, 2)


# 4. sigma_n flows from episode files to batches
def test_sigma_n_flows(tmp_path):
    eps = _fake_episodes(tmp_path)
    assert all(e["sigma_n"] == 0.0 for e in eps if e["source"] == "sim")
    assert all(np.isnan(e["sigma_n"]) for e in eps if e["source"] == "human")
    ds = ChunkDataset(eps)
    b = ds.arrays()
    src = np.array([eps[i]["source"] for i in b["ep"]])
    assert np.all(b["sigma_n"][src == "sim"] == 0.0) and np.all(np.isnan(b["sigma_n"][src == "human"]))
    # filters
    assert all(e["success"] for e in load_episodes(str(tmp_path), {"include_failed": False}))
    assert {e["source"] for e in load_episodes(str(tmp_path), {"sources": ["human"]})} == {"human"}


def test_loss_t_max_per_sample():
    """Locality: samples with t_max only see t < t_max; others keep [t_min, T)."""
    import torch
    from policy.diffusion import Diffusion
    diff = Diffusion(T=100)
    seen = []

    def model(x, t, cond):
        seen.append(t.clone())
        return torch.zeros_like(x)
    x0, cond = torch.zeros(512, 4, 2), None
    t_min = torch.cat([torch.zeros(256), torch.full((256,), 70)]).long()
    t_max = torch.cat([torch.full((256,), 25), torch.full((256,), 100)]).long()
    diff.loss(model, x0, cond, t_min=t_min, t_max=t_max)
    t = seen[0]
    assert t[:256].max() < 25 and t[:256].min() >= 0
    assert t[256:].min() >= 70 and t[256:].max() < 100
