"""Noise-prediction network: a small 1D temporal U-Net (Diffusion Policy style) with FiLM conditioning.

The network only ever sees a `cond` vector produced by a swappable ConditionEncoder. Phase 1 uses
ObsTaskEncoder (observation + task one-hot); Phase 2 adds an encoder over (observation, context demo).
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------- condition encoders
class ConditionEncoder(nn.Module):
    """Interface: forward(batch) -> cond [B, cond_dim]. `batch` is a dict of tensors."""
    cond_dim: int


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.Mish(), nn.Linear(h, o))


class ObsTaskEncoder(ConditionEncoder):
    """Observation and task are encoded by separate MLPs and concatenated."""

    def __init__(self, obs_dim, task_dim=6, obs_emb=128, task_emb=64):
        super().__init__()
        self.obs_enc = mlp(obs_dim, 256, obs_emb)
        self.task_enc = mlp(task_dim, 64, task_emb)
        self.cond_dim = obs_emb + task_emb

    def forward(self, batch):
        return torch.cat([self.obs_enc(batch["obs"]), self.task_enc(batch["task"])], -1)


# ---------------------------------------------------------------------------- U-Net
class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)


class ConvBlock(nn.Sequential):
    def __init__(self, i, o, k=5):
        super().__init__(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(8, o), nn.Mish())


class CondResBlock(nn.Module):
    def __init__(self, i, o, cdim):
        super().__init__()
        self.b1, self.b2 = ConvBlock(i, o), ConvBlock(o, o)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cdim, 2 * o))
        self.skip = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, c):
        h = self.b1(x)
        scale, bias = self.film(c).unsqueeze(-1).chunk(2, 1)
        h = self.b2(h * (1 + scale) + bias)
        return h + self.skip(x)


class TemporalUnet(nn.Module):
    def __init__(self, act_dim, cond_dim, dims=(64, 128, 256), t_emb=64):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalEmb(t_emb), nn.Linear(t_emb, 4 * t_emb), nn.Mish(), nn.Linear(4 * t_emb, t_emb))
        cdim = cond_dim + t_emb
        ch = [act_dim, *dims]
        self.down = nn.ModuleList()
        for k, (i, o) in enumerate(zip(ch[:-1], ch[1:])):
            last = k == len(dims) - 1
            self.down.append(nn.ModuleList([CondResBlock(i, o, cdim), CondResBlock(o, o, cdim),
                                            nn.Identity() if last else nn.Conv1d(o, o, 3, 2, 1)]))
        self.mid = nn.ModuleList([CondResBlock(dims[-1], dims[-1], cdim), CondResBlock(dims[-1], dims[-1], cdim)])
        self.up = nn.ModuleList()
        for i, o in reversed(list(zip(dims[:-1], dims[1:]))):
            self.up.append(nn.ModuleList([CondResBlock(2 * o, i, cdim), CondResBlock(i, i, cdim),
                                          nn.ConvTranspose1d(i, i, 4, 2, 1)]))
        self.final = nn.Sequential(ConvBlock(dims[0], dims[0]), nn.Conv1d(dims[0], act_dim, 1))

    def forward(self, x, t, cond):
        """x [B, H, act_dim], t [B] int timesteps, cond [B, cond_dim] -> eps [B, H, act_dim]"""
        c = torch.cat([cond, self.t_emb(t)], -1)
        h, skips = x.transpose(1, 2), []
        for r1, r2, ds in self.down:
            h = r2(r1(h, c), c)
            skips.append(h)
            h = ds(h)
        for m in self.mid:
            h = m(h, c)
        for r1, r2, us in self.up:
            h = r2(r1(torch.cat([h, skips.pop()], 1), c), c)
            h = us(h)
        return self.final(h).transpose(1, 2)


class Policy(nn.Module):
    """ConditionEncoder + noise-prediction network. eps = policy(x_t, t, batch)."""

    def __init__(self, encoder: ConditionEncoder, act_dim=4, dims=(64, 128, 256)):
        super().__init__()
        self.encoder = encoder
        self.net = TemporalUnet(act_dim, encoder.cond_dim, dims)

    def encode(self, batch):
        return self.encoder(batch)

    def forward(self, x, t, cond):
        return self.net(x, t, cond)
