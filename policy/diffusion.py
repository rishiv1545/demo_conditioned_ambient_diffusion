"""Hand-rolled DDPM (cosine schedule, epsilon prediction) with DDIM sampling."""
import math

import torch


class Diffusion:
    def __init__(self, T=100, device="cpu"):
        self.T = T
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        abar = f / f[0]
        betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        self.alpha_bar = torch.cumprod(1 - betas, 0).float().to(device)   # alpha_bar[t], t = 0..T-1

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0, cond, t_min=None, reduce=True):
        """Epsilon-prediction MSE. t_min (optional, [B] ints): sample i draws t uniformly from [t_min_i, T-1];
        None means standard DDPM (t_min = 0 for every sample)."""
        B = x0.shape[0]
        if t_min is None:
            t = torch.randint(0, self.T, (B,), device=x0.device)
        else:
            t_min = torch.as_tensor(t_min, device=x0.device).long().clamp(0, self.T - 1)
            t = t_min + (torch.rand(B, device=x0.device) * (self.T - t_min).float()).long()
        noise = torch.randn_like(x0)
        eps = model(self.q_sample(x0, t, noise), t, cond)
        per = ((eps - noise) ** 2).flatten(1).mean(1)
        return per.mean() if reduce else per

    @torch.no_grad()
    def sample(self, model, cond, shape, n_steps=10, generator=None):
        """Deterministic DDIM (eta = 0) with n_steps evenly spaced timesteps."""
        x = torch.randn(shape, device=cond.device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, n_steps).round().long().tolist()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            eps = model(x, torch.full((shape[0],), t, device=x.device, dtype=torch.long), cond)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x
