"""Ambient flow matching for SmolVLA: per-sample restriction of flow times. This is the whole patch.

SmolVLA interpolates x_t = t * noise + (1 - t) * actions, so t = 1 is pure noise. Its default training time
distribution is t = 0.001 + 0.999 * Beta(1.5, 1). Ambient: samples with sigma_n > 0 (phone-derived, i.e.
corrupted) are mapped to t_min + (1 - t_min) * t, so they only supervise the noisy end, where the corruption is
drowned out by the added noise. Clean samples (sigma_n == 0) keep the default distribution. t_min = 0 is off.

The times are passed to SmolVLAPolicy.forward(batch, time=...); LeRobot's source is not modified.
"""
import torch


def smolvla_default_time(bsize, device, generator=None):
    """Same distribution as lerobot's VLAFlowMatching.sample_time: 0.001 + 0.999 * Beta(1.5, 1)."""
    # Beta(1.5, 1) by inverse CDF: F(x) = x^1.5, so x = u^(1/1.5). Exact, and works with a torch.Generator.
    u = torch.rand(bsize, generator=generator, device="cpu").to(device)
    return u.pow(1.0 / 1.5) * 0.999 + 0.001


def ambient_time(sigma_n, t_min, generator=None):
    """Flow times [B] for a batch with per-sample sigma_n [B]. t_min <= 0 reproduces the default exactly."""
    sigma_n = sigma_n.reshape(-1)
    t = smolvla_default_time(sigma_n.shape[0], sigma_n.device, generator)
    if t_min > 0:
        noisy = sigma_n > 0
        t = torch.where(noisy, t_min + (1.0 - t_min) * t, t)
    return t
