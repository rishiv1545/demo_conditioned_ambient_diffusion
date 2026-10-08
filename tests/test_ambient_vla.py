import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.ambient import ambient_time, smolvla_default_time  # noqa: E402


def test_default_matches_smolvla_distribution():
    t = smolvla_default_time(200_000, "cpu", torch.Generator().manual_seed(0))
    ref = torch.distributions.Beta(1.5, 1.0).sample((200_000,)) * 0.999 + 0.001
    assert t.min() >= 0.001 and t.max() <= 1.0
    for q in (0.1, 0.5, 0.9):
        assert abs(torch.quantile(t, q) - torch.quantile(ref, q)) < 0.01


def test_off_switch_is_identity():
    sig = torch.tensor([0.0, 1.0, 0.0, 2.0] * 1000)
    a = ambient_time(sig, 0.0, torch.Generator().manual_seed(1))
    b = smolvla_default_time(len(sig), "cpu", torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_noisy_samples_only_get_noisy_times():
    sig = torch.tensor([0.0, 1.0] * 50_000)
    t = ambient_time(sig, 0.6, torch.Generator().manual_seed(2))
    clean, noisy = t[sig == 0], t[sig > 0]
    assert noisy.min() >= 0.6 and noisy.max() <= 1.0
    assert clean.min() < 0.1                     # clean samples still cover low noise
    assert (clean < 0.6).float().mean() > 0.4     # and are not restricted
