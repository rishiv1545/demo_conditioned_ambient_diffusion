"""Table of the small-policy ambient sweep (scripts/run_ambient_sweep.sh) and the t_min choice for the VLA runs.

    python scripts/summarize_ambient_sweep.py [--out outputs/ambient_sweep]

t_min is chosen as the C+N+amb setting with the best blue success at the phone-like noise level (L2), and is mapped
to SmolVLA's flow-matching time by matching the noise-to-signal ratio: DDPM step t has sqrt(1 - abar_t) / sqrt(abar_t)
(cosine schedule, T = 100), flow time tau has tau / (1 - tau) (x_tau = tau * noise + (1 - tau) * actions).
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np


def abar(t, T=100, s=0.008):
    """alpha_bar[t] of policy/diffusion.py (product of the clamped betas; equal to f(t+1)/f(0) away from t = T-1)."""
    f = lambda u: math.cos((u / T + s) / (1 + s) * math.pi / 2) ** 2
    out = 1.0
    for k in range(t + 1):
        out *= 1 - min(1 - f(k + 1) / f(k), 0.999)
    return out


def flow_t(t):
    r = math.sqrt((1 - abar(t)) / abar(t))
    return r / (1 + r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/ambient_sweep")
    a = p.parse_args()
    res = {}
    for f in glob.glob(os.path.join(a.out, "*", "summary.json")):
        res[os.path.basename(os.path.dirname(f))] = json.load(open(f))
    if not res:
        raise SystemExit(f"no finished runs in {a.out}")
    groups = {}   # seeds: <name>_s<k> joins <name>
    for n, s in res.items():
        groups.setdefault(re.sub(r"_s\d+$", "", n), []).append((s["seen"]["success"], s["heldout"]["success"]))

    def key(n):
        return (n.split("_")[0], re.sub(r"\d+$", "", n), int(re.findall(r"\d+$", n)[0]) if re.search(r"\d$", n) else -1)
    print(f"{'run':14s} {'seeds':>5s} {'seen %':>16s} {'blue %':>16s}   (mean ± std over seeds; single seed: value)")
    for n in sorted(groups, key=key):
        v = np.array(groups[n]) * 100
        f = lambda c: f"{v[:, c].mean():5.1f} ± {v[:, c].std():4.1f}" if len(v) > 1 else f"{v[0, c]:5.1f}       "
        print(f"{n:14s} {len(v):5d} {f(0):>16s} {f(1):>16s}")
    for row, pre in (("4 clean", "L2_CNa"), ("0 clean", "L2_Na"), ("24 clean", "K24_Na")):
        c = {int(n[len(pre):]): np.mean([b for _, b in groups[n]]) for n in groups if re.fullmatch(pre + r"\d+", n)}
        if c:
            best = max(c, key=c.get)
            print(f"best t_min, {row}: {best} (blue {c[best] * 100:.1f}%) -> SmolVLA --ambient_t_min {flow_t(best):.2f}")
    print("mapping:", ", ".join(f"t={t}: {flow_t(t):.2f}" for t in (10, 25, 50, 75, 90)))


if __name__ == "__main__":
    main()
