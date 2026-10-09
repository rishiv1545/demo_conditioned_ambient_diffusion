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
        s = json.load(open(f))
        res[os.path.basename(os.path.dirname(f))] = s
    if not res:
        raise SystemExit(f"no finished runs in {a.out}")
    fmt = lambda s, k: f"{s[k]['success'] * 100:5.1f} [{s[k]['ci95'][0] * 100:4.1f}, {s[k]['ci95'][1] * 100:4.1f}]"
    order = lambda n: (0 if n == "C" else 1 if n == "Ceil" else 2, n[:2], re.sub(r"\d+$", "", n), int((re.findall(r"\d+$", n) or [0])[0]))
    print(f"{'run':12s} {'seen % [95% CI]':>22s} {'blue % [95% CI]':>22s}")
    for n in sorted(res, key=order):
        print(f"{n:12s} {fmt(res[n], 'seen'):>22s} {fmt(res[n], 'heldout'):>22s}")
    cands = {int(n[6:]): res[n]["heldout"]["success"] for n in res if re.fullmatch(r"L2_CNa\d+", n)}
    if cands:
        best = max(cands, key=lambda t: (cands[t], -t))
        print(f"\nchosen t_min (L2, C+N+amb): {best} of 100 -> SmolVLA --ambient_t_min {flow_t(best):.2f}")
        print("mapping:", ", ".join(f"t={t}: {flow_t(t):.2f}" for t in sorted(cands)))


if __name__ == "__main__":
    main()
