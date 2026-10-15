"""Held-out paired evaluation (colab/runs.json evalsets.P5_fresh): success with Wilson 95% intervals per checkpoint
and exact McNemar tests on the seeds two checkpoints share.

    python colab/dcad.py fetch P5_naive_15r P5_amb_15 P5_qual_15 ...
    python scripts/paired_eval.py                     # markdown tables -> stdout (and --out json)
"""
import argparse
import csv
import json
import os
from math import comb, sqrt

COLAB = "outputs/colab"
TAG = "_br_zn_fresh"
CHECKPOINTS = [   # label, run, step
    ("naive 5k", "P5_naive_15r", 5000), ("naive 7.5k", "P5_naive_15r", 7500), ("naive 10k", "P5_naive_15r", 10000),
    ("ambient 7.5k", "P5_amb_15", 7500), ("ambient 10k", "P5_amb_15", 10000), ("quality 10k", "P5_qual_15", 10000),
    ("VN (naive) 10k", "P5_naive_15_vn", 10000), ("self-improvement 13k", "P5_SI_15", 13000),
    ("naive@30 10k", "P5_naive_30", 10000), ("ambient@30 10k", "P5_amb_30", 10000)]
PAIRS = [("ambient 10k", "naive 10k"), ("ambient 10k", "naive 5k"), ("ambient 10k", "quality 10k"),
         ("naive 5k", "naive 10k"), ("self-improvement 13k", "ambient 10k"), ("ambient@30 10k", "naive@30 10k"),
         ("VN (naive) 10k", "naive 10k"), ("VN (naive) 10k", "ambient 10k")]


def wilson(s, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = s / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p (binomial test on the discordant pairs b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load(run, step):
    """{seed: success} from the largest-k fresh eval of run at step."""
    ds = [d for d in os.listdir(f"{COLAB}/{run}") if d.startswith(f"eval_step{step:06d}_k") and d.endswith(TAG)
          and os.path.exists(f"{COLAB}/{run}/{d}/episodes.csv")] if os.path.isdir(f"{COLAB}/{run}") else []
    if not ds:
        return None
    d = max(ds, key=lambda d: int(d.split("_k")[1].split("_")[0]))
    return {int(r["seed"]): r["success"] == "1" for r in csv.DictReader(open(f"{COLAB}/{run}/{d}/episodes.csv"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    a = p.parse_args()
    res = {lab: load(run, step) for lab, run, step in CHECKPOINTS}
    out = {"checkpoints": {}, "pairs": []}
    print("| checkpoint | episodes | success % | Wilson 95% CI |\n|---|---|---|---|")
    for lab, run, step in CHECKPOINTS:
        r = res[lab]
        if r is None:
            print(f"| {lab} | – | *pending* | |")
            continue
        s, n = sum(r.values()), len(r)
        lo, hi = wilson(s, n)
        out["checkpoints"][lab] = {"run": run, "step": step, "n": n, "success": s / n, "ci95": [lo, hi]}
        print(f"| {lab} | {n} | {100 * s / n:.1f} | {100 * lo:.0f}–{100 * hi:.0f} |")
    print("\n| comparison (A vs B) | shared episodes | A % | B % | A only | B only | exact McNemar p |\n"
          "|---|---|---|---|---|---|---|")
    for x, y in PAIRS:
        if res[x] is None or res[y] is None:
            print(f"| {x} vs {y} | *pending* | | | | | |")
            continue
        seeds = sorted(set(res[x]) & set(res[y]))
        b = sum(res[x][s] and not res[y][s] for s in seeds)
        c = sum(res[y][s] and not res[x][s] for s in seeds)
        pa = 100 * sum(res[x][s] for s in seeds) / len(seeds)
        pb = 100 * sum(res[y][s] for s in seeds) / len(seeds)
        pv = mcnemar_exact(b, c)
        out["pairs"].append({"a": x, "b": y, "n": len(seeds), "a_success": pa, "b_success": pb, "a_only": b,
                             "b_only": c, "p": pv})
        print(f"| {x} vs {y} | {len(seeds)} | {pa:.1f} | {pb:.1f} | {b} | {c} | "
              f"{pv:.2g} |" if pv >= 1e-4 else f"| {x} vs {y} | {len(seeds)} | {pa:.1f} | {pb:.1f} | {b} | {c} | < 1e-4 |")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
