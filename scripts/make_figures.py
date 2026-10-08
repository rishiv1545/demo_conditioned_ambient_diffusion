"""Results table (markdown + CSV) and seen/held-out bar chart from evaluate.py outputs.

    python scripts/make_figures.py --runs A B Bprime --out outputs/m3
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {"A": "A: sim only", "B": "B: sim + human (all)", "Bprime": "B′: sim + human (successful replays)",
          "oracle": "Oracle: sim, all 9 tasks"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["A", "B", "Bprime"])
    p.add_argument("--out", default="outputs/m3")
    a = p.parse_args()
    res = {}
    for r in a.runs:
        f = os.path.join(a.out, r, "summary.json")
        if os.path.exists(f):
            with open(f) as fh:
                res[r] = json.load(fh)
        else:
            print(f"skipping {r}: {f} not found")
    if not res:
        return
    lines = ["| run | seen tasks | held-out tasks |", "|---|---|---|"]
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "seen", "seen_lo", "seen_hi", "heldout", "heldout_lo", "heldout_hi", "n_seen", "n_heldout"])
        for r, s in res.items():
            S, Hd = s["seen"], s["heldout"]
            w.writerow([r, S["success"], *S["ci95"], Hd["success"], *Hd["ci95"], S["n"], Hd["n"]])
            fmt = lambda d: f"{d['success'] * 100:.0f}% [{d['ci95'][0] * 100:.0f}–{d['ci95'][1] * 100:.0f}]"
            lines.append(f"| {LABELS.get(r, r)} | {fmt(S)} | {fmt(Hd)} |")
    table = "\n".join(lines)
    with open(os.path.join(a.out, "results.md"), "w") as fh:
        fh.write(table + "\n\nSuccess rate with 95% Wilson confidence interval.\n")
    print(table)

    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(res), 3.6))
    x = np.arange(len(res))
    wd = 0.38
    for k, (split, col) in enumerate((("seen", "#4C72B0"), ("heldout", "#DD8452"))):
        m = np.array([res[r][split]["success"] for r in res])
        lo = np.array([res[r][split]["ci95"][0] for r in res])
        hi = np.array([res[r][split]["ci95"][1] for r in res])
        ax.bar(x + (k - 0.5) * wd, m, wd, yerr=[m - lo, hi - m], capsize=3, color=col,
               label="seen tasks" if split == "seen" else "held-out tasks")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(r, r).split(":")[0] for r in res])
    ax.set_ylim(0, 1)
    ax.set_ylabel("success rate")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "baselines.png"), dpi=150)
    print("wrote", os.path.join(a.out, "baselines.png"))


if __name__ == "__main__":
    main()
