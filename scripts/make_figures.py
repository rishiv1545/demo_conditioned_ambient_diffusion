"""README figures for the single-object phone task, from the fetched Colab results (outputs/colab/<run>/eval_*).

    python scripts/make_figures.py --single    # success vs number of phone demos (P5_naive_*), 15 clean sim demos as a reference
    python scripts/make_figures.py --methods   # final checkpoint by method at 15 phone demos (held-out evaluation) + table
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# validated categorical slots (dataviz reference palette, light)
AMB, NAIVE, CLEAN = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLAB = "outputs/colab"


def style(ax):
    ax.set_facecolor(SURF)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def eval_point(run, step, tag="_br_zn"):
    """(success %, ci95 lo %, hi %) of a run's eval at step (largest k if several), or None."""
    import glob
    fs = sorted(glob.glob(f"{COLAB}/{run}/eval_step{step:06d}_k*{tag}/summary.json"),
                key=lambda f: int(f.split("_k")[-1].split("_")[0]))
    if not fs:
        return None
    h = json.load(open(fs[-1]))["heldout"]
    return 100 * h["success"], 100 * h["ci95"][0], 100 * h["ci95"][1]


def single_task(out_fig="docs/figures"):
    """Success vs number of phone demos on the single-object task (end of training), with 15 clean sim demos as a
    reference point."""
    phone = [(15, "P5_naive_15", 10000), (30, "P5_naive_30", 10000), (60, "P5_naive_60", 15000),
             (100, "P5_naive_all", 15000)]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURF)
    style(ax)
    rows = ["| demos | data | end of training % |", "|---|---|---|"]
    xs, ys = [], []
    for n, run, st in phone:
        e = eval_point(run, st)
        if e:
            xs.append(n); ys.append(e[0])
            rows.append(f"| {n} | phone demos | {e[0]:.1f} |")
    ax.plot(xs, ys, color=AMB, lw=2, marker="o", ms=6, markeredgecolor=SURF, markeredgewidth=1.5, label="phone demos")
    ref = eval_point("S1b_sim_15", 10000)
    if ref:
        rows.append(f"| 15 | clean sim demos | {ref[0]:.1f} |")
        ax.plot([15], [ref[0]], ls="none", marker="D", ms=7, color=CLEAN, markeredgecolor=SURF, markeredgewidth=1.5,
                label="15 clean sim demos (reference)")
    ax.set_xscale("log")
    ax.set_xlim(12, 125)
    ax.set_xticks([15, 30, 60, 100])
    ax.set_xticklabels(["15", "30", "60", "100"])
    ax.minorticks_off()
    ax.set_ylim(0, 105)
    ax.set_xlabel("demos (single object, one task)", fontsize=8, color=INK2)
    ax.set_ylabel("success %, end of training", fontsize=8, color=INK2)
    ax.set_title("Phone-only post-training of SmolVLA, single-object task", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    os.makedirs(out_fig, exist_ok=True)
    fig.savefig(os.path.join(out_fig, "single_task_demos.png"), dpi=160, facecolor=SURF)
    print("\n".join(rows))


METHODS_15 = [("naive", "P5_naive_15r", NAIVE), ("ambient", "P5_amb_15", AMB), ("quality-tagged", "P5_qual_15", "#8c5ac8"),
              ("velocity-normalized (naive)", "P5_naive_15_vn", "#52514e")]
STEPS_15 = [2500, 5000, 7500, 10000]


def methods(out_fig="docs/figures"):
    """Bar chart: each method's final checkpoint at 15 phone demos on the held-out paired evaluation (fresh layouts,
    _br_zn_fresh); and the markdown table of the fixed-step curves on the standard layouts."""
    rows = ["| method | 2.5k | 5k | 7.5k | 10k (40 ep., 95% CI) | validation, 10k |", "|---|---|---|---|---|---|"]
    for label, run, c in METHODS_15:
        pts = {st: eval_point(run, st) for st in STEPS_15}
        if not any(pts.values()):
            rows.append(f"| {label} | – | – | – | pending | – |")
            continue
        val = eval_point(run, 10000, "_br_zn_val")
        cell = lambda st: f"{pts[st][0]:.0f}" if pts[st] else "–"
        end = pts[10000]
        rows.append(f"| {label} | {cell(2500)} | {cell(5000)} | {cell(7500)} | "
                    + (f"{end[0]:.1f} ({end[1]:.0f}–{end[2]:.0f})" if end else "–")
                    + (f" | {val[0]:.1f} |" if val else " | – |"))
    base = eval_point("P5_naive_15r", 5000)
    if base:
        rows.insert(2, f"| *baseline: naive, 5k checkpoint* | | {base[0]:.0f} | | | |")
    si = json.load(open(f"{COLAB}/P5_SI_15/self_improve_best.json")) if os.path.exists(
        f"{COLAB}/P5_SI_15/self_improve_best.json") else None
    sp = eval_point("P5_SI_15", 13000)
    rows.append(f"| self-improvement (1 round on {dict(P5_naive_15r='naive', P5_amb_15='ambient', P5_qual_15='quality-tagged').get(si['best'], si['best']) if si else '?'}) | | | | "
                + (f"{sp[0]:.1f} ({sp[1]:.0f}–{sp[2]:.0f}) at 13k" if sp else "pending") + " | |")
    import glob
    short = {"naive": "naive", "ambient": "ambient", "quality-tagged": "quality-\ntagged",
             "velocity-normalized (naive)": "velocity-\nnormalized"}
    bars = [(short[l], run, 10000) for l, run, _ in METHODS_15] + [("self-\nimprovement\n(13k)", "P5_SI_15", 13000)]
    vals = []
    for l, r, st in bars:
        fs = sorted(glob.glob(f"{COLAB}/{r}/eval_step{st:06d}_k*_br_zn_fresh/summary.json"),
                    key=lambda f: int(f.split("_k")[-1].split("_")[0]))
        if fs:
            sm = json.load(open(fs[-1]))
            vals.append((l, 100 * sm["heldout"]["success"], sm["k"]))
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURF)
    style(ax)
    x = np.arange(len(vals))
    ax.bar(x, [v for _, v, _ in vals], width=0.6, color=AMB)
    for xi, (_, v, k) in zip(x, vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in vals], fontsize=7.5)
    ax.set_ylim(0, 105)
    ks = sorted({k for _, _, k in vals})
    ax.set_ylabel(f"success % (held-out layouts, {' or '.join(map(str, ks))} episodes)", fontsize=8, color=INK2)
    ax.set_title("15 phone demos: final checkpoint (10k steps) by method", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_fig, "methods_15.png"), dpi=160, facecolor=SURF)
    print("\n".join(rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true", help="success vs number of phone demos")
    p.add_argument("--methods", action="store_true", help="final checkpoint by method at 15 phone demos")
    a = p.parse_args()
    if a.methods:
        methods()
    if a.single:
        single_task()


if __name__ == "__main__":
    main()
