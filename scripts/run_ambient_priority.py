"""Priority-ordered small-policy ambient runs (Mac, MPS), 2 lanes in parallel. Phases run in order; within a phase,
runs share the lanes. Finished runs (summary.json) are skipped; a run whose training is in progress elsewhere is
waited for and then only evaluated; a trained run without an eval is only evaluated.

    PY=~/miniforge3/envs/dcad/bin/python python scripts/run_ambient_priority.py [--lanes 2]

Run names (results in outputs/ambient_sweep/<name>/summary.json):
  C, L2_CN, L2_CNa<t>   4 clean demos per blue task: clean only / + noisy (L2, all tasks) naive / ambient t_min=t
  L2_N0, L2_Na<t>       0 clean blue: noisy blue demos only, naive / ambient
  Ceil, K24_N, K24_Na<t>  24 clean per blue task
  K4_P*, K0_P*          the same with the real phone replays (data/human/3) instead of synthetic noise
  <name>_s<k>           training seed k (seed 0 = no suffix)
  L3_*, L1_*, K8*, K12*   lower priority: other noise levels and clean amounts
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "ambient_sweep")
PY = os.environ.get("PY", sys.executable)
S, P = "data/synth_L2", "data/human/3"
BASE = ["--data", "data/sim_v2"]


def cell(name, noisy=None, clean=0, tasks="all", t=0, seed=0, src="synthetic", extra=()):
    """Train args for one run."""
    a = list(BASE) + ([noisy] if noisy else []) + list(extra)
    a += ["--sources", f"sim,{src}" if noisy else "sim", "--human_tasks", tasks]
    if clean:
        a += ["--clean_heldout_per_task", str(clean)]
    if t:
        a += ["--ambient_t_min", str(t)]
    if seed:
        a += ["--seed", str(seed)]
        name = f"{name}_s{seed}"
    return name, a


def done(name):
    return os.path.exists(os.path.join(OUT, name, "summary.json"))


def run(job):
    name, args = job
    if done(name):
        return name, "done"
    ck = os.path.join(ROOT, "checkpoints", f"sweep_{name}", "ckpt.pt")
    log = os.path.join(OUT, f"train_{name}.log")
    while not os.path.exists(ck) and os.path.exists(log) and time.time() - os.path.getmtime(log) < 300:
        time.sleep(60)   # being trained by another process (e.g. the previous sweep script)
    if not os.path.exists(ck):
        with open(log, "w") as f:
            r = subprocess.run([PY, "policy/train.py", "--run_name", f"sweep_{name}", *args, "--sim_tasks", "seen",
                                "--sim_per_task", "20"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            return name, f"train failed ({log})"
    with open(os.path.join(OUT, f"eval_{name}.log"), "w") as f:
        r = subprocess.run([PY, "policy/evaluate.py", "--ckpt", ck, "--k", "50", "--out", os.path.join(OUT, name),
                            "--videos", "1"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        return name, "eval failed"
    s = json.load(open(os.path.join(OUT, name, "summary.json")))
    return name, f"seen {s['seen']['success']:.3f} blue {s['heldout']['success']:.3f}"


def best_t(prefix, ts):
    """t_min with the best blue success among finished seed-0 runs <prefix><t> (ties: higher seen)."""
    sc = {}
    for t in ts:
        f = os.path.join(OUT, f"{prefix}{t}", "summary.json")
        if os.path.exists(f):
            s = json.load(open(f))
            sc[t] = (s["heldout"]["success"], s["seen"]["success"])
    return max(sc, key=sc.get) if sc else ts[-1]


def phase(title, jobs, lanes):
    print(f"\n== {title}: {len(jobs)} runs", flush=True)
    with ThreadPoolExecutor(lanes) as ex:
        for name, res in ex.map(run, jobs):
            print(f"{time.strftime('%H:%M')} {name}: {res}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lanes", type=int, default=2)
    p.add_argument("--only", default=None, help="run just this named group (e.g. bprime)")
    a = p.parse_args()
    if a.only == "bprime":   # control: successful phone replays only (is the ambient gain just failed demos?)
        ok = ["--include_failed", "0"]
        phase("B': successful phone replays only", [j for k in (0, 1, 2) for j in (
            cell("K4_Pok", P, clean=4, seed=k, src="human", extra=ok),
            cell("K4_Pok_a75", P, clean=4, t=75, seed=k, src="human", extra=ok))], a.lanes)
        return
    os.makedirs(OUT, exist_ok=True)
    T = [25, 50, 75, 90]
    # 1) the missing comparison for the 0-clean ambient result
    phase("1: 0 clean + noisy, naive", [cell("L2_N0", S, tasks="heldout")], a.lanes)
    # 2) t_min 90 (and the missing 75s) for the 0-, 4- and 24-clean rows
    phase("2: t_min up to 90", [
        cell("L2_Na50", S, tasks="heldout", t=50), cell("L2_Na75", S, tasks="heldout", t=75),
        cell("L2_Na90", S, tasks="heldout", t=90), cell("L2_CNa90", S, clean=4, t=90),
        cell("K24_Na75", S, clean=24, t=75), cell("K24_Na90", S, clean=24, t=90)], a.lanes)
    # 3) 3 training seeds for the key cells (best t_min per row from phase 2)
    t4, t0 = best_t("L2_CNa", [10] + T), best_t("L2_Na", T)
    print(f"best t_min: 4 clean -> {t4}, 0 clean -> {t0}", flush=True)
    phase("3: seeds", [cell(n, *rest, seed=k) for k in (1, 2) for n, *rest in [
        ("C", None, 4), ("L2_CN", S, 4), (f"L2_CNa{t4}", S, 4, "all", t4),
        ("L2_N0", S, 0, "heldout"), (f"L2_Na{t0}", S, 0, "heldout", t0)]], a.lanes)
    # 4) the real phone replays, same cells: t sweep, then seeds for naive and best ambient
    phase("4a: real phone replays", [cell("K4_P", P, clean=4, src="human"),
                                     cell("K0_P", P, tasks="heldout", src="human")]
          + [cell(f"K4_Pa{t}", P, clean=4, t=t, src="human") for t in (50, 75, 90)]
          + [cell(f"K0_Pa{t}", P, tasks="heldout", t=t, src="human") for t in (50, 75, 90)], a.lanes)
    p4, p0 = best_t("K4_Pa", [25, 50, 75, 90]), best_t("K0_Pa", [25, 50, 75, 90])
    print(f"best t_min (phone): 4 clean -> {p4}, 0 clean -> {p0}", flush=True)
    phase("4b: phone seeds", [j for k in (1, 2) for j in (
        cell("K4_P", P, clean=4, seed=k, src="human"), cell(f"K4_Pa{p4}", P, clean=4, t=p4, seed=k, src="human"),
        cell("K0_P", P, tasks="heldout", seed=k, src="human"),
        cell(f"K0_Pa{p0}", P, tasks="heldout", t=p0, seed=k, src="human"))], a.lanes)
    # lower priority: a second noise level (does the best t_min rise with noise?), then the rest
    for L in ("L3", "L1"):
        N = f"data/synth_{L}"
        phase(f"5: noise level {L}", [cell(f"{L}_CN", N, clean=4)]
              + [cell(f"{L}_CNa{t}", N, clean=4, t=t) for t in [10] + T], a.lanes)
    phase("6: 8/12 clean", [cell(f"K{k}", None, clean=k) for k in (8, 12)]
          + [cell(f"K{k}_{v}", S, clean=k, t=t) for k in (8, 12) for v, t in (("N", 0), ("Na50", 50), ("Na90", 90))],
          a.lanes)


if __name__ == "__main__":
    main()
