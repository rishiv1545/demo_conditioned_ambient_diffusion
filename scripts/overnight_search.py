"""Overnight plan (2026-10-10), on ONE Colab VM that is never recreated:

Step 1: wait for P_only / S_only (blue, 3 tasks); works = >= 50% blue success at any evaluated checkpoint.
Step 2: demo-count search on the single task blue-orange from smolvla_base (T1_sim_<n>, nested sets of sim_bo_320):
  a) S_only fails: 80, 160, 320 upward (320 failing = flag: something beyond data quantity);
  b) S_only works, P_only fails: 20, then 10 (or 30 if 20 fails), then the phone run T1_phone_11 (+ T1_sim_10 to match);
  c) both work: as b.
  Then bisect on the grid until the threshold is within a factor of 2.
Budget guard: skip (and stop) when the balance minus the run's estimate would drop below the reserve.
VM drop: no retries; write a note to NOTES.md and stop. At the end: stop the VM, write outputs/overnight/report.md.

    nohup python scripts/overnight_search.py > outputs/overnight/log.txt 2>&1 &
"""
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "outputs", "overnight")
STATE = os.path.join(OUT, "state.json")
DCAD = [sys.executable, os.path.join(REPO, "colab", "dcad.py")]
COLAB = os.path.expanduser("~/.local/bin/colab")
RESERVE = 6.0
RATE = 5.3                       # units per A100 hour (colab usage)
WORKS, PARTIAL = 0.5, 0.2
GRID = [5, 10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320]
STAGES = "data,cache,train,eval"
EXPORT_LOG = os.path.join(REPO, "outputs", "gen_bo.log")
DATASET = os.path.join(REPO, "data", "lerobot", "sim_bo_320")


def log(msg):
    print(time.strftime("%H:%M"), msg, flush=True)


def run(cmd, timeout=None):
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout + r.stderr


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"runs": {}, "branch": None, "why": None, "skipped": [], "flags": [], "done": False}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


def balance():
    rc, out = run([COLAB, "usage"], timeout=120)
    m = re.search(r"Current balance: ([\d.]+)", out)
    return float(m.group(1)) if m else None


def session_alive():
    rc, out = run([COLAB, "sessions"], timeout=120)
    return rc == 0 and any(line.startswith("[dcad] ") for line in out.splitlines())


def estimate_units(name):
    if name.startswith("T1_"):
        steps = 10000 if int(name.rsplit("_", 1)[1]) <= 40 else 15000
        hours = steps / 4.3 / 3600 + 0.1 + 0.15      # train + data/cache/load + 2 evals of 20 episodes
    else:
        hours = 3.0
    return round(hours * RATE * 1.15, 1)


def summaries(name):
    """[(step, success, failure breakdown)] from the fetched eval summaries of a run."""
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "outputs", "colab", name, "eval_step*", "summary.json"))):
        s = json.load(open(f))
        per = s["per_task"]
        outc = next(iter(s["outcomes"].values()), {}) if s.get("outcomes") else {}
        rows.append({"step": s["step"], "success": sum(per.values()) / len(per), "per_task": per,
                     "outcomes": {k: round(v, 3) for k, v in sorted(outc.items(), key=lambda kv: -kv[1])},
                     "k": s["k"], "dir": os.path.basename(os.path.dirname(f))})
    return rows


def best(name, st):
    rows = st["runs"].get(name, {}).get("evals", [])
    return max((r["success"] for r in rows), default=None)


def vm_dropped(st, where):
    msg = (f"\n### Overnight run stopped: Colab VM dropped ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
           f"- While: {where}. No retry (a new VM needs Drive mounted by hand).\n"
           f"- Completed runs: {', '.join(n for n, r in st['runs'].items() if r.get('evals')) or 'none'}; "
           f"state in `outputs/overnight/state.json`, partial report in `outputs/overnight/report.md`.\n"
           f"- To resume: `python colab/dcad.py up`, mount Drive, then `python scripts/overnight_search.py` "
           f"(finished runs are skipped).\n")
    with open(os.path.join(REPO, "NOTES.md"), "a") as f:
        f.write(msg)
    st["flags"].append(f"VM dropped during {where}")
    save_state(st)
    write_report(st)
    log("VM dropped: " + where)
    sys.exit(3)


def wait_job(st, where):
    rc, out = run(DCAD + ["wait", "--every", "3"])
    print(out[-3000:], flush=True)
    first = next((ln for ln in out.splitlines() if "JOB " in ln), "")
    last_state = [ln for ln in out.splitlines() if " JOB " in ln or ln.startswith("JOB ")]
    if rc != 0 and not session_alive():
        vm_dropped(st, where)
    if last_state and "state died" in last_state[-1]:
        vm_dropped(st, where + " (job died)")
    return out


def run_one(st, name):
    if st["runs"].get(name, {}).get("evals"):
        return True
    est = estimate_units(name)
    bal = balance()
    if bal is None:
        log("could not read the balance; continuing with the last known one")
        bal = st.get("last_balance", 0)
    st["last_balance"] = bal
    if bal - est < RESERVE:
        st["skipped"].append({"run": name, "balance": bal, "estimate": est})
        save_state(st)
        log(f"skip {name}: balance {bal} - estimate {est} < reserve {RESERVE}")
        return False
    if not session_alive():
        vm_dropped(st, f"before {name}")
    log(f"launch {name} (balance {bal}, estimate {est})")
    t0 = time.time()
    rc, out = run(DCAD + ["run", name, "--no_sync", "--stages", STAGES], timeout=900)
    print(out[-1500:], flush=True)
    if "started job" not in out:
        if not session_alive():
            vm_dropped(st, f"launching {name}")
        st["flags"].append(f"{name}: launch failed: {out[-300:]}")
        save_state(st)
        return False
    wait_job(st, name)
    after = balance()
    rows = summaries(name)
    st["runs"][name] = {"evals": rows, "hours": round((time.time() - t0) / 3600, 2),
                        "units": round(bal - after, 1) if after is not None else None}
    if after is not None:
        st["last_balance"] = after
    if not rows:
        st["flags"].append(f"{name}: no eval results (see outputs/colab/{name}/job.log)")
    save_state(st)
    write_report(st)
    log(f"{name}: " + ", ".join(f"step {r['step']} {r['success']:.0%}" for r in rows))
    return bool(rows)


def n_of(name):
    return int(name.rsplit("_", 1)[1])


def next_count(st):
    res = {n_of(k): best(k, st) >= WORKS for k in st["runs"] if k.startswith("T1_sim_") and best(k, st) is not None}
    tested = set(res)
    w = sorted(n for n, ok in res.items() if ok)
    hi = w[0] if w else None
    f = sorted(n for n, ok in res.items() if not ok and (hi is None or n < hi))
    lo = f[-1] if f else None
    if hi is not None and lo is not None and hi / lo <= 2:
        return None
    if hi is None:
        seq = [80, 160, 320] if st["branch"] == "a" else [20, 30, 40, 80, 160, 320]
        cand = [n for n in seq if n not in tested and (lo is None or n > lo)]
        if not cand:
            st["flags"].append(f"no tested count works (up to {max(tested)}): something beyond data quantity is wrong")
        return cand[0] if cand else None
    if lo is None:
        cand = [g for g in GRID if g <= hi / 2 and g not in tested]
        return cand[-1] if cand else None
    between = [g for g in GRID if lo < g < hi and g not in tested]
    return min(between, key=lambda g: abs(math.log(g) - 0.5 * math.log(lo * hi))) if between else None


def threshold(st):
    res = {n_of(k): best(k, st) >= WORKS for k in st["runs"] if k.startswith("T1_sim_") and best(k, st) is not None}
    w = sorted(n for n, ok in res.items() if ok)
    hi = w[0] if w else None
    f = sorted(n for n, ok in res.items() if not ok and (hi is None or n < hi))
    lo = f[-1] if f else None
    if hi is None:
        return f"> {lo} demos (no tested count reached {WORKS:.0%})" if lo else "unknown"
    if lo is None:
        return f"<= {hi} demos (the smallest tested count works)"
    return f"between {lo} (fails) and {hi} (works) demos"


def verdict(x):
    return "works" if x >= WORKS else "partial" if x >= PARTIAL else "fails"


def write_report(st):
    lines = [f"# Overnight report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             "| run | demos | steps | checkpoint | success | failure breakdown | units |", "|---|---|---|---|---|---|---|"]
    for name, r in st["runs"].items():
        demos = {"P_only": "38 phone (3 blue tasks)", "S_only": "38 sim (3 blue tasks)"}.get(name, None)
        if demos is None:
            demos = f"{n_of(name)} {'phone' if 'phone' in name else 'sim'} (blue-orange)"
        steps = 20000 if name in ("P_only", "S_only") else (10000 if n_of(name) <= 40 else 15000)
        for i, e in enumerate(r.get("evals", [])):
            fails = ", ".join(f"{k.replace('_', ' ')} {v:.0%}" for k, v in e["outcomes"].items() if k != "success")
            lines.append(f"| {name if i == 0 else ''} | {demos if i == 0 else ''} | {steps if i == 0 else ''} | "
                         f"{e['step']} | {e['success']:.0%} (k={e['k']}/task) | {fails} | "
                         f"{r.get('units') if i == 0 else ''} |")
    lines += ["", f"**Branch:** {st.get('branch')}: {st.get('why')}", f"**Threshold (single task, sim):** {threshold(st)}"]
    if st["skipped"]:
        lines.append("**Skipped (budget):** " + "; ".join(f"{s['run']} (balance {s['balance']}, estimate {s['estimate']})"
                                                          for s in st["skipped"]))
    if st["flags"]:
        lines.append("**Flags:** " + "; ".join(st["flags"]))
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines) + "\n")


def ensure_dataset(st):
    if st.get("dataset_pushed"):
        return True
    while "EXPORT_DONE" not in open(EXPORT_LOG).read():
        if not subprocess.run(["pgrep", "-f", "gen_bo.sh"], capture_output=True).stdout:
            st["flags"].append("sim_bo_320 export failed (outputs/gen_bo.log)")
            save_state(st)
            return False
        time.sleep(60)
    log("pushing sim_bo_320 to Drive")
    rc, out = run(DCAD + ["push-data", DATASET, "--no_sync"], timeout=7200)
    print(out[-800:], flush=True)
    if "on Drive:" not in out:
        if not session_alive():
            vm_dropped(st, "uploading sim_bo_320")
        st["flags"].append(f"sim_bo_320 upload failed: {out[-300:]}")
        save_state(st)
        return False
    st["dataset_pushed"] = True
    save_state(st)
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    st = load_state()
    st["start_balance"] = st.get("start_balance") or balance()
    save_state(st)
    have_data = ensure_dataset(st)              # export + upload while step 1 trains

    # Step 1 (already launched): P_only, S_only
    if not all(st["runs"].get(n, {}).get("evals") for n in ("P_only", "S_only")):
        rc, out = run(DCAD + ["status"], timeout=300)
        if "state running" not in out:          # resuming after a VM drop: relaunch (training resumes from Drive)
            if not session_alive():
                vm_dropped(st, "before relaunching P_only/S_only")
            rc, out = run(DCAD + ["run", "P_only", "S_only", "--no_sync"], timeout=900)
            print(out[-1500:], flush=True)
            if "started job" not in out:
                vm_dropped(st, "relaunching P_only/S_only")
        wait_job(st, "P_only/S_only")
        for n in ("P_only", "S_only"):
            st["runs"][n] = {"evals": summaries(n), "hours": None, "units": None}
        bal = balance()
        if bal is not None and st.get("start_balance"):
            st["runs"]["P_only"]["units"] = round((st["start_balance"] - bal) / 2, 1)   # shared job: split evenly
            st["runs"]["S_only"]["units"] = st["runs"]["P_only"]["units"]
        save_state(st)
        write_report(st)
    p, s = best("P_only", st), best("S_only", st)
    if p is None or s is None:
        st["flags"].append("step 1 has no eval results; branch a by default")
        p, s = p or 0.0, s or 0.0
    if s >= WORKS and p >= WORKS:
        st["branch"], st["why"] = "c", f"both work (P_only best {p:.0%}, S_only best {s:.0%})"
    elif s >= WORKS:
        st["branch"], st["why"] = "b", f"S_only works ({s:.0%}), P_only {verdict(p)} ({p:.0%})"
    else:
        st["branch"], st["why"] = "a", f"S_only {verdict(s)} ({s:.0%}), P_only {verdict(p)} ({p:.0%})"
    save_state(st)
    log(f"branch {st['branch']}: {st['why']}")

    # Step 2: single-task demo-count search
    if have_data:
        while (n := next_count(st)) is not None:
            if not run_one(st, f"T1_sim_{n}"):
                break
        if st["branch"] in ("b", "c"):
            if run_one(st, "T1_phone_11") and not st["runs"].get("T1_sim_10", {}).get("evals"):
                run_one(st, "T1_sim_10")             # the sim count matching the 11 phone replays
    st["done"] = True
    save_state(st)
    write_report(st)
    log("all done; stopping the VM")
    rc, out = run(DCAD + ["down"], timeout=2400)
    print(out[-500:], flush=True)
    st["end_balance"] = balance()
    save_state(st)
    write_report(st)


if __name__ == "__main__":
    main()
