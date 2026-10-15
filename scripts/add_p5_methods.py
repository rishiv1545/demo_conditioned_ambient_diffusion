"""python scripts/add_p5_methods.py N : add the method-comparison runs at N phone demos to colab/runs.json, print the run list."""
import json, sys
N = int(sys.argv[1])
p = "colab/runs.json"; r = json.load(open(p)); R = r["runs"]
ZN = "--tasks blue-red --noise_scale 0"; VAL = ZN + " --seed_base 5000000"
steps = 10000 if N <= 30 else 15000
base = f"--unfreeze_vlm 1 --vlm_lr 1e-5 --steps {steps} --lr_schedule cosine --warmup 500 --save_every 2500"
common = {"eval_k": 20, "eval_steps": list(range(2500, steps + 1, 2500)),
          "eval_variants": [["_br_zn", ZN], ["_br_zn_val", VAL]]}
fixed = (f"Same recipe as P5_naive_{N} (cosine over {steps // 1000}k, seed 0) with checkpoints every 2.5k, evaluated at "
         "every fixed step on the standard seeds (_br_zn, reported) and on validation seeds (_br_zn_val, seed_base "
         "5,000,000; selection only)")
naive = "P5_naive_15r" if N == 15 else f"P5_naive_{N}r"
new = {}
if N != 15:
    new[naive] = {"data": [f"phone_s5?per_task={N}"], "train": base, **common,
                  "_note": f"naive at N={N}, the curve reference for the method comparison. {fixed}"}
new[f"P5_amb_{N}"] = {"data": [f"phone_s5?per_task={N}"],
                      "train": base + f" --episode_info quality/phone_s5_first{N}.json --ambient_per_episode 1", **common,
                      "_note": f"ambient at N={N}: per-episode t_min from the quality scores (clean = the top 30% of these N clips by quality score, quality/phone_s5_first{N}.json). {fixed}"}
new[f"P5_qual_{N}"] = {"data": [f"phone_s5?per_task={N}"],
                       "train": base + f" --episode_info quality/phone_s5_first{N}.json --quality_tag 1", **common,
                       "_note": f"quality-tagged at N={N} (eval asks for high quality). {fixed}"}
new[f"P5_naive_{N}_vn"] = {"data": [f"phone_s5_vn?per_task={N}"], "train": base, **common,
                           "_note": f"naive at N={N} on the velocity-normalized replays (phone_s5_vn, human/velocity_norm.py). {fixed}"}
new[f"P5_SI_{N}"] = {"stages": ["data", "cache", "self_improve"], "eval_k": 20,
                     "self_improve_best": {"candidates": [naive, f"P5_amb_{N}", f"P5_qual_{N}"], "tag": "_br_zn_val",
                                           "task": "blue-red", "rounds": 1, "n": 100, "steps_per_round": 3000,
                                           "eval_k": 20, "eval_tag": "_br_zn"},
                     "_note": f"one self-improvement round on the best of naive / ambient / quality at N={N}, chosen on the "
                              "validation seeds at the final checkpoint (_br_zn_val); reported on the standard seeds (_br_zn)"}
for k in new:
    assert k not in R, f"{k} exists"
R.update(new)
open(p, "w").write(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
order = ([] if N != 15 else []) + list(new)
print(" ".join(["P5_naive_15", "P5_naive_15r"] + [k for k in order if k != "P5_naive_15r"]))
