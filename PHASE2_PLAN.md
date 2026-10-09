# Phase 2 plan: SmolVLA post-trained with ambient flow matching

## Why the change

The challenge asks us to show VLA/world-model knowledge; a small state-based policy doesn't. The headline model is now **SmolVLA** (`lerobot/smolvla_base`, about 450M params) post-trained in our MuJoCo task, from images, robot state and a language instruction. The Phase 1 diffusion policy stays as a **mechanism study** in the README:
- the encoder-shortcut diagnosis (10 cm probe, rejected copycat hypothesis, bilinear fix with before/after numbers), and
- the ambient-loss sweep on synthetic noise (step 4b), which also picks t_min for the VLA runs.

Training runs on a **Colab A100**, driven from the Mac (`colab/dcad.py`, run registry `colab/runs.json`). No AWS. The small-policy sweeps run on the Mac (MPS), costing no Colab units.

## Main question (reframed 2026-10-09)
Not "phone demo vs robot demo", but: **can noisy demonstrations improve post-training when clean data is scarce, and does the ambient loss make them usable?** Phone replays are the realistic noise source. The scarce-data case is the held-out blue cube: the sim-only model (V0) does not ground "blue" at all (probe: blue accuracy 0.03–0.12 from 12.5k to 30k steps, below chance, while red/green reach 0.8–0.94).

## Task and split (unchanged)
- 3 cubes, 3 zones, 9 tasks. **Object split:** the blue cube is never a target in sim demos (but is a distractor in every scene). Seen = the 6 red/green tasks; held out = blue-yellow, blue-purple, blue-orange.
- Phone demos: 4 per red/green task, 8 per blue task (48 in total), retargeted and replayed in sim, then **re-rendered in sim** for SmolVLA.

## Steps

### 1. Image observations
- `PickPlaceEnv` gains rendered camera observations. Start with **one top-down camera** roughly matching the phone viewpoint (about 0.75 m above the workspace, looking down) at a low resolution (256×256; SmolVLA pads to 512 internally). Add a wrist camera only if one camera clearly isn't enough, and record the evidence.
- Robot state for SmolVLA: `observation.state` = EE position (3) + gripper width (1), matching the action space (absolute EE target xyz + gripper). No privileged object positions: the policy has to find the cubes in the image.
- Language instruction per task: `"put the <cube> cube in the <zone> zone"`.

### 2. LeRobot datasets
- Export sim demos (red/green tasks) and phone replays (all 9 tasks, re-rendered in sim) as LeRobot datasets (10 fps, video-encoded images), keeping the object split.
- A per-frame `sigma_n` feature: 0 for sim, > 0 for phone-derived frames.
- Store on Google Drive (or the HF Hub) so Colab can load them.

### 3. Ambient flow matching
- SmolVLA's flow matching: `x_t = t·noise + (1 − t)·actions`, so t = 1 is pure noise. The default draws t ~ Beta(1.5, 1)·0.999 + 0.001.
- Ambient constraint: samples with `sigma_n > 0` get `t ← t_min + (1 − t_min)·t`, so phone-derived samples only supervise the noisy end. This mirrors the DDPM `t_min` hook.
- Implemented in our training loop by passing `time=` to `SmolVLAPolicy.forward` (no LeRobot source patch), behind `--ambient_t_min` (0 = off), with a unit test.

### 4. Runs
**4a. VLA runs** (same recipe and steps as V0 = `V0_2cam_unfrozen`: VLM language layers unfrozen at lr 1e-5, expert 1e-4 constant, batch 32, 15k steps; the 30k continuation showed no further probe gain after 15k). Red/green sim data (`sim_seen_v2c_100`, 100/task) in all runs:

| run (`runs.json`) | blue data | loss |
|---|---|---|
| V0 (`V0_2cam_unfrozen`) | none | standard |
| **C** (`C_blue4`) | 4 clean sim demos per blue task | standard |
| **C+N** (`CN_blue4`) | C + all phone replays (4 per red/green task, 8 per blue task) | naive |
| **C+N+amb** (`CNamb_blue4`) | same as C+N | ambient, t_min from 4b |
| **N+amb** (`Namb`) | phone replays for blue only, no clean blue | ambient |
| **Ceiling** (`Ceil_blue24`) | 24 clean sim demos per blue task | standard |

- The number of clean blue demos is `clean_blue_per_task` in `runs.json` (first N per task of `sim_blue_v2c_24`); `phone: all | blue` picks the phone replays; `ambient: true` uses `defaults.ambient_t_min`.
- Report **seen and blue success** for each (fixed eval seeds, 20 episodes/task intermediate, 50 final, Wilson 95% CIs), plus the per-cube grounding probe.

**4b. Small-policy sweep (Mac)** (`scripts/run_ambient_sweep.sh`, table + t_min: `scripts/summarize_ambient_sweep.py`): the same C / C+N / C+N+amb / N+amb / Ceiling structure on the small diffusion policy (20 sim demos per red/green task, 4 clean per blue task), with synthetic phone-like noise (per-episode xy/z offset, jitter, gripper timing) at three levels (0.4×, 1× = 2.5 cm xy as measured on phone replays, 2×) and t_min ∈ {10, 25, 50, 75} of T = 100. t_min for the VLA = the best C+N+amb setting at the 1× level, mapped to flow-matching time by matching the noise-to-signal ratio (DDPM √(1−ᾱ_t)/√ᾱ_t = τ/(1−τ); t = 25 → τ = 0.31, 50 → 0.51).

### 5. Colab practicalities (as built)
- A100 via the colab CLI: `python colab/dcad.py run <runs...>` runs data → vision cache → train → probe → eval detached, all state on Drive, resumable; `wait --stop` fetches results and stops the VM (Drive is flushed first). Details in NOTES.md ("Colab pipeline").
- Frozen-VLM training (SmolVLA default) did not ground language; unfreezing the language layers (SigLIP + connector frozen, so the vision cache stays valid) is the recipe.
- 0.187 s/step on the A100 (15k steps ≈ 47 min); eval renders only when a new action chunk is planned.

### 6. Stretch (only after the 4a runs): demo-prompted SmolVLA
- Replace the language instruction with 2–3 keyframe images (start, grasp, release) from a **different** demo of the same task, fed through SmolVLA's existing multi-image input. Train with prompts drawn from other episodes of the same task (the Phase 1 context sampler).
- Evaluate with prompts from (a) sim renders, (b) phone-replay renders, (c) raw phone frames.
- Fallback if images are too slow on Colab: serialize the demo trajectory as about 16 text waypoints in the instruction.

## Limitations
- The phone data comes from a **controlled, marker-calibrated setup**: a fixed top-down camera at a measured height, ArUco markers giving metric table coordinates in every frame, known object heights, a flat uncluttered table, one hand, and an exaggerated pinch protocol. Its noise (≈ 2–4 cm grasp offsets, gripper timing) therefore **understates the gap to in-the-wild human video** (moving or unknown cameras, no metric scale, occlusion, clutter, different embodiments and grasps). A positive result shows that moderately noisy demonstrations can be made useful, not that arbitrary human video can.
- Phone demos are retargeted and replayed in sim, then re-rendered: the VLA never sees real phone pixels, so this tests action noise, not the visual domain gap.
- One task family (pick-and-place of 3 cubes into 3 zones), one held-out object, and single training seeds for the VLA runs.

