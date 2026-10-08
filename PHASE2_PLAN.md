# Phase 2 plan: SmolVLA post-trained with ambient flow matching

## Why the change

The challenge asks us to show VLA/world-model knowledge; a small state-based policy doesn't. The headline model is now **SmolVLA** (`lerobot/smolvla_base`, about 450M params) post-trained in our MuJoCo task, from images, robot state and a language instruction. The Phase 1 diffusion policy stays as a **mechanism study** in the README:
- the encoder-shortcut diagnosis (10 cm probe, rejected copycat hypothesis, bilinear fix with before/after numbers), and
- an ambient-loss validation on synthetic noise. **Not done yet:** this has to be run on the small policy (corrupt a fraction of sim demos with known noise, compare the naive loss with t_min).

Training runs on **Google Colab** (free T4 first; Colab Pro L4/A100 if the timing says so). No AWS. Development and evaluation also run locally on the M4 Pro.

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
- **V0**: sim only. **V1**: sim + phone, naive. **V2**: sim + phone, ambient.
- Evaluate seen vs held-out (blue) success with the same protocol: fixed eval seeds, 20 episodes per task for intermediate checks, 50 for final numbers, Wilson 95% CIs.
- **V0 on blue tasks = the zero-shot language-grounding check** (does "blue" transfer from pre-training without any blue demos?).

### 5. Colab practicalities
- Keep SmolVLA's default frozen vision encoder; train the action expert as the defaults do (`train_expert_only=True`).
- Don't assume 20k steps. Find the step count where held-out success stops improving (start around 2–5k), evaluating saved checkpoints.
- Checkpoint to Google Drive frequently (model, optimizer, scheduler, step, RNG) and support `--resume`, so a disconnect costs minutes.
- `colab/train_smolvla.ipynb`: clone the repo, install, mount Drive, train/eval a chosen config (V0/V1/V2).
- **First: time a 500-step fine-tune on the Mac (MPS) and on a Colab T4**, and project the time for the full V0–V2 plan, so the user can decide on Colab Pro.

### 6. Stretch (only after V0–V2): demo-prompted SmolVLA
- Replace the language instruction with 2–3 keyframe images (start, grasp, release) from a **different** demo of the same task, fed through SmolVLA's existing multi-image input. Train with prompts drawn from other episodes of the same task (the Phase 1 context sampler).
- Evaluate with prompts from (a) sim renders, (b) phone-replay renders, (c) raw phone frames.
- Fallback if images are too slow on Colab: serialize the demo trajectory as about 16 text waypoints in the instruction.
