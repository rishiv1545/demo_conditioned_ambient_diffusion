# NOTES: running log

Design decisions, failures and measured numbers. Every number here comes from a script in the repo.

## Platform (changed 2026-10-08)
- **Main path is local**: MacBook with M4 Pro (macOS 26, arm64, 12 cores). Conda env `dcad` with Python 3.11 (a plain venv works too; see the README). Colab is now only a one-click reproduction notebook for reviewers (`colab/run_all.ipynb`: sets `MUJOCO_GL=egl`, trains on cuda). Previously the plan had training and evaluation on a Colab T4.
- `requirements.txt` pins exactly what installed and imported on macOS arm64: mujoco 3.15.0, torch 2.14.1, mediapipe 1.1.0, opencv-contrib-python 5.0.0.93, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, imageio 2.38.0 (+ imageio-ffmpeg 0.6.0), tqdm 4.70.1, pytest 9.1.1.
- Device: `--device auto` picks cuda → mps → cpu. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in train/evaluate.
- **MPS vs CPU timing** (`policy/train.py`, 300 steps, batch 256, 4.82M params, 2-frame obs, 120 sim episodes): **mps 17 s (57 ms/step), cpu 259 s (860 ms/step)**. MPS is about 15× faster, so `auto` (→ mps) stays the default on the Mac. 20k steps take about 18–19 min on MPS, inside the 30-minute target.
- Rendering: `MUJOCO_GL=egl` is set only on Linux (`sim/env.py`); macOS uses MuJoCo's default offscreen (CGL) renderer.
- Multiprocessing: all pools use the `spawn` context explicitly (same behavior on macOS and Linux). Worker functions are module-level and scripts are `__main__`-guarded. `--workers` defaults to cores − 2.
- Evaluation runs the policy on CPU in each worker (1 torch thread per worker): batch-1 inference on a small U-Net. That's cheaper than shipping observations to one MPS process.

## Milestone 0
- `scripts/make_markers.py` → `assets/markers.pdf`: DICT_4X4_50, IDs 0–3, 6 cm (the black square including the 1-bit border), two per A4 page. Checked by rasterizing the pages and re-detecting them with cv2.aruco: all 4 detected, measured side ≈ 5.97–5.98 cm at 100 dpi.
- Table frame convention (shared by sim and human): origin at the center of marker 0 (near-left), +x toward marker 1 (near-right, 50 cm), +y toward marker 3 (far-left, 35 cm). HOME is near-right. The sim world frame **is** this table frame (z = 0 at the table top), so retargeting is the identity plus optional offsets.
- The calibration is two separate clips (`calib_table.mp4`, `calib_box.mp4`) rather than one clip that has to be segmented. It's simpler and more robust.

## Milestone 1: sim
- **Why a custom MuJoCo scene + menagerie Panda (not LIBERO/robosuite):** the task family needs full control of objects, zones, randomization and replaying human layouts, plus fast headless rollouts on a laptop CPU. A 1-file scene with direct MuJoCo calls is simpler than adapting a benchmark.
- Vendored `mujoco_menagerie/franka_emika_panda` (Apache-2.0; LICENSE kept). `panda.xml` is **modified**: meshdir relative to `assets/scene.xml`, base pose on link0, `ee` site at the fingertip center (0.1034 m below the hand). Unused menagerie files removed.
- **Robot base at (0.25, −0.30, 0), facing +y.** At first it was at y = −0.20, and the expert failed when a cube sat right in front of the base (x≈0.25, y≈0.04): joint 4 hit its fold limit (−3.07 rad), leaving 13 cm of error. Moving the base back 10 cm makes the whole workspace reachable top-down: worst error 3.2 mm over a grid including 2 cm outside the rectangle, z = 0.018.
- **Gravity compensation**: the menagerie position actuators sagged 5–10 mm under gravity. I added `qfrc_applied = qfrc_bias` on the arm joints each substep (the real Panda does gravity compensation internally). Tracking error at hold dropped to 0.05 mm, and to under 2 mm after 4 s moves to the far corners.
- Control: 10 Hz, physics dt 0.002 s → 50 substeps; joint targets are linearly interpolated across substeps. IK is damped least squares (λ = 0.05) on the `ee` site's position + orientation (pointing down, fingers opening along x), 15 iterations per control step from the current q, with a nullspace pull toward the home posture. The EE target moves at most 5 cm per step (0.5 m/s).
- Gripper: menagerie tendon actuator, stiffened from kp = 100 to kp = 400 (force ≈ 8 N on a 4 cm cube versus 2 N by default). Cubes 4 cm, 50 g, friction 1.5, condim 4, elliptic cones, impratio 10. With these values grasps never slipped in 900 test episodes.
- Layout sampling: zones Chebyshev ≥ 12 cm apart, cubes ≥ 9 cm apart (Euclidean), cubes Chebyshev ≥ 9 cm from every zone center (a cube never starts in a zone).
- Success: the target cube center lies within the target zone square (|dx|, |dy| ≤ 5 cm), the cube center is below 2.5 cm (resting), and the gripper width is over 4.5 cm (released).
- Expert: an open-loop waypoint plan made at reset (hover → descend → close 0.5–0.7 s → lift → move → descend → open → retreat). Randomized speed 0.15–0.3 m/s, slower descents, hover heights 7–13 / 9–15 cm, ±4 mm grasp jitter, ±12 mm placement jitter, random 0.05–0.3 s pauses.
- **Results** (`scripts/eval_expert.py --n 100`, seeds 500000+): **100/100 success on every one of the 9 tasks (900/900)**, 28 s with 12 workers. Headless speed: 457 control steps/s single process (a 200-step rollout takes 0.44 s). Videos: `outputs/m1/expert_*.mp4` (front + top view).
- Rendering colors: the first lighting washed out orange to yellow and purple to pink, so I reduced the headlight and directional light and darkened orange and purple.
- Data: `scripts/gen_sim_data.py --n_per_task 50 --tasks all` → 450 successful episodes in 17 s (each episode is about 60–90 steps). Training takes the first `--sim_per_task` (default 20) per task. Held-out-task sim data exists only for an optional oracle run; baselines A/B/B′ never load it.

## Milestone 2: phone pipeline (code ready, waiting on real clips)
- Homography: per-frame ArUco centers → gaps linearly interpolated → 9-frame temporal median → per-frame `getPerspectiveTransform`. Marker table coordinates come from the 4 measured sides + 2 diagonals (circle intersections).
- Height: the hand size is measured **in table-plane meters** (landmarks mapped through the homography), which removes camera tilt and lens position from the scale. Size = mean(wrist→middle MCP, index MCP→pinky MCP). z = H·(1 − s₀/s), with s₀ from `calib_table`. `calib_box` gives the reported height error. xy gets a parallax correction toward the table point under the image center.
- Gripper: aperture = |thumb tip − index tip| / hand size, hysteresis close < 0.65 / open > 0.9, then a 0.2 s minimum-duration filter. Thresholds are set from geometry (a pinched 4 cm cube ≈ 0.5, a spread hand > 1.2) and will be tuned on real clips.
- Objects: HSV thresholds on the warped first frame (largest blob ≥ 4 cm² per color). Cube centroids are parallax-corrected for z = 2 cm. The last frame is checked to label whether the human completed the task.
- Retarget: 1:1, zero xy offset, `z_offset` (default 0), z ≥ 1.2 cm, resampled to 10 Hz. Replay starts the arm at the first retargeted point, runs open-loop and appends 5 hold steps. Failures are kept and flagged (`success=False`).
- Videos are downscaled to 960 px wide on read (a 15 s 1080p clip would take about 3 GB of RAM).
- **Household objects instead of colored cubes (2026-10-08).** The human uses household stand-ins, so:
  - `session.json` maps each canonical task name (red/green/blue objects, yellow/purple/orange zones) to a real item (`label`, `height_cm`) and its HSV range(s). Clip file names keep the canonical task names. Missing HSV entries fall back to the defaults in `human/objects.py`.
  - `scripts/calib_colors.py`: one click per item on a top-down frame. Each click samples a 17×17 px patch (about 1 cm), and the HSV range is fit as a circular hue mean ± (95th-percentile deviation + 6), with S and V bounded **on both sides** (5th/95th percentile ± 40). `--check_all` reports per clip whether all 6 items are found. A first version bounded S/V only from below, so a gray item matched the whole table in the synthetic test. Now weakly saturated items (median S < 50) get the full hue range and are separated by S/V alone.
  - `scripts/click_layout.py`: the manual fallback. Click the 3 object tops and 3 zone centers on each clip's top-down first frame. Object clicks are parallax-corrected with `height_cm`. The result goes in `<clip>.layout.json`, which overrides color detection in `process_phone.py` (recorded as `layout_source` in the CSV and the episode meta).
  - Grasp height: the human pinches an object at about half its height, while the sim cube's center is at 2 cm. Grasping and placing both happen at about h/2, so each clip's trajectory is shifted in z by −(h_target/2 − 2 cm). This is stored as `z_offset` in the episode meta.
  - Tests: hue wrap-around for red, synthetic non-default colors (pink "red", cyan "yellow", gray "purple") recovered from one click each within 6 mm while the defaults fail, click→table conversion with parallax, and the layout JSON round-trip.
- **Height-from-hand-size assumes a roughly top-down camera.** The homography makes xy correct at any camera angle, and measuring hand size in table-plane meters cancels the *table-plane* perspective. But the model s = L·H/(H − z) treats the camera as a point at height H above the measured hand, with the hand's rigid segment parallel to the table. With a tilted camera, the effective distance to the hand varies across the workspace and the hand segments get foreshortened differently, so z picks up a position-dependent bias. `camera_height_cm` should be the vertical lens height. The `calib_box` height error is reported in `outputs/m2/<session>/calibration.json` either way, and goes in the results whatever it is.
- Tests on synthetic inputs (`tests/test_human.py`): a perspective-warped canvas with markers and colored squares (objects recovered within 6 mm), height/parallax round-trip, gripper cleanup, gap filling, and an expert path turned into a "human" one (30 fps, 2–3 mm noise, gripper 0.1 s late) whose retarget and replay succeed. **Not yet run on real hand video.**

## Milestone 3: policy
- 1D temporal U-Net (Chi et al. style), dims (64, 128, 256), kernel 5, GroupNorm, Mish, FiLM from [cond, timestep embedding]: **4.82M params**. Action chunk H = 16, run 8 actions then re-plan. Observation history of 2 frames.
- `ConditionEncoder` interface. Phase 1 `ObsTaskEncoder`: separate MLPs for the observation (2×19 → 128) and the task one-hot (6 → 64), concatenated.
- Hand-rolled DDPM: T = 100, cosine schedule, ε-prediction; DDIM with 10 steps (η = 0, x0 clipped to [−1, 1]). `loss(model, x0, cond, t_min=None)` has the per-sample t_min hook.
- Normalization: per-dimension min/max → [−1, 1] from the training set, stored in the checkpoint. Observation stats are per frame.
- Training: AdamW lr 3e-4, wd 1e-4, 500-step warmup + cosine, batch 256, grad clip 1, EMA 0.999, 20k steps. The whole dataset is held as tensors on the device.
- Evaluation: 50 episodes per task on fixed seeds (1,000,000 + task·10,000 + k), never used for data. Success = `env.success()` holds for 5 consecutive control steps (stable placement and release), else failure at 200 steps (20 s). Wilson 95% CIs on the pooled seen and held-out rates.
- Bug found: the eval runner normalized the stacked 2-frame observation with per-frame stats (shape error). Fixed by normalizing each frame.

### Experiment log (Milestone 3)
- **A, 20 sim demos/task (120 episodes), 20k steps** (`checkpoints/A_concat_combo_n20`, combination split): training took 1077 s on MPS (18 min), final loss 0.0019. Evaluation (`outputs/m3/exp/A_concat_combo_n20`, 50 eps/task, 71 s with 10 workers): **seen 1.0% [0.3, 2.9], held-out 0.7% [0.1, 3.7]**.
  - Diagnosis: closed-loop on **training layouts** (15 episodes) it scores **100%**. Offline, its chunk predictions on training samples match the demos to within a few mm and barely change when the task one-hot is swapped. It memorized layout → trajectory and never learned "go to the cube named by the task". In videos with new layouts it confidently picks some cube and places it somewhere in between zones.
  - Next: the same model with 200 demos/task (1200 episodes) to test whether data volume alone fixes it.
- **A, 200 sim demos/task (1200 episodes), concat encoder, 20k steps** (`checkpoints/A_concat_combo_n200`, combination split): 1072 s, final loss 0.0153. **Seen 1.7% [0.7, 3.8], held-out 0.0% [0, 2.5]**. Ten times the data didn't help, so memorization wasn't the root cause.
  - Failure mode, 12 closed-loop episodes on new layouts: the gripper closes 0.6–24 cm away from the target cube; in 11 of 12 nothing is lifted.
  - Probe: shifting the target cube 10 cm in the observation moves the predicted chunk endpoint by only 0.3–2.5 cm, and swapping the task one-hot barely changes it. At t = 0 the predicted reach point is roughly the mean of the three cubes. The concat encoder never learned the multiplicative "select the coordinates of the cube named by the one-hot".
  - **Copycat hypothesis, tested and rejected.** The first probe (t = 12, mid-reach) suggested a copycat shortcut: with 2-frame observations, the EE's own recent motion predicts the next chunk, so the cube positions might be ignored. A second probe at **t = 0**, where both frames are identical and there's no motion to copy, showed the same insensitivity: the target cube shifted 10 cm moved the prediction 1.7–2.5 cm, and the three task one-hots gave reach points within 2 cm of each other, near the mean of the cubes. So the root cause is the missing multiplicative selection, not copying. (A copycat effect may still exist on top of it; it wasn't isolated, e.g. with `--n_obs 1`.)
  - Fix: `ObsTaskEncoder(bilinear=True)` adds an MLP over the outer product task ⊗ obs (6 × 38 = 228 features), so the selected object's coordinates are linear in the input. The obs and task branches stay separate, as Phase 2 requires. 5.37M params.
- **Quick test, bilinear, 200/task, 8k steps** (`checkpoints/A_bil_combo_n200_8k`, combination split, k = 10): **seen 60/60, held-out 30/30**.
  - ⚠️ **Consequence for the project's premise:** an encoder that factors into cube × zone composes the held-out (cube, zone) pairs for free from sim data alone. With one-hot conditioning, the held-out split no longer needs phone data. Flagged for a decision (see the end of this file / the summary).

## Split change: hold out an object, not combinations (2026-10-08)
- **Finding (combination split):** with the bilinear (factored) encoder, sim-only A composes the held-out (cube, zone) combinations for free: 100% held-out in the 8k-step test above. Holding out combinations therefore can't motivate phone data. This stays available as `--heldout combo`.
- **Combination split, full run** (`checkpoints/A_bil_combo_n20`: bilinear, 20 demos/task, 20k steps, k = 50): **seen 89.0% [85.0, 92.1], held-out 61.3% [53.3, 68.8]** (red-yellow 0.60, green-purple 0.58, blue-orange 0.66). Even with 20 demos/task, composition gives most of the held-out performance. (The 200/task combination run was stopped once the split changed.)
- **New default split, `--heldout object`:** the blue cube is never the target in a sim demo. Seen in sim = the 6 red and green tasks; held out = blue-yellow, blue-purple, blue-orange. The blue cube is still in every sim scene as a distractor (same observation; `tests/test_env.py::test_object_split_blue_is_distractor` checks that it sits on the table, unmoved, through red and green demos). Also available: `object:<cube>`, `zone:<zone>`, or an explicit task list. `evaluate.py` defaults to the split stored in the checkpoint.
- Recording plan: 4 phone clips for each red/green task and 8 for each blue task (48 in total).
- The bilinear encoder is now the default (`--bilinear 1`). Checkpoints from before the option load as concat.

### Baseline A under the object split (canonical `checkpoints/A`, `outputs/m3/A`)
- Bilinear encoder, 20 sim demos for each of the 6 red/green tasks (120 episodes), 20k steps: 1110 s on MPS, final loss 0.0015. Evaluation: 450 episodes in 48 s.
- **Seen (red/green) 88.7% [84.6, 91.8]; held out (blue) 0.0% [0.0, 2.5]** (0/150). Per task: red-yellow 0.94, red-purple 0.86, red-orange 0.84, green-yellow 0.96, green-purple 0.86, green-orange 0.86; blue-* 0.00.
- Held-out is near 0% as intended. Behavior on blue tasks (15 probe episodes): it lifts nothing in 14 and lifts the red cube in 1, never the blue cube. The unseen blue one-hot leaves it with no learned target, so it hovers.
- Seen-task failures (~11%) are left for later; videos are in `outputs/m3/A/`.

# Phase 2: SmolVLA (see PHASE2_PLAN.md)

## Environment
- `lerobot[smolvla]==0.4.4` is installed into the same `dcad` env. It caps torch < 2.11, so torch went 2.14.1 → 2.10.0 (and numpy 2.4.6 → 2.2.6). It also pulls `opencv-python-headless`, whose `cv2` has no GUI; the click tools need one, so `opencv-contrib-python` at the **same** version (4.12.0.88) is installed after it (Cocoa GUI, ArUco included). `pip check` is clean and all tests pass after the switch. The pre-switch freeze is kept outside the repo.
- macOS warns that PyAV's bundled FFmpeg and Homebrew's FFmpeg (loaded by torchcodec) define the same Objective-C classes. No failures observed so far.
- **Fixed: LeRobot padding-key mismatch.** `SmolVLAPolicy.forward` reads `batch["actions_id_pad"]` (sic), but datasets provide `action_is_pad`, so upstream training never masks padded chunk tails (copies of the episode's last action) out of the loss. `vla/train_smolvla.py` sets `batch["actions_id_pad"] = batch["action_is_pad"]` after preprocessing, so padded tails are now masked. `tests/test_feature_cache.py` checks that `action_is_pad` matches LeRobot's on a padded sample.

## Image observation (step 1)
- A single `phone` camera at 256×256 (SmolVLA pads to 512). Render cost is about 7 ms/frame on the Mac.
- **A straight-down camera doesn't work:** at both 0.75 m and 1.2 m, the arm hides most of the workspace during reach and carry (the elbow fills the middle of the image), and colors wash out under the overhead light. I compared 5 viewpoints at grasp and carry moments. Chosen: **0.95 m up, beyond the far edge, tilted 22° from vertical** (`assets/scene.xml`). All 6 items stay visible, the gripper and target are visible at grasp and place, and colors are correct. This view is rotated 180° compared with the human phone clips (far edge at the bottom), which only matters for stretch goal (c), raw phone frames. One camera looked sufficient on inspection; a wrist camera is deferred until a failure analysis says otherwise.
- `observation.state` = [EE xyz, gripper width]; action = [EE target xyz, gripper]. No privileged object positions: SmolVLA has to find the cubes in the image. Instruction: "put the <cube> cube in the <zone> zone".

## LeRobot export (step 2)
- `vla/export_lerobot.py` re-simulates each stored episode from its layout and actions, rendering the camera. Replay fidelity: max |obs − stored obs| = 6e-5 over 18 test episodes. `sigma_n` is a per-frame feature (0 sim, `--phone_sigma` for phone). It's not an `observation.*` key, so SmolVLA never sees it as an input.
- Video codec H.264 (LeRobot's default is AV1). With 4–8 workers, loading takes about 0.01–0.02 s per batch of 32 with either codec, so data loading is not the bottleneck. Export runs at about 2.7 s/episode.

## Ambient flow matching (step 3)
- SmolVLA: x_t = t·noise + (1 − t)·actions (t = 1 is pure noise), with t ~ 0.001 + 0.999·Beta(1.5, 1). `vla/ambient.py` draws the same distribution (inverse CDF, so it takes a seeded generator) and maps samples with sigma_n > 0 to t_min + (1 − t_min)·t. The times go in via `SmolVLAPolicy.forward(batch, time=...)`: **no LeRobot source patch**. `--ambient_t_min 0` is off and bit-identical to the default sampler. Tests: `tests/test_ambient_vla.py` (distribution match, off switch, restriction).

## Timing (step 5): Mac, M-series with 18 GB unified memory, MPS
- Fine-tuning setup as the defaults: frozen vision encoder, `train_expert_only`. Parameter counts: see the 500-step run below.
- Per training step (profiled with a standalone script on the 18-episode H.264 smoke dataset):

  | micro-batch | forward | backward + AdamW | per sample |
  |---|---|---|---|
  | 8 | 0.89 s | 0.21 s | 0.14 s |
  | 16 | 1.8–2.0 s | 0.40 s | 0.14 s |
  | 32 | **35–53 s** | 0.9 s | memory thrash (doesn't fit in 18 GB) |
  | 16, fp32 | 2.4–2.9 s | 0.45 s | fp32 is slower than the mixed bf16 default |

  So the Mac runs at **about 7 samples/s** at micro-batch 16. The forward pass dominates (frozen SigLIP on images padded to 512 px plus the frozen VLM layers). The default is micro-batch 16 × grad_accum 2 (effective 32), about 4.5 s/step, so 500 steps ≈ 38 min.
- An earlier smoke run at micro-batch 32 measured 24 s/step: that was the memory thrash.
- A possible speedup if needed: cache frozen image features per frame (64 tokens × 960 dims, fp16 ≈ 123 KB/frame).

## Frozen-vision-feature cache (`vla/feature_cache.py`)
- Valid because `train_expert_only=True` freezes the **whole** VLM, including SigLIP and the connector, and we use no image augmentation. Cached: `embed_image(resize_with_pad_512(img)·2 − 1)` per frame, fp16, 64 tokens × 960 dims. Everything after it (√dim scaling, language/state tokens, VLM layers, expert) runs live. `use_cached_features(policy)` patches only that policy instance; **eval always uses raw images**. `--features raw` trains from raw images (needed for the stretch goal's image prompts).
- **Equivalence test** (`tests/test_feature_cache.py`, CPU fp32): the loss from cached features matches raw images under identical noise and time (rtol 2e-3), and the cached dataset reproduces LeRobot's state, action chunk, `action_is_pad` and task.
- **Cache size:** 24,211 frames → **2.98 GB** (2.8 GiB). It's built once per dataset and backbone dtype. On MPS the build took **2429 s (40 min)**, about 10 frames/s for SigLIP at 512 px. It's not uploaded to the Hub (it's dtype-specific); the Colab notebook keeps it on Drive.
- **Time per step on MPS** (micro-batch 16 × grad_accum 2 = effective 32, sim_seen):

  | | s/step | 500 steps |
  |---|---|---|
  | raw images (100 steps measured, steady) | 4.8 | 40 min |
  | **cached features** (100 steps; first 25 slower while the memmap pages in) | **1.33** | **11 min** |

  **3.6× faster.** The cache build (40 min) pays for itself after about 700 steps.
- **Eval cost** (raw images, 18 envs batched, `n_action_steps=10`): 18 episodes that all ran 200 steps took 115 s, so ≤ 6.4 s/episode. 20 episodes/task ≈ 19 min and 50/task ≈ 48 min at most (successes end early).

## Backbone dtype on GPUs without bf16 (`vla/common.py: resolve_dtype`)
- `--dtype auto` keeps SmolVLA's default (bf16 backbone) on MPS and Ampere+ CUDA, and casts the whole policy to **fp32** on CUDA without bf16 (T4) or on CPU. A full fp16 cast is not offered: the backbone was trained in bf16 (fp16 can overflow), and fp32 inputs such as the state would then hit fp16 layers. Not yet run on a T4; that's the user's Colab timing check.

## V0 overnight plan (Mac)
- `scripts/run_v0_mac.sh` (under `caffeinate -i`): 5000 steps from the cache (≈ 1.9 h), 20-episode/task evals every 1000 steps (≤ 1.6 h), then a final 50-episode/task eval of the best-by-seen checkpoint (≤ 48 min): ≈ 4.5 h. Training resumes and finished evals are skipped on re-run. Started 2026-10-08.
- Projection for V1/V2 on Colab: pending the user's T4 timing. On the Mac, each would take about as long as V0 (the phone data adds ≈ 15% frames, plus building its cache).

## Real setup check (2026-10-08, two still photos, `data/raw_phone/IMG_8679/8680.HEIC`)
- **Markers:** all 4 detected at video-like resolution (756×1008). They were taped with IDs rotated one corner (0→1 along the 35 cm side). This is fixed in software with `session.json: "corner_ids": [1, 2, 3, 0]` (near-left, near-right, far-right, far-left as seen from the user's seat), rather than re-taping. Handedness is correct (not mirrored); with the fix the warped top-down view lines up with the 50 × 35 grid.
- **Photo 1** (brown stone, silver block, teal figurine; yellow/green/red paper): blurry, with a big shadow. Color detection: teal, green and red within 0.2 cm; **brown stone, silver block and yellow paper failed** (they look like shadowed wood, bright wood/tape, and wood hue ≈ 21 respectively).
- **Photo 2, the revised setup** (black cylinder, dark-blue lid, teal figurine; white/green/red paper): sharp, no big shadow. Blue lid, white paper, green paper and red paper are found within **0.1 cm**. The black cylinder matched the black ArUco squares (44.7 cm off), so detection now **ignores a 10 cm square around each marker** (`detect_objects(exclude_xy=...)`); with a sample from the cylinder's dark rim it lands within **0.3 cm**. **Teal figurine vs dark-green paper:** similar hue (≈ 81 vs 76); separable only by brightness and, depending on the sample, detection picks part of the paper (12 cm off). A joint "shrink margins until items don't overlap" fit didn't help (the confusing pixels are unsampled parts of the paper), so it was reverted.
- Conclusion: with this setup, 4–5 of 6 items detect automatically; **use `scripts/click_layout.py` for all clips** (6 clicks/clip, a few minutes for 48) so layouts never depend on the teal/green separation.
- **Photo 3, final setup** (`IMG_8681.HEIC`: all-black cap, dark-blue lid, purple keycap; white/green/red paper): **all 6 items detected automatically** in 3 sampling variants (click-tool-like patches of radius 5 and 8 px at 1.6 px/mm, and an offset click on the cap): black cap 0.7–0.8 cm, everything else 0.1–0.2 cm. The photo is slightly motion-blurred and has a shadow on the left, and detection still holds. Color detection is the default for this setup; `click_layout.py` stays the fallback for clips where `calib_colors.py --check_all` reports a miss.

## Milestone 2: first real clips (session `1`, 2026-10-08)
- Clips: `calib_table`, `calib_box`, and 3 demos (`blue_to_orange`, `green_to_yellow`, `red_to_purple`), 1080×1920 portrait at 30 fps, 4–12 s. `calib_box` was recorded with the phone rotated 180° relative to the others (no effect: per-frame homography). The pipeline now accepts the short file-name style `<cube>_to_<zone>.MOV` (`human/objects.py: parse_clip`).
- `session.json`: corner_ids [1, 2, 3, 0] and the item mapping as recommended; **camera height measured by the user: 17 in = 43.2 cm**; **box 17 cm** (measured). Still placeholders: rect nominal 50 × 35, object heights (cap 3.0, keycap 1.2, lid 2.0 cm).
- **Tracking quality:** markers detected in 69–100% of frames per ID (gaps interpolated); **hand tracked in 100% of frames** in all clips. The rigid mount is confirmed by identical solvePnP poses across clips. Track errors vary by up to ~5 mm between identical runs (MediaPipe is not bit-deterministic).
- **Camera height / height model.** A first solvePnP estimate with *assumed main-lens* intrinsics gave 78 cm; with it, the box was estimated at 28.8 cm (**+11.8 cm** error). The size ratio (8.0 → 12.7 cm) instead implies ≈ 46 cm, which matches the user's measurement (43.2 cm) and a ≈ 100° FOV, i.e. **the 0.5× ultra-wide lens**. A two-point fit from the box gives 45.7 cm (2.5 cm from the measurement). **With the measured 43.2 cm: box estimated at 16.0 cm, error −1.0 cm** (`outputs/m2/1/calibration.json`). Default: use the measured height; `--fit_camera_height 1` fits it from the box when it wasn't measured.
- **Object/zone detection on video** was much worse than on stills (white patch vs the pale hand, tape and glare; black cap vs shadows; exposure drift between clips, since exposure was not locked). Added hand masking (the first-frame hand hull + 3 cm is ignored), which fixed the cap in 2 clips, but it's still unreliable. For these 3 clips the layouts were read manually off rendered top-down first frames (`<clip>.layout.json`, a stand-in for `click_layout.py`).
- **Replay, before/after the two allowed fixes** (3 clips):

  | config | box-height error | replay success |
  |---|---|---|
  | (a) H = 78 cm (assumed intrinsics), raw z | +11.8 cm | 0/3 |
  | (b) H fitted from box (45.7 cm), raw z | −0.9 cm | 0/3 |
  | (c) H fitted + **contact-anchored z** | −0.9 cm | 1/3 |
  | (d) **H measured (43.2 cm) + contact-anchored z** (default) | −1.0 cm | **1/3** (red-purple) |

  Fix 1 = correct camera height. Fix 2 = contact-anchored z (`human/retarget.py: anchor_contact_z`): at the moments the grip closes and opens, the pinch point is at the object's half-height, so the posture-dependent bias of the size-based z (a pinching hand vs the flat calibration hand) is measured there and removed (linear in between, constant outside). Raw z at the grasp had been ≈ 9–10 cm above the object.
- **Remaining failure modes** (measured in replay at the gripper-close step):
  - blue-orange: EE 2.3 cm off the cube in xy and at z 4.0 cm when the gripper closes (the commanded z was correct; **the arm lags** the human, who closes instantly on contact) → closes on air.
  - green-yellow: 3.5 cm xy offset at close → closes on air. Also, **release is detected 16 cm late** (at HOME): the keycap is only 1.2 cm wide, so the fingers barely open on release and the aperture/hand-size ratio stays below the fixed open threshold (0.9) until the hand flattens. The human did complete the task (last frame: keycap on the white patch).
  - red-purple ✓: 1.5 cm xy, z 2.1 cm at close; final cube 5.0 cm from the zone center (just inside).
- Not fixed (beyond the plan's 1–2 fixes; candidates): a short dwell inserted at the close/open events so the arm settles; release threshold relative to the aperture at grasp (e.g. open when aperture rises by > 0.2 over its grasp value) instead of a fixed 0.9.

### Fixes 3–4 (requested by the user after the first diagnosis) and session 2
- **User protocol changes for the full set:** pinch the keycap across its long side (≈ 2.5 cm); exaggerate opening and closing (spread wide before the grasp and after the release); pause ≈ 0.5 s at each grasp and release; lock exposure; keep the 0.5× lens; fill in object heights and rectangle measurements; use `click_layout.py` for every clip.
- **Session 2** (3 re-recorded demos, same mount and items; no calibration clips, so `session.json: "calib_from": "../1"`). Clips are 17–19 s. Layouts again read off rendered first frames by Claude (stand-in for `click_layout.py`).
- **Aperture traces** (aperture / hand size; plot in `outputs/m2/`): session 2 shows the exaggerated pattern clearly: rest ≈ 0.9 → spread 1.8–2.2 → grasp 0.6–0.85 → spread → rest ≈ 0.9. The fixed thresholds fail: the lid's grasp opening (≈ 0.85) never drops below 0.65, and the resting hand (0.9) sits at the open threshold.
- **Fix 3, relative gripper thresholds** (`human/extract.py: grasp_interval`, `--grip_mode relative`, default): spread = 97th percentile of the clip; close = falling crossing of 0.6 × spread (the hand must be open first); open = rise above m + 0.5 × (spread − m), with m = the grasp opening (median over 0.5 s after closing); one grasp per clip, so relaxing at HOME isn't a second grasp.
- **Fix 4, dwell** (`human/retarget.py: add_dwell`, `--dwell_s 0.5`, default): at each gripper switch, hold the EE 0.5 s with the old state and 0.5 s with the new one, so the arm settles before closing and the gripper finishes before moving on. Replay steps carry their source time, so the side-by-side phone panel freezes during dwells.
- **Before/after** (replay success; d = after fixes 1–2):

  | config | session 1 | session 2 |
  |---|---|---|
  | (d) fixed thresholds, no dwell | 1/3 | 1/3 (blue-orange: **no grasp detected**) |
  | (e) + relative thresholds | 1/3 | 1/3 (all 3 extracted) |
  | (f) + 0.5 s dwell (**default**) | 1/3 | 1/3 |

- **Detection check on session 2 (new protocol), config f:** exactly one close and one open per clip; release over the right zone in all 3 (EE 0.9–1.9 cm from the zone center); EE height at close 2.1–2.3 cm (cube center 2.0). **The only remaining failure mode is xy offset at the grasp:** successes closed 0.6–1.0 cm from the cube, failures 2.3–3.8 cm (beside the cube). Suspected sources: manual layout reads (±1 cm), placeholder object heights in the parallax correction, and MediaPipe fingertip landmarks (nail tips, seen from above). Not tuned further on these clips (per the user); the candidate is contact-anchored xy (snap the grasp point to the detected object, like z), to be judged on the full 48.
- **Matching-view sim camera** (`human/camera_match.py`): the phone pose from solvePnP on the 4 markers (median over 30 frames of `calib_table`), with the focal length solved so that the camera height equals the measured 43.2 cm (pinhole, principal point at the image center). Result: tilt 8.4°, FOV 68 × 100° (ultra-wide, as expected), marker reprojection error 23.8 px at 960 × 1707 (lens distortion and the nominal rectangle; ≈ 7 px at video display size). The `phone_match` camera in `assets/scene.xml` is set at runtime. A blend check shows the sim zones and cubes over the real patches and objects. Side-by-side videos are now 3 panels: phone | sim from the phone's viewpoint | sim front.

## Env v2: gripper rotated 90°, gripper starts (and ends) closed (2026-10-08, user request)
- **Why:** in the phone clips the human pinches across the near and far sides of an object, while the robot closed left–right; and the human's relaxed hand (fingers together) is "closed" while the robot started open.
- **Changes** (`EnvConfig.gripper_yaw_deg = 90`, `start_gripper_closed = True`; v1 = 0° / open, kept as `LEGACY_V1`): `sim/ik.py: down_rot(yaw)` (yaw 0 reproduces the v1 orientation exactly); reset with the fingers closed; **"released" redefined** as fingertip center > 4.5 cm from the cube center (an open-gripper test no longer works when the gripper closes again at rest). The expert waits closed 0.2–0.6 s, opens on the way to the cube, and after the place returns to HOME and closes, like the human. The phone gripper signal mirrors the hand (`grasp_interval(rest_closed=True)`): closed at rest, open while spread, closed on the object, open after release, closed when relaxed again.
- **Checks:** reachability with yaw 90 is unchanged (worst 3.2 mm over the workspace grid incl. 2 cm outside; finger axis exactly along y). **Expert v2: 900/900 (100%)** (`outputs/m1_v2/expert_success.csv`), episodes ≈ 100 steps. All tests pass (grasp-interval test covers both semantics).
- **Versioning:** each episode stores `meta.env_cfg`; the LeRobot export rebuilds that env and writes `env_cfg` into `episodes.json`; `vla/eval_smolvla.py` evaluates in the training data's env (absent = v1). New data: `data/sim_v2` (200/task, all tasks; 2 min 19 s), `data/lerobot/sim_seen_v2`. The Phase 1 mechanism-study results (small diffusion policy) and the first SmolVLA V0 run are **v1**; the V0 run is kept as `checkpoints/vla/V0_pilot_envv1` (5000 steps, 1.99 h, loss 0.0147) and evaluated once (step 5000, 20 eps/task) as a "does SmolVLA learn this task" check; V0 is being retrained on v2.
- **Object sizes measured by the user:** black cap Ø 1.5 in × 1 3/8 in tall (3.81 / 3.49 cm), purple keycap 1.5 × 0.5 in footprint × 3/8 in tall (3.81 × 1.27 / 0.95 cm), blue lid Ø 2 in × 5/8 in (5.08 / 1.59 cm). Only the rectangle measurements are still nominal.
- **Phone replays with v2 + measured heights:** session 1 (old protocol) 1/3; **session 2 (new protocol) 2/3** (blue-orange ✓, red-purple ✓; grasp 0.8–1.3 cm from the cube). Session 2 shows the intended 4 gripper switches per clip (open, close on object, open at the zone, close at rest). The failure (s2 green-yellow, keycap) closes 3.0 cm off along y (now the finger axis), so a finger lands on top of the cube (EE stuck at z 3.8). Suspected: the manually read keycap position. To be re-checked with the user's `click_layout.py` clicks; not tuned.

## SmolVLA V0 pilot (env v1): language is not grounded
- `checkpoints/vla/V0_pilot_envv1`: 300 sim demos (50 × 6 red/green tasks), cached features, 16×2 effective batch 32, lr 1e-4 cosine to 2.5e-6 over 5000 steps (≈ 6.6 epochs), 1.99 h on MPS, final loss 0.0147.
- **Eval (step 5000, 20 eps/task):** seen **0.8% [0.1, 4.6]** (1/120), held-out 0% (0/60). 1163 s for 180 episodes.
- **Diagnosis** (evaluator now logs a failure analysis: target lifted, nearest zone, still held, other cubes moved, grasp offset):
  - 0/18 seen episodes lifted the target cube; the gripper closes a **median 17.6 cm** from the target cube (z is right: 1.9 cm above its center); 25% move a wrong cube. The gripper channel is crisp (predictions ≈ exactly 0/1).
  - On **training layouts** it's no better: 1/18, grasp offset median 8.7 cm. Offline on training frames, the predicted 10-step chunks are off by a median 3–3.5 cm (max over the chunk), flat from step 1000 to 5000 while the loss fell 2.6× → not a matter of more steps at this LR.
  - **Instruction-swap probe** (same image and state, "red"/"green"/"blue" in the instruction): the predicted grasp point moves by only 1–6 cm between colors and goes to the same cube regardless of the named color (e.g. it heads for the red cube for all three instructions). **The policy ignores the language**; it learned a generic "go to a salient cube" motion. The same failure the Phase 1 concat encoder had (no multiplicative instruction × scene interaction), now in the VLA, with only the action expert trainable (VLM frozen) and 300 demos.
- Not a plumbing bug: the image path, state, normalization, gripper channel, task strings and env version were checked (eval uses the training data's env version; the cache equals raw images, per the test).
- **Grounding probe** (`vla/probe_grounding.py`, about 1 min per checkpoint): for 8 eval layouts × 3 instruction colors, is the predicted grasp point nearest the named cube? Pilot checkpoints 500–5000: **accuracy 0.29–0.46 (chance 0.33)**, median distance to the named cube 12–17 cm, so grounding never appeared during training. This is the cheap check to run before any full eval.
- **Disk incident:** the v2 V0 run crashed at step 2000 with "No space left on device" while the probe ran on MPS alongside it (memory pressure → swap on a nearly full disk; the project itself uses ≈ 7 GB). Fixes: optimizer state (2× the weights, ≈ 0.4 GB) is now kept only for the newest checkpoint (`save_checkpoint` prunes older ones); the v1 vision cache was deleted; **never run two MPS jobs at once**. The v2 run can resume from step 1500.

## Bug check before choosing a fix (2026-10-08, user request): no bug found
1. **Language tokens actually fed to the model** (decoded with the SmolVLM2 tokenizer after LeRobot's preprocessor): in a cached-path training batch, each sample has its own instruction, 9 tokens, e.g. `'put the green cube in the orange zone\n'` (complete, newline appended, padded to 48 with the mask excluding pads). Probe batches: `put the {red|green|blue} cube in the yellow zone\n` → identical ids except the color token (2382 / 2654 / 4461). Not empty, not identical, not truncated.
2. **Vision cache:** `[30644, 64, 960]` fp16, built only from `embed_image(images)` (SigLIP + connector); no text or prefix states. Cached samples carry their own instruction: 0/200 mismatches against the raw LeRobot dataset at the same frames; all 6 seen instructions present.
3. **Probe extended with distance to the nearest cube of any color** (pilot, step 5000): grounding accuracy 0.33 (chance), named cube 12.4 cm, **nearest cube of any color 5.9 cm (single samples: 6.8 cm; within 2 cm in only 3%)**. Probe validity: on the expert's own actions the same metric (lowest point of the first 50-step chunk) is 0.3 cm from the named cube, 24/24 within 2 cm, for both v1 and v2.
- **Conclusion: two problems, both real**: no language grounding *and* poor localization (grasp points don't land on any cube). → Option 2 started overnight (below); **for tomorrow's plan: a closer camera and/or higher image resolution** (the 256 px image is upsampled to 512 and becomes 64 tokens, each covering ≈ 9 cm of the workspace), alongside the Colab Pro / unfreezing decision.

## Option 2 overnight (started 2026-10-08)
- `scripts/run_option2_mac.sh`: export `sim_seen_v2_200` (200 sim demos × 6 red/green tasks, env v2), 15k steps with lr held at 1e-4 after a 300-step warmup (`--lr_schedule constant`), effective batch 32, checkpoints every 2500 steps; then the grounding probe on every checkpoint; then a 20-episode/task eval of the last checkpoint; then the remaining ambient-synthetic runs. One MPS job at a time. The ambient-synthetic sweep was paused for this (run A was at step 7000 of 20000; restarted from scratch at the end).

## Revised overnight run: wrist camera + 512 px (2026-10-08, user request)
- **Why:** option 2 as first specified (more data, longer training) addressed language grounding but not localization: the pilot's grasp points landed a median 6.8 cm from the nearest cube of any color. SmolVLA turns every image into **64 tokens** (512 px → SigLIP 32×32 patches → pixel shuffle → 8×8), so with the old 256 px scene image (upsampled to 512) each token covers **≈ 9 cm** of the workspace. **A wrist camera is the standard remedy**: it gives close-up, gripper-relative views where precise alignment matters, and the scene camera keeps the global layout. Rendering at 512 px directly also removes the 2× upsampling blur.
- **Cameras (images v2):** `scene` = same position as `phone` (0.25, 0.55, 0.95), aim and FOV fit so the workspace box (+2 cm margin, up to 16 cm carry height) just fills the square image: aim (0.25, 0.265, 0.04), **fovy 38°**, tilt 17.4° (the width of 54 cm is the limiting dimension). `wrist` = on the Panda hand, 5 cm to the side of the fingers (hand x), looking at a point just past the fingertips, image x = finger axis, fovy 75° (`assets/franka_emika_panda/panda.xml`). Both at **512 px**; render cost 16 ms/step for both. Visual check at start/reach/grasp/carry/place: all 6 items and the gripper visible in `scene`; `wrist` shows both fingertips and the cube between them, and the target zone during the carry. `phone` is kept unchanged for the pilot; eval/probe now take cameras and size from the run's features (`vla/common.py: image_spec`).
- **Run** (`scripts/run_v0_2cam_mac.sh`): env v2, **100 sim demos per red/green task (600)**, 15k steps, constant lr 1e-4 after a 300-step warmup, effective batch 32, checkpoints every 2500, probe (grounding accuracy + named-cube + nearest-any-cube distance) on every checkpoint after training, then a 20-episode/task eval. **Disk check:** two-camera cache ≈ 61k frames × 2 × 64 × 960 × 2 B ≈ 15 GB; 46 GiB free after deleting the superseded single-camera v2 cache. Expected durations: export ≈ 50 min, cache ≈ 3.4 h (≈ 10 images/s on MPS), training ≈ 7 h, probe + eval ≈ 40 min.
- **Colab/A100 alternative prepared** (`colab/train_smolvla.ipynb`, run `V0_2cam_unfrozen`): the same data and config plus `--unfreeze_vlm 1` = `train_expert_only=False` with SigLIP **and the connector kept frozen** (so the post-connector feature cache stays valid; grounding happens in the language layers where image and instruction tokens interact): **305M trainable** (205M language layers at lr 1e-5 + 100M expert at 1e-4) vs 100M frozen-VLM. The vision cache is built on the runtime's local disk (not Drive). Dataset push to the Hub (`Rishi1545/dcad_sim_seen_v2c_100`, private) is automated after the export, but the Mac's HF token is still **read-only** (403): needs `hf auth login` with a write token.

## Workflow change (2026-10-09): all training on Colab
- The overnight Mac run (`scripts/run_v0_2cam_mac.sh`, frozen VLM) finished the export (600 episodes, 61,577 frames) and the two-camera cache (2 × 7.57 GB, **12,006 s ≈ 3.3 h** on MPS), then was **stopped by the system for critically low memory** before logging its first training step (18 GB unified memory: model + 8 dataloader workers paging a 15 GB memory-mapped cache + swap; 8.8 GiB of disk left). Not restarted.
- **From now on: code locally, train on Colab** (A100; `colab/train_smolvla.ipynb`). No large training runs on the Mac. The Colab notebook builds the cache on the A100 and keeps it on Drive (trusted only with a `COMPLETE` marker, so an interrupted copy is rebuilt).

## Colab A100 run `V0_2cam_unfrozen` (2026-10-09)
- Config: env v2, scene + wrist cameras at 512 px, `sim_seen_v2c_100` (600 demos, Hub `Rishi1545/dcad_sim_seen_v2c_100`), `--unfreeze_vlm 1 --vlm_lr 1e-5` (305M trainable: 205M language layers + 100M expert; SigLIP + connector frozen, so the cache stays valid), constant lr 1e-4 after a 300-step warmup, batch 32, 15k steps, checkpoints every 2500 → the user's Drive `MyDrive/dcad/checkpoints/vla/V0_2cam_unfrozen/`.
- **Speed: 0.187 s/step on an A100-40GB with the vision cache, i.e. ≈ 3.1 min per 1000 steps; 15k steps in 2850 s (47.5 min).** Far faster than the estimate (0.3–0.6 s/step).
- **Loss** (copy: `outputs/vla/V0_2cam_unfrozen/loss_colab.csv`, gitignored): 0.187 (25) → 0.042 (1k) → 0.031 (4k) → 0.026 (5k) → 0.022 (7k) → 0.016 (10k) → 0.015 (12k) → **0.0122 (15k)**, still falling at constant lr (the frozen-VLM pilot plateaued at ≈ 0.0145 under a decaying lr, but on different data and cameras; not directly comparable).
- **Probe: results not yet seen.** The background watcher (cell 5) wrote nothing visible (its stderr went to `/content/probe_stdout.txt`, lost on runtime reset). A foreground `--step all --log …` re-probe printed nothing: a bug where the probe skipped every step already listed in the log file even outside `--watch` mode (fixed: skipping only in `--watch`, and it now prints the checkpoints it sees). So `MyDrive/dcad/V0_2cam_unfrozen_probe.txt` may already contain the watcher's results.
- Colab notebook robustness fixes made on the way: a self-contained clone cell (IPython `$`-expansion leaves the whole line unexpanded if any name is undefined → blank token/repo), a torch sanity check in a fresh process, the Drive cache trusted only with a `COMPLETE` marker, and probe/eval failing loudly if the training dataset's `episodes.json` is missing (no silent fallback to the v1 env).

## Probe of `V0_2cam_unfrozen` (2026-10-09, colab CLI)
- **The final checkpoint (`step_015000`) is missing on Drive**, although Drive's `loss.csv` reaches 15000: Drive uploads big files in the background, and the runtime was deleted right after the final save (also lost: the cache's `COMPLETE` marker and the deletion of `step_012500/train_state.pt`). Latest checkpoint = **12,500**. Fix: `dcad.py down` / `wait --stop` call `drive.flush_and_unmount()` before stopping the VM; the notebook's last cell does the same.
- **Probe segfault on Colab:** MuJoCo's EGL context creation crashed (`Segmentation fault` in `mujoco/egl/__init__.py`) whenever the policy had been loaded first. Cause: Colab ships TensorFlow, `transformers` imports it, and its libraries clash with EGL. `USE_TF=0` fixes it (now set in `vla/common.py`). This is very likely why the background probe watcher "died silently" during training, and it would have broken the eval too.
- **Results** (n = 24 per checkpoint: 8 layouts × 3 instructions; chance 0.33):

  | step | grounding acc. | to named cube | to nearest cube | single samples < 2 cm |
  |---|---|---|---|---|
  | 2,500 | 0.33 | 12.7 cm | 6.8 cm | 4% |
  | 5,000 | 0.50 | 10.3 cm | 6.2 cm | 11% |
  | 7,500 | 0.38 | 10.9 cm | 4.6 cm | 17% |
  | 10,000 | 0.54 | 8.3 cm | 4.3 cm | 15% |
  | 12,500 | **0.58** | **7.4 cm** | **3.0 cm** | 19% |

  Unfreezing the language layers works: grounding rises above chance (frozen-VLM pilot: chance) and localization improves (pilot: 6.8 cm to the nearest cube), both still improving at 12.5k → continue training (`V0_2cam_unfrozen_30k`, initialized from step 12,500). Still far from the < 2 cm target. n = 24 is small (0.58 = 14/24).

## Colab pipeline (`colab/dcad.py`, 2026-10-09)
Driven from the Mac with the colab CLI; the VM runs `colab/job.py` detached (nohup), all state on Drive, so the Mac can sleep.
- **Runs are named** in `colab/runs.json` (data, train args on top of the defaults, `eval_k`, optional `init_from` = continue another run's latest checkpoint). One entry per experiment; don't edit an entry that has run.
- **Stages** (each idempotent, re-run = resume): `data` (Drive `dcad/datasets/<name>.tar` → VM local disk) → `cache` (Drive `dcad/vision_cache/<name>_bf16` with `COMPLETE`, else built on the GPU and saved) → `train` (`--resume` into `dcad/checkpoints/vla/<run>`) → `probe` (all checkpoints → `probe.txt`) → `eval` (latest, `--k eval_k` → `eval_step*_k*/summary.json`). Per run: `job_status.json` (stage, timings, error, commit) and `job.log`; per job: `dcad/jobs/<id>.json`. Several runs in one job run back to back (e.g. a t_min sweep overnight).
- **Commands:** `up` (session + code at the local HEAD, which must be pushed; pip install once per VM), `push-data <dir>` (tar without the cache, 64 MB parts; 434 MB in 45 s), `run <runs...> [--stages]`, `status`, `wait --stop` (poll, fetch, flush Drive, stop), `fetch <runs>` (→ `outputs/colab/<run>/`), `down`.
- **Manual per new VM:** `colab drivemount -s dcad` (interactive). First `pip install -r requirements.txt` on a fresh VM takes ≈ 14 min.
- **CLI pitfalls handled:** `colab exec` exits 0 when the code raises (traceback on stderr) → the driver treats a traceback as failure; upload fails for large files and for missing parent dirs; detached processes linger as zombies (liveness reads `/proc/<pid>/stat`); `colab stop` only knows sessions the CLI created (others: `client.unassign(endpoint)`); idle runtimes from browser tabs keep billing until deleted.
- `colab/train_smolvla.ipynb` is now a thin browser wrapper around the same `job.py`.
- **Eval speed:** the smoke eval took 916 s for 18 episodes (one batch): MuJoCo renders 512 px images at ≈ 0.1 s each on Colab (EGL without the NVIDIA vendor library, i.e. software), 3.7 s per step for 18 envs × 2 cameras, vs 0.06 s physics. `eval_smolvla.py` now renders only when the policy's action queue is empty (once per `n_action_steps` = 10): **identical actions** (max |diff| 0 over 40 steps × 4 envs, same seeds) and 5.4× faster (38 s → 7 s).
- **Probe noise:** the 50-step smoke model scored 0.46 grounding with 8 layouts (n = 24), so n = 24 can't separate 0.58 from noise. The pipeline's probe now uses 32 layouts (n = 96; `defaults.probe` in `runs.json`).
- Smoke run (`smoke`, 50 steps): data 0 s (already extracted), **cache 654 s** (build 578 s on the A100 vs 3.3 h on MPS, + copy to Drive), train 95 s, probe 58 s, eval 944 s (before the render fix).
- `vla/build_cache.py`: one pass over the video for all cameras (was one decode pass per camera) with progress lines; `tests/test_feature_cache.py` passes.

## Phone session 3: the 48 demos (2026-10-09)
- Recorded into per-task folders (`data/raw_phone/3/<task>/IMG_*.MOV`, portrait 1080×1920); flat `<task>_NN.MOV` symlinks feed the pipeline. Same setup as sessions 1–2 (user-confirmed) → `session.json` copied from session 2 (`calib_from: ../1`). Camera from markers: 43.2 cm, tilt 8.5°.
- **Two clips were filed in each other's folders**, found by checking every clip's end frame against the mapping (yellow = white patch, purple = green patch, orange = red patch): IMG_8747 (folder blue_to_orange) is blue→purple, IMG_8748 (folder blue_to_purple) is blue→orange. Relinked; the other 46 end on the correct patch. **Nothing needs re-recording.**
- Markers: marker 3 washed out by glare in 2 clips (never detected) → CLAHE fallback in `detect_markers` (found in 13–21% of frames; gaps interpolated).
- Color detection (session 2's HSV ranges): 17/48 clips fully detected; the misses (mostly the red and green patches, sometimes the lid) need `scripts/click_layout.py`. No hand-tracking failures.
- First pass on the 17 detected clips (before the relabel): replay success 4/17 (blue 1/10). Failures are grasp misses: EE–object offset at gripper close 1–7.6 cm, **with a consistent +x bias of ≈ 1.5–3.5 cm** (worth checking: hand keypoint vs. pinch point, or parallax with the portrait orientation). "human_completed = 0" in 4 correct clips is a detection artifact (the lid/keycap covers the white patch at the end).

### Session 3 processing after clicking (2026-10-09)
- All 48 clips clicked. Hand tracking cached per clip and run on 4 processes: full pass 75 → 12 min, reruns 5–7 min.
- **Grasp bias, clicked layouts** (`scripts/pinch_bias.py`): pinch point − object at the grasp = +0.17 / +0.52 cm (std 0.8 / 1.0, s.e. 0.12 / 0.14, n = 47). The earlier "+2 cm" came from color-detected object positions, not the hand. Corrected with `pinch_offset_cm` [−0.17, −0.52] in session.json (residual 0.0).
- **Why replays failed** (`scripts/replay_diagnose.py`): the arm reached its targets (lag < 0.1 cm) and the targets were on the object, but **the cube was pushed before the grasp** (median 1.1 cm in −x, up to 5.3 cm): the human opens the hand ≈ 3.4 cm from the object at ≈ 4 cm height and comes in low from the side; the Panda following that path hits the cube with its (still opening) fingers. 19/47 never lifted.
- **Fix: approach-from-above retargeting rule** (`RetargetConfig.approach_clear` = 5 cm, radius 6 cm, `human/retarget.py:approach_from_above`): within 6 cm of the grasp point the gripper is open, moves over the point 5 cm above it and descends vertically; after the release it rises 5 cm before moving on. Grasp/release points and step count unchanged. **Replay success 44% → 75%** (blue 9/24 → 21/24); remaining failures: 6 no lift, 3 just outside the zone (5.2–5.6 cm), 2–3 dropped.
- `green_to_purple_04`: no grasp detected because the release opening (thin keycap) stayed below the open threshold set from the pickup spread → retry with half the margin. **Final: 48/48 extracted, 37/48 replays succeed (77%)**; `phone_v2` (48 episodes, 7438 frames) on Drive.
- Side-by-side videos for every clip: `outputs/m2/3/<clip>_side_by_side.mp4` (phone with hand overlay | sim from the phone viewpoint | sim front; caption with task and replay result). The phone-viewpoint camera is estimated from the session's own clips (same pose as session 1: 43.2 cm, 8.6° tilt, as the user set it up).

### First results (2026-10-09, in progress)
- VLA `V0_2cam_unfrozen_30k`, eval step 30k (k = 20): **seen 39.2% [30.9, 48.1], blue 0/60**.
- VLA `C_blue4` (15k): probe blue 0.12; **eval seen 24.2% [17.4, 32.6], blue 1.7% [0.3, 8.9]** (1/60). Not yet comparable to V0 (only evaluated at 30k): V0 at 15k is queued (`eval_steps`).
- 2026-10-09 ~14:00: a network blip made the local watcher think the job had ended and try to create a new session (stopped at a login prompt; no VM created; the VM job was unaffected). Fixed: `session_exists` exits instead of answering False when the list can't be read; `wait` retries status. Chain now: Ceil_blue24 → CN_blue4 → V0 eval at 15k → stop (`outputs/colab_chain.log`).
- Small policy (L2 = phone-like noise): C seen 87.3 / blue 40.0; Ceil 95.3 / 94.7; L2_CN 74.3 / 40.0; L2_CNa25 76.3 / 38.0 (the remaining runs are going).

## Limitations (2026-10-09)
- **The phone data comes from a controlled, marker-calibrated setup**, so it understates the gap to in-the-wild human video. Fixed top-down phone at a measured height (43 cm, 0.5× lens), ArUco markers giving metric table coordinates in every frame, known object sizes and heights, a flat uncluttered table, a single right hand, and a recording protocol (exaggerated open/close, pauses) designed for the pipeline. In-the-wild video has moving/unknown cameras, no metric scale, occlusion, clutter and different grasps. Measured noise here: ≈ 2–4 cm grasp offsets and gripper timing errors; session 2 replays 2/3.
- Phone demos are retargeted and replayed in sim and re-rendered for the VLA, so the experiment tests demonstration (action) noise, not the visual domain gap.
- One task family, one held-out object (blue), single seeds for the VLA runs (the small-policy sweep is the cheaper place to check variance).

## Current status / resume here (2026-10-09 ~10:00)
- Colab CLI logged in (`~/.local/bin/colab`). OAuth gotcha: if the consent screen grants fewer scopes than requested, `oauthlib` raises "Scope has changed" and nothing is saved; tick every box (or `OAUTHLIB_RELAX_TOKEN_SCOPE=1`). The CLI hard-codes a DEBUG file log with token material at `~/.config/colab-cli/colab.log`: deleted and replaced by a symlink to `/dev/null` (2026-10-09). Credentials live in `~/.config/colab-cli/token.json`.
- Repo is **public** now (the VM clones without a token). The sim dataset is on Drive as `dcad/datasets/sim_seen_v2c_100.tar`.
- **Next:** `V0_2cam_unfrozen_30k` results (probe n = 96, eval k = 20). Phone data: record the 48 clips (RECORDING.md) → `calib_colors.py` / `click_layout.py` (interactive) → `scripts/phone_to_colab.sh data/raw_phone/<session>` (process_phone → export `phone_v2` → per-task replay summary → push to Drive; checked on session 2: features and env_cfg identical to `sim_seen_v2c_100`) → `python colab/dcad.py run V1_phone_naive V2_phone_ambient_t03` (add more t_min entries to `runs.json` for the sweep).
- Paused: the small-policy ambient validation on synthetic noise (`scripts/run_ambient_synthetic.sh`); no large runs locally.
- **Local disk:** deleted the local `sim_seen_v2c_100/vision_cache` (15 GB; Colab keeps its own on Drive). 49 GiB free.
- `V0_2cam_unfrozen_30k`: probe plateaus after 15k (red/green 0.8–0.94, blue 0.03–0.12, nearest cube ≈ 3.6 cm); eval of step 30k pending in `outputs/colab/V0_2cam_unfrozen_30k/`. Then `C_blue4` and `Ceil_blue24` run on Colab (chained, VM stops after).
- Small-policy sweep running on the Mac (`scripts/run_ambient_sweep.sh`, ~9 h, `outputs/ambient_sweep/run.log`); summary: `python scripts/summarize_ambient_sweep.py`, then set `defaults.ambient_t_min` in `colab/runs.json`.
- Phone: user to click the 31 undetected clips (`python scripts/click_layout.py data/raw_phone/3`), then `PY=~/miniforge3/envs/dcad/bin/python scripts/phone_to_colab.sh data/raw_phone/3` → `CN_blue4`, then (after t_min) `CNamb_blue4`, `Namb`.
