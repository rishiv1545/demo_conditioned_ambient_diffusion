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
