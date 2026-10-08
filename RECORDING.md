# Phone recording protocol

Read this once before you start, then use it as a checklist. A session takes about 45–60 minutes.

## 1. Things you need

- [ ] `assets/markers.pdf` printed **at 100% / actual size** (no "fit to page"). Use a ruler to check that each black square is 6.0 cm. Cut the markers out with a white margin around each one.
- [ ] 3 cubes or blocks, 3–5 cm, colored **red, green and blue**, each solid and saturated (painted wooden blocks, Duplo/Lego bricks or wrapped boxes all work).
- [ ] 3 paper squares, about 10 × 10 cm, colored **yellow, purple and orange**.
- [ ] A box (or book stack) of known height, 10–20 cm, with a flat top.
- [ ] A tape measure, tape, and a phone mount (shelf, tripod with an arm, or a stack of books at the table edge).

## 2. Table frame (important: the sim uses the same frame)

Stand at the **near edge** of the table, where you'll stand while recording. Tape the markers so their **centers** sit at the corners of a 50 × 35 cm rectangle:

```
        far edge (away from you)
   ID 3 ●──────── 50 cm ────────● ID 2
        │                       │
      35 cm      workspace     35 cm
        │                       │
   ID 0 ●──────── 50 cm ────────● ID 1
        near edge (you stand here)       [HOME] = just inside ID 1, near-right corner
```

- x runs from ID 0 toward ID 1 (to your right), y runs from ID 0 toward ID 3 (away from you), and the origin is the center of ID 0.
- Keep the markers flat and unwrinkled, all the same way up (the text label toward you).
- **Measure the actual center-to-center distances** 0→1, 1→2, 2→3, 3→0 and the two diagonals 0→2 and 1→3, and write them into `session.json` (below). A few mm off from 50 × 35 is fine, as long as you record it.
- **HOME** is a spot about 5 cm inside the rectangle from ID 1 (near-right). Mark it with a small piece of tape.

## 3. Camera

- [ ] The phone looks **roughly straight down** from **60–90 cm** above the table. Mount it rigidly; it must not move during a clip.
- [ ] Landscape orientation, **1080p at 30 fps**.
- [ ] **Lock focus and exposure** (on iPhone, long-press on the table until "AE/AF LOCK" shows; on Android, use Pro mode or the lock option).
- [ ] All 4 markers stay **fully visible in every frame**, with some margin. Check the edges of the picture before you start.
- [ ] Measure the camera height: lens to table surface, in cm.

## 4. Calibration clips (once per session, before the demos)

Hand pose: hold your right hand **flat, palm down, fingers together**, at the same angle you'll pinch with, i.e. back of the hand facing the camera.

1. `calib_table.mp4`: about 3 s with your hand resting flat on the table at HOME, not moving.
2. `calib_box.mp4`: put the box inside the rectangle, then about 3 s with your hand resting flat on top of the box, not moving. Measure and record the box height.

## 5. Demo clips

For each clip:

1. **Randomize the layout:** place the 3 cubes and 3 paper squares at random spots inside the rectangle. Keep at least 8 cm between any two objects and at least 3 cm from the markers. Keep the cubes square to the rectangle (not rotated).
2. Start recording. **Hand at HOME**, still, for about 1 s.
3. Using your **right hand** with a **thumb–index pinch from above**, pick up the named cube, carry it, place it **in the middle** of the named zone, and release.
4. **Return the hand to HOME**, hold for about 1 s, then stop recording.
5. Keep the hand **open (fingers spread)** whenever you're not holding a cube, and pinch decisively, so the gripper signal is clear.

Each clip should be about **5–15 s**. Move at a calm, natural speed. Keep your arm and sleeve out of the way of the cubes where you can; the hand itself must stay uncovered.

### How many clips, and file names

File name: `<cube>-<zone>_<NN>.mp4`, e.g. `red-yellow_03.mp4`.

| Task | Clips | | Task | Clips | | Task | Clips |
|---|---|---|---|---|---|---|---|
| **red-yellow** (held-out) | **8** | | red-purple | 4 | | red-orange | 4 |
| green-yellow | 4 | | **green-purple** (held-out) | **8** | | green-orange | 4 |
| blue-yellow | 4 | | blue-purple | 4 | | **blue-orange** (held-out) | **8** |

**48 clips in total.** If a clip goes wrong (the cube is dropped, a marker gets covered), just re-record it under the same name. You don't need to keep the bad one.

## 6. Tips

- Use good, even lighting with no hard shadows; daylight plus room lights works well. Avoid glare on the markers.
- Keep anything else **red, green, blue, yellow, purple or orange** out of the camera view, including clothing, mugs and phone cases.
- A long sleeve is fine, but keep the hand itself uncovered (no gloves, and preferably no bright rings or watches).
- Only one hand in view.

## 7. Where to put the files

```
data/raw_phone/<session_name>/        e.g. data/raw_phone/2026-10-10_kitchen/
├── session.json
├── calib_table.mp4
├── calib_box.mp4
├── red-yellow_01.mp4
└── ...
```

`session.json` (fill in your measurements, in cm):

```json
{
  "camera_height_cm": 75.0,
  "rect_cm": {"d01": 50.0, "d12": 35.0, "d23": 50.0, "d30": 35.0, "d02": 61.0, "d13": 61.0},
  "calib_box_height_cm": 15.0,
  "cube_size_cm": 4.0,
  "zone_size_cm": 10.0,
  "hand": "right",
  "notes": ""
}
```

Copy the session folder to `data/raw_phone/` on the Mac that runs the pipeline. It's gitignored, so videos never go into the repo.
