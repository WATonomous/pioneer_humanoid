# Camera calibration

Calibration for the wrist camera, and the code that makes real wrist camera frames match the sim.

| File | What it is |
|---|---|
| `calibrate_intrinsics.py` | ChArUco calibration tool: print a board, capture, solve, write the pinhole target |
| `wrist_cam.py` | Use the wrist camera in code: `rectify(frame)` turns a real frame into the sim's image |
| `wrist_cam_calib_1080/wrist_cam.yaml` | Measured lens of the real wrist camera (ROS camera_info, 1920x1080) |
| `wrist_cam_calib_1080/wrist_cam_pinhole_640x480.yaml` | The ideal camera that real frames are remapped to, and that the sim renders |

## Why this matters

Policies are trained on sim camera images and then run on real camera images. If the two
cameras see differently (field of view, lens distortion, where the camera points), the
policy sees a different world than it trained on, and grasps miss.

So the sim and the real camera meet in the middle, at an ideal **640x480 pinhole camera, f = 700 px
(49.1° x 37.8°)**:

```
sim:   Isaac / MuJoCo renders the pinhole directly ───────────────┐
                                                                  ├─> same image -> policy
real:  1920x1080 frame -> wrist_cam.rectify() -> 640x480 pinhole ─┘
```

`rectify()` undistorts the frame, crops it to 4:3, scales it down and moves the optical centre to the
middle of the image, all in one remap. In the sim, the lens is `WRIST_FOCAL_MM` in
`src/pioneer_humanoid/pioneer_humanoid/arm_params.py`, used by the Pioneer wrist cameras and by
`quest_cameras.py`.

## Using the wrist camera in code

Anything that feeds real wrist camera frames to a policy, or records them into a dataset, must
rectify them first:

```python
from wrist_cam import WristCam, open_camera

cam = WristCam()                # loads wrist_cam_calib_1080/
cap = open_camera(0, cam)       # opens at 1920x1080, the calibrated resolution
ok, raw = cap.read()
img = cam.rectify(raw)          # 640x480, matches the sim wrist camera
```

- Capture at 1920x1080. `rectify()` raises an error for any other size, because a frame that was
  already resized would remap without complaint, just to the wrong place.
- Don't resize or crop the frame yourself before `rectify()`: it does both.
- To see the result, run `python3 wrist_cam.py --camera 0`. It shows the raw and rectified views
  side by side. Straight edges should look straight in the rectified view.

## Calibrating a camera

Calibrate again if you change the camera or the lens, or if the focus ring moves.

### 0. Setup

```bash
python3 -m venv ~/venvs/calib
~/venvs/calib/bin/pip install opencv-python numpy
```

Use a separate venv. LeRobot installs `opencv-python-headless`, which can't open windows, and the
two packages clash in the same environment. Run the commands below from this folder
(`src/perception/camera_calibration/`).

### 1. Make the board

```bash
~/venvs/calib/bin/python calibrate_intrinsics.py board --out charuco_board.png   # --page a4 for A4
```

- Print it at 100% / "actual size". Any scale works as long as you measure the result, but
  "fit to page" makes the board harder to read.
- On macOS, Preview treats the PNG as 72 dpi. Set 300 dpi first:
  `sips -s dpiWidth 300 -s dpiHeight 300 charuco_board.png`.
- Glue it flat to something stiff, like a cardboard box face. Use a glue stick or spray glue,
  since wet glue wrinkles the paper. Tape only the white margin, never the pattern.
- Measure one black square with calipers. Better, measure across 5 squares and divide by 5. The
  current board is 23.84 mm.

### 2. Focus the camera

The wrist camera has a manual-focus screw lens. Point it at the board about 20–30 cm away and turn
the lens until the squares are crisp, then tape the lens so it can't turn. Clean fingerprints off
the lens first: a smudge blurs the whole image.

### 3. Capture

```bash
~/venvs/calib/bin/python calibrate_intrinsics.py capture --camera 0 \
    --width 1920 --height 1080 --square-mm 23.84 --out wrist_cam_calib_1080
```

- **Check it's the right camera.** Cover the lens: the preview should go dark. Camera numbers can
  swap after you replug, and one run calibrated the MacBook camera by mistake.
- **Hold the board 30–50 cm away**, so it fills about a third to half of the image.
- **Tilt it away from the camera**, 30–45°, so one edge is clearly nearer. Spinning it like a
  steering wheel doesn't count. The `tilt` text at the bottom turns green when the tilt is big
  enough, and `near edge: left/right/top/bottom` at the top counts the tilted views.
- **Cover the whole image**, especially the corners and edges, where the distortion is strongest.
  The green shading shows what is covered.
- Hold still briefly at each pose. Frames are saved automatically. Space forces a capture,
  Backspace undoes the last one, Enter solves, q quits.
- To add views to an existing run, use the same command plus `--append`.

### 4. Check the result

`capture` solves when you press Enter (re-run with `solve --square-mm 23.84 --out <folder>`). Check:

| Output | Good |
|---|---|
| RMS reprojection error | under 0.5 px per 640 px of width (1.5 px at 1920) |
| `cx`, `cy` | within ~10% of the image centre |
| `+-` on fx, fy, cx, cy | under 2 px per 640 px of width |
| warnings | none, or only the edge-coverage one with small gaps |

The script warns about each of these. Don't use a result it calls degenerate.

`--rational` (the 8-coefficient wide-angle model) overfits on this camera, so stick with the
default model.

### 5. Write the pinhole target

```bash
~/venvs/calib/bin/python calibrate_intrinsics.py pinhole --out wrist_cam_calib_1080 --focal-px 700
```

This writes `wrist_cam_pinhole_640x480.yaml` and `wrist_cam_pinhole_sample.jpg`, a remapped
calibration image. Check that the board's edges are straight and the corners aren't black. It also
prints the widest focal length the real image fully covers (682 px for the current camera). If a
new camera can't cover 700 px, pick a new focal length, then update `WRIST_FOCAL_MM` in
`arm_params.py` to `focal_px * 20.955 / 640` so the sim matches.

## Gotchas

- **macOS crops instead of scaling.** Asking a 1080p webcam for 640x480 through OpenCV on macOS
  gives an off-centre crop of the sensor, not a scaled image. Calibrations taken that way put the
  optical centre near the top-right corner of the image. Always capture at full resolution.
- **Low error doesn't mean a good calibration.** 40 face-on views gave 0.19 px RMS with a
  principal point 160 px off. Tilted views are what pin down the focal length and centre.
- **Commit the YAMLs, not the images.** `*/images/` is gitignored (~120 MB per run).

## Current wrist camera

Sonix UVC webcam (USB VID 3141, PID 25446), manual-focus lens, calibrated 2026-10-23.

| | Value |
|---|---|
| Raw | 1920x1080, fx 1460, fy 1458, cx 919, cy 491, k1 -0.39 (strong barrel) |
| RMS | 1.21 px over 47 views |
| Raw field of view | 66.6° x 40.6° |
| Pinhole target | 640x480, f 700 px, 49.1° x 37.8° |

## Still to do

- **Hand-eye calibration** (where the camera sits on `link6l`). The sim camera poses are still CAD
  estimates. This step moves the real arm, so follow the real-hardware safety process.
- **More edge views.** The bottom 22% of the image had no corners in this run, so a quick
  `--append` along the bottom edge would firm up the distortion estimate there.
- **Time sync** between camera frames and joint states, for anything that uses the camera while the
  arm moves.
