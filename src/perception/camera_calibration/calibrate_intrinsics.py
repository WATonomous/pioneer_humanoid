#!/usr/bin/env python3
"""Camera intrinsics calibration with a ChArUco board. See README.md for the walkthrough.

  board    printable board PNG
  capture  live capture (auto-saves still, new poses), then solve
  solve    fit from saved images; writes a ROS camera_info YAML to pioneer_humanoid/calib/
  pinhole  ideal pinhole target for wrist_cam.py and the sim, into pioneer_humanoid/calib/,
           plus a sample image in --out
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pioneer_humanoid.camera_info import CALIB_DIR, read_camera_info

# 9x6 squares of 25 mm fits letter and A4. Changing these breaks boards already printed.
SQUARES_X = 9
SQUARES_Y = 6
SQUARE_MM_NOMINAL = 25.0
MARKER_MM_NOMINAL = 18.0
ARUCO_DICT = cv2.aruco.DICT_5X5_100

PRINT_DPI = 300
PAGES_MM = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}  # landscape

SIM_H_APERTURE_MM = 20.955  # horizontal aperture of the sim wrist cams (arm_params.CAMERAS)
PINHOLE_MARGIN = 1.03     # pinhole focal this much above the widest the real image fully covers

MIN_CORNERS = 12          # of (SQUARES_X-1)*(SQUARES_Y-1) = 40; fewer is too weak a view
STILL_PX = 3.0            # mean corner motion between frames (per 640 px of width) for "still"
NOVELTY = 0.08            # how different a pose must be from every saved one to auto-capture
MIN_CAPTURE_GAP_S = 0.4
TARGET_VIEWS = 40
# Face-on views can't separate focal length from distance. Tilt 0.2 ~ 15-20 deg (30 deg ~ 0.35).
TILT = 0.2
VIEWS_PER_TILT = 4
TILT_NAMES = ("left", "right", "top", "bottom")
PREVIEW_MAX_W = 960
MIN_VIEWS = 15


def make_board(square_mm: float = SQUARE_MM_NOMINAL) -> cv2.aruco.CharucoBoard:
    marker_mm = square_mm * MARKER_MM_NOMINAL / SQUARE_MM_NOMINAL
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), square_mm / 1000.0, marker_mm / 1000.0,
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT))


def detect(detector: cv2.aruco.CharucoDetector, gray: np.ndarray):
    """ChArUco corners (N,1,2) and ids (N,1), or (None, None) if too few for a usable view."""
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None, None
    # OpenCV 5 returns (N,2)/(N,); 4.x returned (N,1,2)/(N,1), which drawing still expects.
    return corners.reshape(-1, 1, 2), ids.reshape(-1, 1)


# ----------------------------------------------------------------------------- board

def cmd_board(args) -> int:
    page_w_mm, page_h_mm = PAGES_MM[args.page]
    px_per_mm = PRINT_DPI / 25.4
    page = np.full((round(page_h_mm * px_per_mm), round(page_w_mm * px_per_mm)), 255, np.uint8)

    board_w = round(SQUARES_X * SQUARE_MM_NOMINAL * px_per_mm)
    board_h = round(SQUARES_Y * SQUARE_MM_NOMINAL * px_per_mm)
    img = make_board().generateImage((board_w, board_h), marginSize=0, borderBits=1)
    y0 = (page.shape[0] - board_h) // 2
    x0 = (page.shape[1] - board_w) // 2
    page[y0:y0 + board_h, x0:x0 + board_w] = img

    label = (f"ChArUco {SQUARES_X}x{SQUARES_Y}  DICT_5X5_100  square {SQUARE_MM_NOMINAL:g} mm  "
             f"marker {MARKER_MM_NOMINAL:g} mm  -- print at 100%, then MEASURE a square")
    cv2.putText(page, label, (x0, y0 + board_h + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2,
                cv2.LINE_AA)
    cv2.imwrite(args.out, page)
    print(f"wrote {args.out} ({args.page}, {PRINT_DPI} dpi, landscape)")
    print("Print at 100% / actual size, tape it flat to something rigid, and measure a square.")
    return 0


# ----------------------------------------------------------------------------- capture

def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        sys.exit(f"could not open camera {index} (try --camera 1; on macOS, allow camera "
                 "access for your terminal in System Settings > Privacy & Security)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    focus_locked = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    deadline = time.time() + 3.0  # some UVC cameras on macOS take a moment to send the first frame
    ok, frame = cap.read()
    while not ok and time.time() < deadline:
        time.sleep(0.05)
        ok, frame = cap.read()
    if not ok:
        sys.exit(f"camera {index} opened but returned no frame (unplug and replug it, and check "
                 "no other app is using it)")
    if frame.shape[1] != width or frame.shape[0] != height:
        sys.exit(f"camera gives {frame.shape[1]}x{frame.shape[0]}, asked for {width}x{height}. "
                 "Calibrate at the resolution you will run at; pass --width/--height to match.")
    if not focus_locked:
        print("WARNING: could not confirm autofocus is off. If the image refocuses while you "
              "move the board, lock focus with a UVC tool and start again.")
    return cap


def pose_features(corners: np.ndarray, ids: np.ndarray, board, size) -> np.ndarray:
    """Board centre, size and tilt (log ratio of opposite edges), from a homography: no intrinsics yet."""
    obj = board.getChessboardCorners()[ids.ravel(), :2]
    H, _ = cv2.findHomography(obj, corners.reshape(-1, 2))
    w_m = SQUARES_X * board.getSquareLength()
    h_m = SQUARES_Y * board.getSquareLength()
    outline = np.array([[0, 0], [w_m, 0], [w_m, h_m], [0, h_m]], np.float64).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(outline, H).reshape(4, 2) / np.array(size, np.float64)
    top, right = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1])
    bottom, left = np.linalg.norm(q[3] - q[2]), np.linalg.norm(q[0] - q[3])
    d1, d2 = q[2] - q[0], q[3] - q[1]
    area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
    c = q.mean(axis=0)
    return np.array([c[0], c[1], math.sqrt(area),
                     math.log(max(left, 1e-6) / max(right, 1e-6)),
                     math.log(max(top, 1e-6) / max(bottom, 1e-6))])


def tilt_counts(feats) -> dict[str, int]:
    """Views tilted each way: which edge of the board is nearer the camera."""
    f = np.array(feats).reshape(-1, 5)
    return {"left": int((f[:, 3] > TILT).sum()), "right": int((f[:, 3] < -TILT).sum()),
            "top": int((f[:, 4] > TILT).sum()), "bottom": int((f[:, 4] < -TILT).sum())}


def cmd_capture(args) -> int:
    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(img_dir.glob("view_*.png"))
    if existing and not args.append:
        sys.exit(f"{img_dir} already has {len(existing)} images; pass --append to add to them "
                 "or choose another --out")

    board = make_board(args.square_mm)
    detector = cv2.aruco.CharucoDetector(board)
    cap = open_camera(args.camera, args.width, args.height)
    size = (args.width, args.height)

    # One entry per saved view: (path, pose features, board outline in the image).
    saved: list[tuple[Path, np.ndarray, np.ndarray]] = []
    for p in existing:  # so --append shows what is already covered
        c, i = detect(detector, cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if c is not None:
            saved.append((p, pose_features(c, i, board, size), cv2.convexHull(c.astype(np.int32))))
    n_prior = len(saved)  # Backspace never deletes images from earlier runs (--append)

    prev: dict[int, np.ndarray] = {}
    last_capture = 0.0
    flash_until = 0.0
    win = "intrinsics (space: capture, backspace: undo, enter: solve, q: quit)"
    print("Move the board around: fill the corners and edges of the image, tilt it ~30-45 deg "
          "left/right/up/down, near and far. Hold still briefly at each pose.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera stopped returning frames")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect(detector, gray)

        still, novel, motion = False, False, None
        if corners is not None:
            cur = {int(k): v for k, v in zip(ids.ravel(), corners.reshape(-1, 2))}
            common = cur.keys() & prev.keys()
            if len(common) >= MIN_CORNERS:
                motion = float(np.mean([np.linalg.norm(cur[k] - prev[k]) for k in common]))
                still = motion < STILL_PX * args.width / 640
            prev = cur
            feat = pose_features(corners, ids, board, size)
            novel = all(np.linalg.norm(feat - f) > NOVELTY for _, f, _ in saved)
        else:
            prev = {}

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        manual = key == ord(" ")
        auto = still and novel and now - last_capture > MIN_CAPTURE_GAP_S
        if corners is not None and (manual or auto):
            path = img_dir / f"view_{int(now * 1000)}.png"
            cv2.imwrite(str(path), frame)
            saved.append((path, feat, cv2.convexHull(corners.astype(np.int32))))
            last_capture, flash_until = now, now + 0.15
        elif key == 8 or key == 127:  # backspace / delete: drop the last view
            if len(saved) > n_prior:
                saved.pop()[0].unlink(missing_ok=True)
        elif key in (13, 10):
            break
        elif key in (ord("q"), 27):
            cap.release()
            cv2.destroyAllWindows()
            print(f"quit; {len(saved)} views kept in {img_dir}")
            return 0

        vis = frame.copy()
        tint = np.zeros_like(vis)
        for _, _, hull in saved:
            cv2.fillConvexPoly(tint, hull, (0, 160, 0))
        vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)
        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, (0, 0, 255))
        if vis.shape[1] > PREVIEW_MAX_W:  # full-res frames don't fit on a laptop screen
            vis = cv2.resize(vis, None, fx=PREVIEW_MAX_W / vis.shape[1],
                             fy=PREVIEW_MAX_W / vis.shape[1], interpolation=cv2.INTER_AREA)
        n = len(saved)
        status = "still" if still else "moving" if corners is not None else "no board"
        if motion is not None and not still:
            status += f" ({motion:.1f} px, hold steadier)"
        if corners is not None and still and not novel:
            status = "seen this pose, move on"
        if corners is not None:  # live tilt, so "turned away" vs "spun in the image" is visible
            tilt_lr, tilt_tb = feat[3], feat[4]
            ok_tilt = max(abs(tilt_lr), abs(tilt_tb)) > TILT
            cv2.putText(vis, f"tilt L/R {tilt_lr:+.2f}  T/B {tilt_tb:+.2f}  (need +-{TILT})",
                        (10, vis.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 0) if ok_tilt else (0, 200, 255), 2, cv2.LINE_AA)
        colour = (0, 200, 0) if n >= MIN_VIEWS else (0, 200, 255)
        cv2.putText(vis, f"views {n}/{TARGET_VIEWS}  {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        tilts = tilt_counts([f for _, f, _ in saved])
        x = 10
        cv2.putText(vis, "near edge:", (x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        x += 110
        for name in TILT_NAMES:
            k = min(tilts[name], VIEWS_PER_TILT)
            done = k >= VIEWS_PER_TILT
            cv2.putText(vis, f"{name} {k}/{VIEWS_PER_TILT}", (x, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 0) if done else (0, 200, 255), 2, cv2.LINE_AA)
            x += 125
        all_tilts = all(v >= VIEWS_PER_TILT for v in tilts.values())
        if n >= TARGET_VIEWS and all_tilts:
            cv2.putText(vis, "enough views - press Enter to solve", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)
        elif n >= TARGET_VIEWS:
            cv2.putText(vis, "need more tilted views (orange above)", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
        if now < flash_until:
            vis = cv2.bitwise_not(vis)
        cv2.imshow(win, vis)

    cap.release()
    cv2.destroyAllWindows()
    if len(saved) < MIN_VIEWS:
        print(f"only {len(saved)} views (need {MIN_VIEWS}); run capture again with --append")
        return 1
    return cmd_solve(args)


# ----------------------------------------------------------------------------- solve

def per_view_rms(obj, img, rvecs, tvecs, K, dist) -> np.ndarray:
    errs = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        errs.append(math.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1))))
    return np.array(errs)


def calibrate(obj, img, size, flags):
    rms, K, dist, rvecs, tvecs, std_intr, _, _ = cv2.calibrateCameraExtended(
        obj, img, size, None, None, flags=flags)
    return rms, K, dist, rvecs, tvecs, std_intr.ravel()


def write_camera_info(path: Path, name: str, size, K, dist, model: str, notes: list[str]):
    w, h = size
    P = np.zeros((3, 4))
    P[:, :3] = K

    def row(vals):
        return "[" + ", ".join(f"{v:.8g}" for v in vals) + "]"

    lines = [f"# {n}" for n in notes] + [
        f"image_width: {w}",
        f"image_height: {h}",
        f"camera_name: {name}",
        "camera_matrix:", "  rows: 3", "  cols: 3", f"  data: {row(K.ravel())}",
        f"distortion_model: {model}",
        "distortion_coefficients:", "  rows: 1", f"  cols: {dist.size}",
        f"  data: {row(dist.ravel())}",
        "rectification_matrix:", "  rows: 3", "  cols: 3", f"  data: {row(np.eye(3).ravel())}",
        "projection_matrix:", "  rows: 3", "  cols: 4", f"  data: {row(P.ravel())}",
    ]
    path.write_text("\n".join(lines) + "\n")


def cmd_solve(args) -> int:
    out = Path(args.out)
    paths = sorted((out / "images").glob("view_*.png"))
    if not paths:
        sys.exit(f"no images in {out / 'images'}; run capture first")

    board = make_board(args.square_mm)
    detector = cv2.aruco.CharucoDetector(board)
    obj, img, used, feats, size = [], [], [], [], None
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != size:
            sys.exit(f"{p.name} is {gray.shape[1]}x{gray.shape[0]}, others are {size[0]}x{size[1]}")
        corners, ids = detect(detector, gray)
        if corners is None:
            print(f"  skip {p.name}: board not found")
            continue
        o, i = board.matchImagePoints(corners, ids)
        feats.append(pose_features(corners, ids, board, size))
        obj.append(o.astype(np.float32))
        img.append(i.astype(np.float32))
        used.append(p)
    if len(used) < MIN_VIEWS:
        print(f"only {len(used)} usable views (need {MIN_VIEWS})")
        return 1

    model = "rational_polynomial" if args.rational else "plumb_bob"
    flags = cv2.CALIB_RATIONAL_MODEL if args.rational else 0
    rms, K, dist, rvecs, tvecs, std = calibrate(obj, img, size, flags)

    # One pass of outlier rejection: a blurred or misdetected view drags the whole fit.
    errs = per_view_rms(obj, img, rvecs, tvecs, K, dist)
    bad = errs > max(3.0 * np.median(errs), 1.0)
    if bad.any() and len(used) - bad.sum() >= MIN_VIEWS:
        for p, e in zip(used, errs):
            if e > max(3.0 * np.median(errs), 1.0):
                print(f"  dropping {p.name}: {e:.2f} px")
        keep = ~bad
        obj = [o for o, k in zip(obj, keep) if k]
        img = [i for i, k in zip(img, keep) if k]
        used = [p for p, k in zip(used, keep) if k]
        rms, K, dist, rvecs, tvecs, std = calibrate(obj, img, size, flags)
        errs = per_view_rms(obj, img, rvecs, tvecs, K, dist)

    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    hfov = math.degrees(2 * math.atan(w / (2 * fx)))
    vfov = math.degrees(2 * math.atan(h / (2 * fy)))

    # How much of the image the corners reached: an uncovered edge leaves distortion unknown.
    pts = np.concatenate([i.reshape(-1, 2) for i in img])
    reach = (pts[:, 0].min() / w, 1 - pts[:, 0].max() / w, pts[:, 1].min() / h,
             1 - pts[:, 1].max() / h)

    print(f"\n{len(used)} views, {model}")
    print(f"  RMS reprojection error  {rms:.3f} px   (worst view {errs.max():.2f} px)")
    print(f"  fx {fx:.2f} +- {std[0]:.2f}   fy {fy:.2f} +- {std[1]:.2f}")
    print(f"  cx {cx:.2f} +- {std[2]:.2f}   cy {cy:.2f} +- {std[3]:.2f}   (image centre "
          f"{w / 2:.0f}, {h / 2:.0f})")
    print(f"  dist {np.array2string(dist.ravel(), precision=5)}")
    print(f"  FOV {hfov:.1f} x {vfov:.1f} deg   (sim values come from the pinhole step)")

    px_scale = w / 640  # pixel thresholds below are for a 640-wide image
    if rms > 0.5 * px_scale:
        hint = " Wide lens? try --rational." if not args.rational else ""
        print(f"WARNING: RMS {rms:.2f} px is above {0.5 * px_scale:.2f} px. Check for blur, a board that is not "
              f"flat, or autofocus hunting.{hint}")
    if max(reach) > 0.15:
        print("WARNING: corners never got within 15% of some image edge "
              f"(left/right/top/bottom gaps {', '.join(f'{r:.0%}' for r in reach)}). "
              "Capture more views near the edges, or the distortion there is a guess.")
    tilts = tilt_counts(feats)
    if min(tilts.values()) < VIEWS_PER_TILT // 2:
        print("WARNING: too few tilted views (near edge: "
              + ", ".join(f"{k} {v}" for k, v in tilts.items())
              + "). Mostly face-on views can't pin down focal length or centre; recapture with "
              "the board tilted 30-45 deg each way.")
    if abs(cx - w / 2) > 0.1 * w or abs(cy - h / 2) > 0.1 * h:
        print(f"WARNING: principal point ({cx:.0f}, {cy:.0f}) is far from the image centre. "
              "The fit is almost certainly degenerate: do not use this result.")
    if max(std[:4]) > 2.0 * px_scale:
        print(f"WARNING: focal length / principal point uncertainty above {2.0 * px_scale:.1f} px: "
              "add tilted views.")

    yaml_path = Path(args.calib_dir) / f"{args.name}.yaml"
    notes = [f"intrinsics from {Path(__file__).name} on {datetime.date.today().isoformat()}",
             f"rms {rms:.4f} px over {len(used)} views; ChArUco {SQUARES_X}x{SQUARES_Y} "
             f"square {args.square_mm} mm",
             f"hfov {hfov:.2f} deg, vfov {vfov:.2f} deg"]
    write_camera_info(yaml_path, args.name, size, K, dist, model, notes)
    print(f"\nwrote {yaml_path}")
    return 0


def widest_pinhole_focal(size, K, dist, out_size) -> float:
    """Smallest focal (px, at out_size) whose centred, undistorted view the real image fully covers."""
    w, h = size
    n = 400
    edge = np.linspace(0, 1, n)
    border = np.concatenate([np.stack([edge * (w - 1), np.zeros(n)], 1),
                             np.stack([edge * (w - 1), np.full(n, h - 1.0)], 1),
                             np.stack([np.zeros(n), edge * (h - 1)], 1),
                             np.stack([np.full(n, w - 1.0), edge * (h - 1)], 1)])
    xy = cv2.undistortPoints(border.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    aspect = out_size[1] / out_size[0]
    # A border point cuts into the centred rectangle (half-width a) only if it is inside on both axes.
    a = np.maximum(np.abs(xy[:, 0]), np.abs(xy[:, 1]) / aspect).min()
    return (out_size[0] / 2) / a


def cmd_pinhole(args) -> int:
    """Write the ideal pinhole camera real frames are remapped to, which is what the sim renders."""
    out, calib_dir = Path(args.out), Path(args.calib_dir)
    size, K, dist = read_camera_info(calib_dir / f"{args.name}.yaml")
    K, dist = np.array(K), np.array(dist)
    ow, oh = args.size
    widest = widest_pinhole_focal(size, K, dist, (ow, oh))
    f = args.focal_px if args.focal_px else round(widest * PINHOLE_MARGIN)
    if f < widest:
        print(f"WARNING: focal {f} px is below {widest:.1f} px: the corners will be black (outside "
              "the real image)")
    K_new = np.array([[f, 0, ow / 2], [0, f, oh / 2], [0, 0, 1]], np.float64)
    hfov = math.degrees(2 * math.atan(ow / (2 * f)))
    vfov = math.degrees(2 * math.atan(oh / (2 * f)))

    path = calib_dir / f"{args.name}_pinhole_{ow}x{oh}.yaml"
    notes = [f"ideal pinhole that {args.name}.yaml frames are remapped to (zero distortion)",
             "real frames: cv2.initUndistortRectifyMap(K, dist, None, K_new, (w, h), cv2.CV_16SC2) "
             "then cv2.remap",
             f"hfov {hfov:.2f} deg, vfov {vfov:.2f} deg; widest fully-covered focal {widest:.1f} px",
             f"Isaac: focal_length={f * SIM_H_APERTURE_MM / ow:.4f} mm, "
             f"horizontal_aperture={SIM_H_APERTURE_MM}, vertical_aperture={SIM_H_APERTURE_MM * oh / ow:.4f}"]
    write_camera_info(path, args.name, (ow, oh), K_new, np.zeros(5), "plumb_bob", notes)
    print(f"wrote {path}")
    for n in notes[2:]:
        print(f"  {n}")

    # A remapped sample so the result can be eyeballed: straight lines straight, no black corners.
    samples = sorted((out / "images").glob("view_*.png"))
    if samples:
        m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (ow, oh), cv2.CV_16SC2)
        img = cv2.imread(str(samples[len(samples) // 2]))
        sample = out / f"{args.name}_pinhole_sample.jpg"
        cv2.imwrite(str(sample), cv2.remap(img, m1, m2, cv2.INTER_LINEAR))
        print(f"  sample: {sample}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("board", help="write a printable ChArUco board PNG")
    b.add_argument("--out", default="charuco_board.png")
    b.add_argument("--page", choices=sorted(PAGES_MM), default="letter")
    b.set_defaults(func=cmd_board)

    def common(p):
        p.add_argument("--square-mm", type=float, required=True,
                       help="measured size of one printed square, in mm")
        p.add_argument("--out", default="wrist_cam_calib", help="folder for the captured images")
        p.add_argument("--calib-dir", default=str(CALIB_DIR), help="where the result YAML goes")
        p.add_argument("--name", default="wrist_cam", help="camera_name in the output YAML")
        p.add_argument("--rational", action="store_true",
                       help="8-coefficient distortion model, for wide-angle lenses")

    c = sub.add_parser("capture", help="live capture, then solve")
    common(c)
    c.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    # Full sensor by default: smaller sizes are an off-centre crop on macOS (README gotchas).
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=1080)
    c.add_argument("--append", action="store_true", help="add to images already in --out")
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser("solve", help="fit from images saved by capture")
    common(s)
    s.set_defaults(func=cmd_solve)

    ph = sub.add_parser("pinhole", help="write the undistorted pinhole camera the sim matches")
    ph.add_argument("--out", default="wrist_cam_calib", help="image folder, for the sample image")
    ph.add_argument("--calib-dir", default=str(CALIB_DIR), help="holds <name>.yaml; pinhole YAML goes here too")
    ph.add_argument("--name", default="wrist_cam")
    ph.add_argument("--size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ph.add_argument("--focal-px", type=float, default=None,
                    help=f"pinhole focal at --size (default: widest covered x {PINHOLE_MARGIN})")
    ph.set_defaults(func=cmd_pinhole)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
