"""Render shuttlecock detections into a video and measure the real-time factor.

Two modes:

1. Full video (every frame of the held-out clips):
     python visualize_predictions.py --weights best.pt \
       --video "C:/datasets/raw_videos/PAC 1.MP4" --video "C:/datasets/raw_videos/PAC 6.MP4" \
       --manifest C:/datasets/badminton_eval/manifest.jsonl --output_dir C:/datasets/badminton_eval/videos
   --manifest is optional here. When given, frames that were hand-labeled also
   show the ground-truth box and a TP/FN/FP/TN badge.

2. Labeled frames only (no raw video needed):
     python visualize_predictions.py --weights best.pt \
       --manifest C:/datasets/badminton_eval/manifest.jsonl --output_dir ... [--only_errors]
   Writes one slideshow per source video from the sampled eval images.

Drawing:
  - yellow box: top prediction (the one eval_metrics.py scores)
  - orange boxes: other detections above --confidence
  - green box: hand label (labeled frames only)
  - dots: the last --trail top-prediction centers

Timing (video mode): the real-time factor (RTF) is processing time divided by
video duration. RTF < 1 means the model keeps up with the camera. It is
reported for the model alone (preprocess + inference + postprocess) and for
model + video decoding. Drawing and writing the output video are excluded.
"""

import argparse
import csv
import json
import statistics
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from eval_metrics import classify, detections, load_rows, resolve_class_id

YELLOW = (0, 255, 255)
ORANGE = (0, 140, 255)
GREEN = (0, 220, 0)
OUTCOME_COLORS = {
    "tp": (0, 200, 0),
    "tn": (0, 200, 0),
    "fn": (0, 0, 255),
    "fp": (0, 0, 255),
    "mislocalized": (0, 0, 255),
}


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    # H.264 plays everywhere but needs an OpenCV build with an encoder; mp4v always works locally.
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open a video writer for {path}")


class Renderer:
    def __init__(self, args):
        self.args = args
        self.trail = deque(maxlen=max(1, args.trail))
        self.scale = 1.0

    def setup(self, width: int):
        self.scale = self.args.output_width / width if self.args.output_width else 1.0
        self.trail.clear()

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        return round(width * self.scale), round(height * self.scale)

    def pt(self, x: float, y: float) -> tuple[int, int]:
        return round(x * self.scale), round(y * self.scale)

    def box(self, image, box, color, thickness, text=None):
        x0, y0, x1, y1 = box
        p0, p1 = self.pt(x0, y0), self.pt(x1, y1)
        # Pad tiny boxes so a 10 px shuttle is still visible in the video.
        pad = 6
        cv2.rectangle(image, (p0[0] - pad, p0[1] - pad), (p1[0] + pad, p1[1] + pad), color, thickness)
        if text:
            cv2.putText(image, text, (p0[0] - pad, p0[1] - pad - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw(self, frame, candidates, threshold, header, row=None, outcome=None):
        image = cv2.resize(frame, None, fx=self.scale, fy=self.scale) if self.scale != 1.0 else frame.copy()
        shown = [c for c in candidates if c["confidence"] >= threshold]
        top = shown[0] if shown else None

        if self.args.trail > 0:
            self.trail.append((top["x"], top["y"]) if top else None)
            points = list(self.trail)
            for age, point in enumerate(points):
                if point is not None:
                    radius = 2 + round(4 * (age + 1) / len(points))
                    cv2.circle(image, self.pt(*point), radius, YELLOW, -1)

        if row is not None and row.get("bbox"):
            self.box(image, row["bbox"], GREEN, 2)
        for candidate in shown[1:]:
            self.box(image, candidate["box"], ORANGE, 1, f"{candidate['confidence']:.2f}")
        if top:
            self.box(image, top["box"], YELLOW, 2, f"shuttle {top['confidence']:.2f}")

        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(image, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if outcome:
            label = {"mislocalized": "WRONG SPOT"}.get(outcome, outcome.upper())
            text = f"LABELED: {label}" + (" (no shuttle)" if row["label"] == "negative" else "")
            (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(image, (image.shape[1] - w - 20, 40), (image.shape[1], 76), OUTCOME_COLORS[outcome], -1)
            cv2.putText(image, text, (image.shape[1] - w - 10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return image


def predict(model, frame, args, class_id):
    extra = {"half": True} if args.half else {}
    result = model.predict(frame, imgsz=args.imgsz, conf=args.confidence, device=args.device, verbose=False, **extra)[0]
    return detections(result, class_id), result.speed


def summarize_timing(name, model_ms, pipeline_ms, speeds, video_fps):
    model_ms, pipeline_ms = model_ms[1:] or model_ms, pipeline_ms[1:] or pipeline_ms  # drop warm-up
    speeds = speeds[1:] or speeds
    breakdown = {key: statistics.fmean(s[key] for s in speeds) for key in speeds[0]}
    budget = 1000.0 / video_fps
    mean_model = statistics.fmean(model_ms)
    mean_pipe = statistics.fmean(pipeline_ms)
    p95 = sorted(model_ms)[max(0, round(0.95 * len(model_ms)) - 1)]
    stats = {
        "video_fps": video_fps,
        "frame_budget_ms": budget,
        "model_ms_mean": mean_model,
        "model_ms_p95": p95,
        "model_ms_max": max(model_ms),
        "model_fps": 1000.0 / mean_model,
        "rtf_model": mean_model / budget,
        "decode_plus_model_ms_mean": mean_pipe,
        "rtf_decode_plus_model": mean_pipe / budget,
        "frames_over_budget": sum(ms > budget for ms in model_ms) / len(model_ms),
        "ultralytics_ms_mean": breakdown,  # preprocess / inference / postprocess
    }
    print(
        f"{name}: model {mean_model:.1f} ms/frame (p95 {p95:.1f}, max {max(model_ms):.1f}) = {1000 / mean_model:.0f} FPS "
        f"vs {video_fps:.0f} FPS video -> RTF {stats['rtf_model']:.2f} "
        f"({1 / stats['rtf_model']:.1f}x real time); with decoding RTF {stats['rtf_decode_plus_model']:.2f}; "
        f"{100 * stats['frames_over_budget']:.1f}% of frames over the {budget:.1f} ms budget"
    )
    print("  ultralytics breakdown (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in breakdown.items()))
    return stats


def render_video(model, class_id, video_path: Path, labels: dict, args, renderer: Renderer) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = round(args.start_s * fps)
    end = min(total, round(args.end_s * fps)) if args.end_s else total
    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    renderer.setup(width)
    stem = video_path.stem
    writer = None if args.no_video else open_writer(
        args.output_dir / f"{stem}_pred.mp4", fps, renderer.output_size(width, height)
    )
    csv_file = (args.output_dir / f"{stem}_pred.csv").open("w", newline="", encoding="utf-8")
    log = csv.writer(csv_file)
    log.writerow(["frame", "time_s", "detections", "confidence", "x", "y", "x0", "y0", "x1", "y1", "model_ms"])

    model_ms, pipeline_ms, speeds, tally = [], [], [], {}
    for frame_index in range(start, end):
        t0 = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        t1 = time.perf_counter()
        candidates, speed = predict(model, frame, args, class_id)
        t2 = time.perf_counter()
        speeds.append(speed)
        model_ms.append(1000 * (t2 - t1))
        pipeline_ms.append(1000 * (t2 - t0))

        row = labels.get((video_path.name.lower(), frame_index))
        top = candidates[0] if candidates else None
        outcome = classify(row, top, args.match_radius)[0] if row else None
        if outcome:
            tally[outcome] = tally.get(outcome, 0) + 1
        log.writerow([
            frame_index, f"{frame_index / fps:.3f}", len(candidates),
            *( [f"{top['confidence']:.4f}", f"{top['x']:.1f}", f"{top['y']:.1f}", *(f"{v:.1f}" for v in top["box"])]
               if top else [""] * 7 ),
            f"{model_ms[-1]:.2f}",
        ])
        if writer is not None:
            header = (
                f"{stem}  frame {frame_index}/{total}  t={frame_index / fps:6.2f}s  "
                f"model {model_ms[-1]:5.1f} ms  dets {len(candidates)}"
            )
            writer.write(renderer.draw(frame, candidates, args.confidence, header, row, outcome))
        if frame_index % 200 == 0:
            print(f"  {stem}: frame {frame_index}/{end}", flush=True)

    capture.release()
    csv_file.close()
    if writer is not None:
        writer.release()
        print(f"Wrote {args.output_dir / f'{stem}_pred.mp4'}")
    if tally:
        print(f"  labeled frames in {stem}: {tally}")
    return summarize_timing(stem, model_ms, pipeline_ms, speeds, fps)


def render_labeled_frames(model, class_id, rows: list[dict], args, renderer: Renderer) -> None:
    for video in sorted({row["video"] for row in rows}):
        video_rows = sorted((r for r in rows if r["video"] == video), key=lambda r: r.get("frame_index", 0))
        stem = Path(video).stem
        path = args.output_dir / f"{stem}_labeled{'_errors' if args.only_errors else ''}.mp4"
        writer, written = None, 0
        for number, row in enumerate(video_rows, 1):
            frame = cv2.imread(row["image"])
            if frame is None:
                raise RuntimeError(f"Could not read {row['image']}")
            candidates, _ = predict(model, frame, args, class_id)
            outcome, error = classify(row, candidates[0] if candidates else None, args.match_radius)
            if args.only_errors and outcome in ("tp", "tn"):
                continue
            if writer is None:
                renderer.setup(frame.shape[1])
                writer = open_writer(path, args.slideshow_fps, renderer.output_size(frame.shape[1], frame.shape[0]))
            header = f"{Path(row['image']).name}  ({number}/{len(video_rows)})"
            if error is not None:
                header += f"  error {error:.1f}px"
            writer.write(renderer.draw(frame, candidates, args.confidence, header, row, outcome))
            written += 1
        if writer is not None:
            writer.release()
            print(f"Wrote {path} ({written} frames)")
        else:
            print(f"{stem}: no frames to write")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", default=[], help="Video to render; repeat for several.")
    parser.add_argument("--manifest", type=Path, help="Hand-labeled manifest for ground-truth overlays.")
    parser.add_argument("--image_root", type=Path, help="Folder to find manifest images in if their stored paths moved.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match_radius", type=float, default=50.0)
    parser.add_argument("--class_id", type=int)
    parser.add_argument("--class_name")
    parser.add_argument("--device")
    parser.add_argument("--half", action="store_true", help="FP16 inference (CUDA only).")
    parser.add_argument(
        "--output_width", type=int, default=1280,
        help="Output video width (0 keeps 1920). OpenCV's H.264 bitrate is high, so full size is ~15 MB/s.",
    )
    parser.add_argument("--trail", type=int, default=12, help="How many past centers to draw; 0 disables.")
    parser.add_argument("--start_s", type=float, default=0.0)
    parser.add_argument("--end_s", type=float)
    parser.add_argument("--no_video", action="store_true", help="Only time the model and write the CSV.")
    parser.add_argument("--slideshow_fps", type=float, default=2.0)
    parser.add_argument("--only_errors", action="store_true", help="Labeled-frame mode: keep only misses and false alarms.")
    args = parser.parse_args()
    if not args.video and not args.manifest:
        parser.error("Pass --video, --manifest, or both")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    class_id = resolve_class_id(model, args.class_id, args.class_name)
    rows = load_rows(args.manifest, args.image_root) if args.manifest else []
    renderer = Renderer(args)

    if not args.video:
        render_labeled_frames(model, class_id, rows, args, renderer)
        return

    labels = {(row["video"].lower(), row["frame_index"]): row for row in rows if "frame_index" in row}
    timing = {}
    for video_path in args.video:
        timing[video_path.name] = render_video(model, class_id, video_path, labels, args, renderer)
    timing_path = args.output_dir / "timing.json"
    timing_path.write_text(json.dumps({
        "weights": str(args.weights.resolve()), "imgsz": args.imgsz, "half": args.half, "videos": timing,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {timing_path}")


if __name__ == "__main__":
    main()
