"""Evaluate shuttlecock weights against hand-labeled frames.

Ground truth comes from either:
  --manifest: the JSONL written by prepare_eval.py / annotate_eval.py, or
  --yolo_dir + --split: a YOLO dataset split (for example the PAC 5 val split),
    so thresholds can be tuned without touching the held-out Ticket 3 set.

Positive frames need a center in original image pixels. Negative frames count
any detection as a false positive. A positive is recalled only when the
highest-confidence prediction is within match_radius pixels of the labeled
center. A positive frame whose top prediction is farther away is counted as a
miss and also as "mislocalized".

Inference runs once at the lowest requested threshold. --sweep then reports
metrics at several thresholds from the same predictions.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path, PureWindowsPath

from PIL import Image
from ultralytics import YOLO


PREFERRED_CLASS_NAMES = ("shuttlecock", "birdie", "bird")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_image(path: str, image_root: Path | None) -> str:
    """Return a readable image path, remapping to image_root when needed.

    Manifests store absolute paths, which break when a dataset moves between
    Linux and Windows. PureWindowsPath splits on both / and \\.
    """
    if image_root is not None:
        candidate = image_root / PureWindowsPath(path).name
        if candidate.is_file():
            return str(candidate)
    return path


def load_rows(path: Path, image_root: Path | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as manifest:
        rows = [json.loads(line) for line in manifest if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for row in rows:
        row["image"] = resolve_image(row["image"], image_root)
    return rows


def video_from_stem(stem: str) -> str:
    return stem.rsplit("_f", 1)[0]


def load_yolo_split(dataset_dir: Path, split: str) -> list[dict]:
    """Turn a YOLO split into manifest-style rows (skips offline _aug copies)."""
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split
    rows = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or "_aug" in image_path.stem:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        row = {"image": str(image_path), "video": video_from_stem(image_path.stem), "label": "negative", "bbox": None}
        if lines:
            width, height = Image.open(image_path).size
            _, cx, cy, w, h = map(float, lines[0].split())
            row.update({
                "label": "positive",
                "center_x": cx * width,
                "center_y": cy * height,
                "bbox": [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
            })
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No images in {images_dir}")
    return rows


def resolve_class_id(model, requested_id: int | None, requested_name: str | None) -> int:
    if requested_id is not None:
        return requested_id

    names = model.names
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    normalized = {str(name).lower(): int(index) for index, name in names.items()}

    if requested_name:
        if requested_name.lower() not in normalized:
            raise ValueError(f"Class {requested_name!r} not found in model names: {names}")
        return normalized[requested_name.lower()]

    for name in PREFERRED_CLASS_NAMES:
        if name in normalized:
            return normalized[name]
    if len(normalized) == 1:
        return next(iter(normalized.values()))
    raise ValueError(
        "Could not infer shuttle class. Pass --class_id or --class_name. "
        f"Model classes: {names}"
    )


def detections(result, class_id: int) -> list[dict]:
    """All detections of class_id, highest confidence first."""
    boxes = result.boxes
    found = []
    for box, confidence, predicted_class in zip(boxes.xyxy, boxes.conf, boxes.cls):
        if int(predicted_class) != class_id:
            continue
        x0, y0, x1, y1 = map(float, box.tolist())
        found.append({
            "x": (x0 + x1) / 2.0,
            "y": (y0 + y1) / 2.0,
            "box": [x0, y0, x1, y1],
            "confidence": float(confidence),
        })
    return sorted(found, key=lambda item: item["confidence"], reverse=True)


def top_prediction(candidates: list[dict], threshold: float):
    """Highest-confidence candidate at or above threshold (candidates are sorted)."""
    if candidates and candidates[0]["confidence"] >= threshold:
        return candidates[0]
    return None


def classify(row: dict, prediction, match_radius: float) -> tuple[str, float | None]:
    """Return (outcome, pixel_error). Outcomes: tp, fn, mislocalized, fp, tn."""
    if row["label"] == "negative":
        return ("fp" if prediction else "tn"), None
    if prediction is None:
        return "fn", None
    error = math.hypot(prediction["x"] - float(row["center_x"]), prediction["y"] - float(row["center_y"]))
    return ("tp" if error <= match_radius else "mislocalized"), error


def metric_summary(rows: list[dict], predictions: list, match_radius: float) -> dict:
    counts = {"tp": 0, "fn": 0, "mislocalized": 0, "fp": 0, "tn": 0}
    errors = []
    for row, prediction in zip(rows, predictions):
        if row.get("label") not in {"positive", "negative"}:
            raise ValueError(f"Every row must have label=positive or negative: {row}")
        outcome, error = classify(row, prediction, match_radius)
        counts[outcome] += 1
        if outcome == "tp":
            errors.append(error)

    positive = counts["tp"] + counts["fn"] + counts["mislocalized"]
    negative = counts["fp"] + counts["tn"]
    outputs = counts["tp"] + counts["mislocalized"] + counts["fp"]
    errors.sort()
    return {
        "positive_frames": positive,
        "negative_frames": negative,
        "true_positives": counts["tp"],
        # Misses include mislocalized frames, matching the original definition.
        "false_negatives": counts["fn"] + counts["mislocalized"],
        "mislocalized": counts["mislocalized"],
        "false_positives": counts["fp"],
        "recall": counts["tp"] / positive if positive else None,
        "false_positive_rate": counts["fp"] / negative if negative else None,
        # Of the frames where the model output a shuttle, how often it was right.
        "precision": counts["tp"] / outputs if outputs else None,
        "pixel_error_mean": statistics.fmean(errors) if errors else None,
        "pixel_error_median": statistics.median(errors) if errors else None,
        "pixel_error_p90": errors[max(0, math.ceil(len(errors) * 0.9) - 1)] if errors else None,
        "matched_frames": len(errors),
    }


def evaluate(args) -> None:
    if args.manifest:
        rows = load_rows(args.manifest, args.image_root)
        source = str(args.manifest.resolve())
    else:
        rows = load_yolo_split(args.yolo_dir.resolve(), args.split)
        source = str((args.yolo_dir / "images" / args.split).resolve())

    thresholds = sorted(set([args.confidence, *args.sweep]))
    model = YOLO(str(args.weights))
    class_id = resolve_class_id(model, args.class_id, args.class_name)

    all_candidates = []
    inference_ms = []
    total_ms = []  # preprocess + inference + postprocess
    for row in rows:
        result = model.predict(
            row["image"], imgsz=args.imgsz, conf=thresholds[0], device=args.device,
            half=args.half, verbose=False,
        )[0]
        all_candidates.append(detections(result, class_id))
        inference_ms.append(result.speed["inference"])
        total_ms.append(sum(result.speed.values()))

    def predictions_at(threshold: float) -> list:
        return [top_prediction(candidates, threshold) for candidates in all_candidates]

    primary = predictions_at(args.confidence)
    summary = {
        "match_radius_pixels": args.match_radius,
        "overall": metric_summary(rows, primary, args.match_radius),
        "per_video": {},
        "sweep": {},
    }
    for video in sorted({row["video"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["video"] == video]
        summary["per_video"][video] = metric_summary(
            [rows[i] for i in indices], [primary[i] for i in indices], args.match_radius
        )
    for threshold in thresholds:
        summary["sweep"][f"{threshold:.3f}"] = metric_summary(rows, predictions_at(threshold), args.match_radius)

    # Skip the first image: it includes CUDA warm-up.
    timed = inference_ms[1:] or inference_ms
    timed_total = total_ms[1:] or total_ms
    summary.update({
        "model": str(args.weights.resolve()),
        "source": source,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "class_id": class_id,
        "half": args.half,
        "inference_ms_mean": statistics.fmean(timed),
        "inference_ms_median": statistics.median(timed),
        "model_total_ms_mean": statistics.fmean(timed_total),
        "max_model_fps": 1000.0 / statistics.fmean(timed_total),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    details_path = args.output.with_name(args.output.stem + "_frames.csv")
    with details_path.open("w", newline="", encoding="utf-8") as details_file:
        writer = csv.DictWriter(details_file, fieldnames=[
            "image", "video", "label", "outcome", "confidence", "pred_x", "pred_y", "pixel_error", "detections",
        ])
        writer.writeheader()
        for row, prediction, candidates in zip(rows, primary, all_candidates):
            outcome, error = classify(row, prediction, args.match_radius)
            writer.writerow({
                "image": row["image"],
                "video": row["video"],
                "label": row["label"],
                "outcome": outcome,
                "confidence": prediction["confidence"] if prediction else None,
                "pred_x": prediction["x"] if prediction else None,
                "pred_y": prediction["y"] if prediction else None,
                "pixel_error": error,
                "detections": sum(c["confidence"] >= args.confidence for c in candidates),
            })

    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(
        f"\nInference: {summary['inference_ms_mean']:.1f} ms mean per image at imgsz={args.imgsz} "
        f"({summary['model_total_ms_mean']:.1f} ms with pre/post-processing, "
        f"max {summary['max_model_fps']:.0f} FPS). Run visualize_predictions.py for the real-time factor on video."
    )
    print("\nconf   recall  FP-rate  precision  mislocalized")
    for key, values in summary["sweep"].items():
        fpr = values["false_positive_rate"]
        precision = values["precision"]
        print(
            f"{key}  {values['recall'] or 0:6.3f}  "
            f"{'-' if fpr is None else f'{fpr:7.3f}'}  "
            f"{'-' if precision is None else f'{precision:9.3f}'}  {values['mislocalized']:12d}"
        )
    print(f"\nWrote {args.output}")
    print(f"Wrote {details_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Hand-labeled JSONL manifest.")
    source.add_argument("--yolo_dir", type=Path, help="YOLO dataset folder (with images/ and labels/).")
    parser.add_argument("--split", default="val", help="Split to read with --yolo_dir.")
    parser.add_argument("--image_root", type=Path, help="Folder to find manifest images in if their stored paths moved.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--sweep", type=lambda text: [float(v) for v in text.split(",") if v], default=[],
        help="Comma-separated extra thresholds to report, e.g. 0.05,0.1,0.15,0.2,0.3",
    )
    parser.add_argument("--match_radius", type=float, default=50.0)
    parser.add_argument("--class_id", type=int)
    parser.add_argument("--class_name")
    parser.add_argument("--device")
    parser.add_argument("--half", action="store_true", help="FP16 inference (CUDA only).")
    evaluate(parser.parse_args())
