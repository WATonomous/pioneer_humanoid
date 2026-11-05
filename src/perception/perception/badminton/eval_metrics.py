"""Evaluate shuttlecock weights against a manually labeled JSONL manifest.

Positive frames require a center_x/center_y in original image pixels.
Negative frames use label=negative and count detections as false positives.
A positive is recalled only when the selected prediction is within match_radius
pixels of the hand-labeled center.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from ultralytics import YOLO


PREFERRED_CLASS_NAMES = ("shuttlecock", "birdie", "bird")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as manifest:
        rows = [json.loads(line) for line in manifest if line.strip()]
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
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


def prediction_center(result, class_id: int):
    boxes = result.boxes
    candidates = []
    for box, confidence, predicted_class in zip(boxes.xyxy, boxes.conf, boxes.cls):
        if int(predicted_class) != class_id:
            continue
        x0, y0, x1, y1 = map(float, box.tolist())
        candidates.append(
            {
                "x": (x0 + x1) / 2.0,
                "y": (y0 + y1) / 2.0,
                "confidence": float(confidence),
            }
        )
    return max(candidates, key=lambda item: item["confidence"]) if candidates else None


def metric_summary(rows: list[dict], predictions: list[dict], match_radius: float) -> dict:
    counts = {"positive": 0, "negative": 0, "tp": 0, "fn": 0, "fp": 0}
    errors = []

    for row, prediction in zip(rows, predictions):
        label = row.get("label")
        if label not in {"positive", "negative"}:
            raise ValueError(f"Every row must have label=positive or negative: {row}")

        if label == "negative":
            counts["negative"] += 1
            is_fp = prediction is not None
            counts["fp"] += int(is_fp)
            continue

        counts["positive"] += 1
        dx = prediction["x"] - float(row["center_x"]) if prediction else None
        dy = prediction["y"] - float(row["center_y"]) if prediction else None
        error = math.hypot(dx, dy) if dx is not None else None
        hit = error is not None and error <= match_radius
        counts["tp"] += int(hit)
        counts["fn"] += int(not hit)
        if hit:
            errors.append(error)

    def finalize(bucket: dict) -> dict:
        positive = bucket["positive"]
        negative = bucket["negative"]
        tp = bucket["tp"]
        errors_for_bucket = sorted(bucket["errors"])
        return {
            "positive_frames": positive,
            "negative_frames": negative,
            "true_positives": tp,
            "false_negatives": bucket["fn"],
            "false_positives": bucket["fp"],
            "recall": tp / positive if positive else None,
            "false_positive_rate": bucket["fp"] / negative if negative else None,
            "pixel_error_mean": sum(errors_for_bucket) / len(errors_for_bucket) if errors_for_bucket else None,
            "pixel_error_median": errors_for_bucket[len(errors_for_bucket) // 2] if errors_for_bucket else None,
            "pixel_error_p90": errors_for_bucket[max(0, math.ceil(len(errors_for_bucket) * 0.9) - 1)] if errors_for_bucket else None,
            "matched_frames": len(errors_for_bucket),
        }

    return {
        "match_radius_pixels": match_radius,
        "overall": finalize({**counts, "errors": errors}),
    }


def evaluate(model_path: Path, manifest_path: Path, output_path: Path, imgsz: int, confidence: float, match_radius: float, class_id: int | None, class_name: str | None, device: str | None) -> None:
    rows = load_rows(manifest_path)
    model = YOLO(str(model_path))
    resolved_class_id = resolve_class_id(model, class_id, class_name)
    predictions = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    details_path = output_path.with_name(output_path.stem + "_frames.csv")
    with details_path.open("w", newline="", encoding="utf-8") as details_file:
        writer = csv.DictWriter(details_file, fieldnames=["image", "label", "confidence", "pred_x", "pred_y", "pixel_error"])
        writer.writeheader()
        for row in rows:
            result = model.predict(row["image"], imgsz=imgsz, conf=confidence, device=device, verbose=False)[0]
            prediction = prediction_center(result, resolved_class_id)
            predictions.append(prediction)
            error = None
            if prediction is not None and row["label"] == "positive":
                error = math.hypot(prediction["x"] - float(row["center_x"]), prediction["y"] - float(row["center_y"]))
            writer.writerow({
                "image": row["image"],
                "label": row["label"],
                "confidence": prediction["confidence"] if prediction else None,
                "pred_x": prediction["x"] if prediction else None,
                "pred_y": prediction["y"] if prediction else None,
                "pixel_error": error,
            })

    summary = metric_summary(rows, predictions, match_radius)
    summary.update({
        "model": str(model_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "imgsz": imgsz,
        "confidence": confidence,
        "class_id": resolved_class_id,
    })
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {details_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match_radius", type=float, default=50.0)
    parser.add_argument("--class_id", type=int)
    parser.add_argument("--class_name")
    parser.add_argument("--device")
    args = parser.parse_args()
    evaluate(args.weights, args.manifest, args.output, args.imgsz, args.confidence, args.match_radius, args.class_id, args.class_name, args.device)
