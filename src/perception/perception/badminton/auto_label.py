import argparse
import json
from pathlib import Path

import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict


CLASS_NAME = "shuttlecock"
CLASS_ID = 0


def xyxy_to_yolo(box, img_w, img_h):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2 / img_w
    cy = (y0 + y1) / 2 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return cx, cy, w, h


def auto_label(
    frames_dir, output_dir, prompt, box_thresh, text_thresh, config_path, weights_path
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "WARNING: no GPU detected, this will be extremely slow. Confirm you're on a CUDA machine."
        )

    model = load_model(config_path, weights_path)
    model = model.to(device)

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    review_manifest = output_dir / "review_manifest.jsonl"
    labels_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    print(f"Labeling {len(frame_paths)} frames...")

    n_with_detection = 0

    with open(review_manifest, "w") as manifest:
        for i, frame_path in enumerate(frame_paths):
            image_source, image = load_image(str(frame_path))
            img_h, img_w = image_source.shape[:2]

            boxes, confidences, phrases = predict(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=box_thresh,
                text_threshold=text_thresh,
                device=device,
            )

            label_path = labels_dir / f"{frame_path.stem}.txt"
            lines = []
            for box, conf in zip(boxes, confidences):
                # groundingdino returns cxcywh normalized already in some versions --
                # this assumes xyxy pixel output; adjust if your installed version differs
                cx, cy, w, h = (
                    xyxy_to_yolo(box.tolist(), img_w, img_h)
                    if len(box) == 4 and box.max() > 1
                    else box.tolist()
                )
                lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_path.write_text("\n".join(lines))

            if lines:
                n_with_detection += 1

            manifest.write(
                json.dumps(
                    {
                        "frame": frame_path.name,
                        "n_detections": len(lines),
                        "max_confidence": float(confidences.max())
                        if len(confidences)
                        else 0.0,
                    }
                )
                + "\n"
            )

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(frame_paths)} done")

    print(
        f"\nDone. {n_with_detection}/{len(frame_paths)} frames got at least one detection."
    )
    print(
        f"Review manifest written to {review_manifest} -- sort by max_confidence to spot-check "
        f"low-confidence detections before trusting them as ground truth."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", default="data/frames")
    parser.add_argument("--output_dir", default="data/labeled")
    parser.add_argument("--prompt", default="shuttlecock . birdie .")
    parser.add_argument("--box_thresh", type=float, default=0.35)
    parser.add_argument("--text_thresh", type=float, default=0.25)
    parser.add_argument("--config_path", default="GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--weights_path", default="groundingdino_swint_ogc.pth")
    args = parser.parse_args()

    auto_label(
        args.frames_dir,
        args.output_dir,
        args.prompt,
        args.box_thresh,
        args.text_thresh,
        args.config_path,
        args.weights_path,
    )