"""Fine-tune YOLO26 on the one-class shuttlecock dataset.

Defaults are tuned for a small (~15-20 px at 1920x1080) object and a small
dataset:
  - imgsz 1280 keeps more pixels on the shuttle (960 halves it to ~8 px).
  - scale 0.3 limits random zoom-out so shuttles are not shrunk to a few pixels.
  - patience 50 because the validation split is small and mAP is noisy.
Ultralytics already applies flips and HSV jitter online, so the offline
augment_yolo_dataset.py copies are not needed.
"""

import argparse

from ultralytics import YOLO


def train(args):
    model = YOLO(args.model)
    extra = {"device": args.device} if args.device else {}
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,  # 0 disables early stopping
        close_mosaic=args.close_mosaic,
        scale=args.scale,
        mosaic=args.mosaic,
        cache=args.cache or False,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
        exist_ok=True,
        **extra,
    )

    # Ultralytics re-validates best.pt at the end of training; reuse those numbers
    # instead of calling model.val(), which writes a stray runs/ folder in the cwd.
    metrics = model.metrics
    if metrics is not None and hasattr(metrics, "box"):
        print(f"\nbest.pt val mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}")
    print(f"Weights: {model.trainer.best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset/shuttlecock.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8, help="Lower this if you run out of GPU memory.")
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="yolo26n.pt for nano; try yolo26s.pt if accuracy is short and latency allows",
    )
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--close_mosaic", type=int, default=15)
    parser.add_argument("--scale", type=float, default=0.3, help="Random zoom range (+/-). Ultralytics default is 0.5.")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--cache", choices=["ram", "disk"], help="Cache decoded images to speed up epochs.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/shuttlecock")
    parser.add_argument("--name", default="yolo26n_v2")
    train(parser.parse_args())
