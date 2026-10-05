import argparse
from ultralytics import YOLO


def train(data_yaml, epochs, imgsz, batch, model_variant):
    model = YOLO(
        model_variant
    )

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        close_mosaic=10,
        patience=30,  # early stopping if val mAP plateaus
        project="runs/shuttlecock",
        name="yolo26n_v1",
        exist_ok=True,
    )

    metrics = model.val()
    print(
        f"\nFinal val mAP50: {metrics.box.map50:.4f}, mAP50-95: {metrics.box.map:.4f}"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset/shuttlecock.yaml")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="yolo26n.pt for nano; bump to yolo26s.pt if accuracy is short and latency budget allows",
    )
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.model)
