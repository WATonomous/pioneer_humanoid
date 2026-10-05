import argparse
import time

import numpy as np
from ultralytics import YOLO


def export_trt(weights, imgsz, half, workspace_gb):
    model = YOLO(weights)

    export_path = model.export(
        format="engine",  # TensorRT
        imgsz=imgsz,
        half=half,  # FP16 -- realistic target on Xavier's Volta GPU
        workspace=workspace_gb,
        device=0,
    )

    print(f"TensorRT engine exported to: {export_path}")
    print(
        "Reminder: this engine is locked to this exact device + TensorRT version. "
        "Re-export if you move to different hardware later (e.g. an Orin)."
    )
    return export_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="FP16 -- recommended on Xavier",
    )
    parser.add_argument(
        "--workspace_gb",
        type=int,
        default=4,
        help="TensorRT builder workspace; Xavier has 32GB shared, keep this conservative",
    )
    args = parser.parse_args()

    export_trt(args.weights, args.imgsz, args.half, args.workspace_gb)
