import argparse
import time

import numpy as np
from ultralytics import YOLO


def benchmark(engine_path, imgsz, n_iters, n_warmup):
    model = YOLO(engine_path, task="detect")

    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f"Warming up ({n_warmup} iters)...")
    for _ in range(n_warmup):
        model.predict(dummy_frame, verbose=False)

    print(f"Benchmarking ({n_iters} iters)...")
    latencies_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model.predict(dummy_frame, verbose=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    latencies_ms = np.array(latencies_ms)
    print(f"\nEnd-to-end latency (predict() call, includes pre/post-processing):")
    print(f"  mean:   {latencies_ms.mean():.2f} ms")
    print(f"  median: {np.median(latencies_ms):.2f} ms")
    print(f"  p95:    {np.percentile(latencies_ms, 95):.2f} ms")
    print(f"  p99:    {np.percentile(latencies_ms, 99):.2f} ms")
    print(f"  max:    {latencies_ms.max():.2f} ms")
    print(f"  implied max FPS (mean-based): {1000 / latencies_ms.mean():.1f}")
    print(
        f"\nNote: this uses synthetic (random noise) input for pure latency timing -- "
        f"it does NOT measure accuracy. Run separately against real held-out frames "
        f"to check mAP didn't regress from the .pt -> TensorRT export/quantization step."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to exported .engine file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--n_warmup", type=int, default=20)
    args = parser.parse_args()

    benchmark(args.engine, args.imgsz, args.n_iters, args.n_warmup)
