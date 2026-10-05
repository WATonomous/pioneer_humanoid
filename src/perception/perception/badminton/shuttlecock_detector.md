# Shuttlecock detector tools

Starter scripts for building a YOLO26n shuttlecock detector (issue #123). They run in this order:

```
video clips -> extract_frames -> auto_label -> split_dataset -> train -> (export_trt -> benchmark_latency)
```

**Status: untested.** Treat these as reference code. Read each script, run it on a small sample, and rewrite whatever you need. See "Known issues" below before trusting any output.

## Setup

- Python packages: `opencv-python`, `torch`, `torchvision`, `ultralytics`, `pyyaml`, and `groundingdino-py` (for `auto_label.py` only).
- A CUDA GPU for labeling and training (the PC with the RTX 5070 Ti). The 5070 Ti is Blackwell, so it needs a recent CUDA (12.8 or newer) and a matching PyTorch build. Confirm Ultralytics sees the GPU before anything else.
- Default folder layout (all paths can be changed with flags):
  - `data/raw_videos/`: input clips
  - `data/frames/`: extracted frames
  - `data/labeled/`: labels from auto-labeling
  - `data/dataset/`: final train/val dataset

## The scripts

### 1. `extract_frames.py`

Reads every video in a folder (`.mp4 .mov .avi .mkv`, searched recursively) and saves JPEG frames at a target rate, skipping blurry frames.

```
python extract_frames.py --input_dir data/raw_videos --output_dir data/frames --fps 5 --blur_thresh 50
```

- `--fps`: frames kept per second of video. Neighboring frames are near-duplicates, so a low rate is fine.
- `--blur_thresh`: sharpness cutoff (variance of the Laplacian). Lower keeps more blurry frames. Look at what gets dropped; fast-shuttle frames are the hard cases you want to keep.
- Output files are named `<video>_f00000.jpg`, `<video>_f00001.jpg`, and so on.

### 2. `auto_label.py`

Runs Grounding DINO (zero-shot: you give it a text prompt, no example images) on each frame and writes YOLO-format label files, one `.txt` per frame. Frames with no detection get an empty label file, which is intentional (hard negatives).

```
python auto_label.py --frames_dir data/frames --output_dir data/labeled \
    --prompt "shuttlecock . birdie ." --box_thresh 0.35 --text_thresh 0.25 \
    --config_path GroundingDINO_SwinT_OGC.py --weights_path groundingdino_swint_ogc.pth
```

- Needs the `groundingdino` package **and** the SwinT config and weights, downloaded separately from the GroundingDINO repo. They are not in this repo.
- Writes `labels/*.txt` and `review_manifest.jsonl` (per-frame detection count and max confidence). Sort the manifest by confidence and check the low ones by eye.
- Output is single-class: `shuttlecock` is class 0.

### 3. `split_dataset.py`

Splits the labeled frames into train and validation sets, copies them into the folder layout Ultralytics expects, and writes `shuttlecock.yaml`.

```
python split_dataset.py --labeled_dir data/labeled --frames_dir data/frames --out_dir data/dataset --val_frac 0.15
```

- Shuffles with a fixed seed (42), so results are reproducible.
- Output: `images/{train,val}`, `labels/{train,val}`, and `shuttlecock.yaml`.

### 4. `train.py`

Fine-tunes a pretrained YOLO26n on the dataset, then prints mAP50 and mAP50-95 on the validation set.

```
python train.py --data data/dataset/shuttlecock.yaml --epochs 150 --imgsz 960 --batch 16 --model yolo26n.pt
```

- Starts from COCO-pretrained weights (downloaded automatically). Confirm the weights filename loads.
- Early stopping after 30 epochs without improvement; mosaic augmentation is turned off for the last 10 epochs.
- Results go to `runs/shuttlecock/yolo26n_v1/`. The weights you want are `weights/best.pt`.

### 5. `export_trt.py` (deferred)

Converts `best.pt` to a TensorRT FP16 engine. **Not needed for now:** run PyTorch weights on the PC first and add TensorRT only if latency requires it, or when the robot gets onboard compute.

```
python export_trt.py --weights runs/shuttlecock/yolo26n_v1/weights/best.pt --imgsz 960
```

- The engine only works on the GPU and TensorRT version it was built on. Rebuild it on the target device.
- Comments in the file still describe an older Jetson Xavier / TensorRT 8.5 plan and need updating.

### 6. `benchmark_latency.py` (deferred)

Times inference on an exported engine over random-noise input (mean, median, p95, p99, max).

```
python benchmark_latency.py --engine path/to/best.engine --imgsz 960 --n_iters 200
```

- Measures speed only, not accuracy. Re-check accuracy on the held-out eval set after exporting.
- Model time is not the number that matters in the end. Measure camera frame to published detection.

## Known issues

- `auto_label.py` guesses whether Grounding DINO returns normalized center-width-height boxes or pixel corners, and the guess depends on the installed version. Overlay boxes on a few frames and check them by eye before labeling everything. It also needs to be confirmed to install on the 5070 Ti.
- `split_dataset.py` splits randomly by frame. Frames from the same video are near-duplicates, so a frame-level split leaks them into validation and makes scores look better than they are. Split by source video instead (the video name is the filename prefix before `_f`).
- `train.py` defaults to `imgsz=640`. The shuttlecock is small, so use 960.
- `export_trt.py` and `benchmark_latency.py` are untested and out of date (see above).