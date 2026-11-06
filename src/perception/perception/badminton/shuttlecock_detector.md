# Shuttlecock detector tools

Starter scripts for building a YOLO26n shuttlecock detector (issue #123). The
manual-label training workflow runs in this order:

```
videos -> prepare_eval -> annotate_eval -> build_yolo_dataset -> train -> eval_metrics -> visualize_predictions
```

`augment_yolo_dataset.py` was used for `yolo26n_v1` but is no longer
recommended (see Ticket 4).

For Ticket 3, use the manual evaluation workflow below instead of
`auto_label.py`. Keep all footage, sampled images, manifests, and metric
outputs outside the repository.

**Status: untested.** Treat these as reference code. Read each script, run it on a small sample, and rewrite whatever you need. See "Known issues" below before trusting any output.

## Setup

- Python packages: `opencv-python`, `torch`, `torchvision`, `ultralytics`, `pyyaml`, and `groundingdino-py` (for `auto_label.py` only).
- A CUDA GPU for labeling and training (the PC with the RTX 5070 Ti). The 5070 Ti is Blackwell, so it needs a recent CUDA (12.8 or newer) and a matching PyTorch build. Confirm Ultralytics sees the GPU before anything else.
- Default folder layout (all paths can be changed with flags):
  - `data/raw_videos/`: input clips
  - `data/frames/`: extracted frames
  - `data/labeled/`: labels from auto-labeling
  - `data/dataset/`: final train/val dataset

The `data/` names above are historical examples only. A recommended external
layout is `/path/to/badminton_eval/` with `images/`, `manifest.jsonl`, and
`metrics/` underneath it.

### Ticket 3: manual eval set and baseline

The Ticket 3 tools are `prepare_eval.py`, `annotate_eval.py`, and
`eval_metrics.py`. They use center points as ground truth and never call the
automatic Grounding DINO labeler.

First sample one or two videos. This example creates roughly 240 frames from
the first two clips and writes them under `/tmp` only as an example; use a
durable external directory for real work:

```
python prepare_eval.py \
  --input_dir /home/matthewyang/datasets/badminton_eval/raw_videos \
  --output_dir /path/to/badminton_eval \
  --video "PAC 1.MP4" --video "PAC 2.MP4" \
  --every_seconds 0.5
```

Annotate the generated manifest interactively:

```
python annotate_eval.py /path/to/badminton_eval/manifest.jsonl
```

Click-drag a bounding box around the shuttle for a positive frame. The tool
derives and stores the box center for pixel-error metrics. Press `n` for a
frame with no shuttle. The annotation is saved as JSONL with pixel coordinates
in the original 1920x1080 image. A completely out-of-frame shuttle is
negative.

Run the baseline or a trained checkpoint:

```
python eval_metrics.py \
  --weights yolo26n.pt \
  --manifest /path/to/badminton_eval/manifest.jsonl \
  --output /path/to/badminton_eval/metrics/stock_yolo26n.json \
  --imgsz 960 --confidence 0.25 --match_radius 50
```

The script writes a JSON summary and a per-frame CSV. It reports recall,
false-positive rate on negative frames, precision, "mislocalized" frames (a
detection on a positive frame that is more than `--match_radius` away), and
mean/median/p90 pixel error for matched positive detections, overall and per
video. `--sweep 0.05,0.1,0.15,0.2,0.3` reports the same metrics at several
confidence thresholds from one inference pass. `--image_root DIR` finds images
by filename when the manifest's absolute paths came from another machine.

To score a YOLO split instead of a manifest (for example, to tune the
threshold on PAC 5 without touching the held-out set), pass
`--yolo_dir <dataset> --split val` instead of `--manifest`. For a fine-tuned
single-class checkpoint, class 0 is inferred automatically. For stock
multi-class weights, it selects `shuttlecock`, `birdie`, or `bird`; pass
`--class_id` or `--class_name` when that is not the desired class.

For the downloaded stock `yolo26n.pt` COCO weights, an empirical scan of the
hand-labeled frames found that class 32 (`sports ball`) bounds the shuttlecock
far better than class 14 (`bird`). Use `--class_id 32` for that stock baseline.

### Ticket 4: train from manual boxes

Keep the Ticket 3 videos completely out of training. In the current external
dataset, use `PAC 1.MP4` and `PAC 6.MP4` for held-out evaluation, and use
`PAC 2.MP4` through `PAC 5.MP4` for training. Reserve one training video for
validation so no source video appears in both splits.

Create and annotate training frames:

```
python prepare_eval.py \
  --input_dir /home/matthewyang/datasets/badminton_eval/raw_videos \
  --output_dir /home/matthewyang/datasets/badminton_train \
  --video "PAC 2.MP4" --video "PAC 3.MP4" \
  --video "PAC 4.MP4" --video "PAC 5.MP4" \
  --every_seconds 0.5

python annotate_eval.py \
  /home/matthewyang/datasets/badminton_train/manifest.jsonl \
  --max_width 1000 --max_height 650
```

Build the YOLO directory and split by source video. This example uses `PAC
5.MP4` for validation and `PAC 2` through `PAC 4` for training:

```
python build_yolo_dataset.py \
  --manifest /home/matthewyang/datasets/badminton_train/manifest.jsonl \
  --output_dir /home/matthewyang/datasets/badminton_train/dataset \
  --val_video "PAC 5.MP4" --clean
```

The converter derives YOLO class-0 boxes from the hand-labeled pixel boxes.
Negative frames receive empty label files. It also writes a split manifest and
fails if a requested validation video is not present.

`yolo26n_v1` was trained after `augment_yolo_dataset.py --copies 1` (offline
flips and color jitter). Ultralytics already applies both online, so the copies
add no information and double the epoch time. Remove them before training new
models:

```
python augment_yolo_dataset.py \
  --dataset_dir /home/matthewyang/datasets/badminton_train/dataset --remove
```

Never augment the validation split or the Ticket 3 evaluation images.

Train, keeping the run outside the repository. `train.py` defaults are tuned
for this dataset: `imgsz 1280` (shuttles are a median ~17 px wide at 1920x1080,
so 960 shrinks them to ~8 px), `scale 0.3` (less random zoom-out of tiny
objects), `patience 50` (the PAC 5 val split is small and noisy), batch 8 (fits
an 8 GB GPU at 1280):

```
python train.py \
  --data /home/matthewyang/datasets/badminton_train/dataset/shuttlecock.yaml \
  --project /home/matthewyang/datasets/badminton_train/runs \
  --name yolo26n_v2 --cache ram
```

`yolo26n_v1` was trained with `--epochs 150 --imgsz 960 --batch 16
--patience 30 --close_mosaic 10 --scale 0.5` to reproduce it.

Pick the confidence threshold on the PAC 5 val split, not on the Ticket 3 set:

```
python eval_metrics.py \
  --weights .../runs/yolo26n_v2/weights/best.pt \
  --yolo_dir /home/matthewyang/datasets/badminton_train/dataset --split val \
  --output .../metrics/yolo26n_v2_val.json \
  --imgsz 1280 --class_id 0 --sweep 0.05,0.1,0.15,0.2,0.3,0.4
```

Then evaluate once on the untouched Ticket 3 set at that threshold and the
same `imgsz` the model was trained at:

```
python eval_metrics.py \
  --weights .../runs/yolo26n_v2/weights/best.pt \
  --manifest /home/matthewyang/datasets/badminton_eval/manifest.jsonl \
  --output /home/matthewyang/datasets/badminton_eval/metrics/yolo26n_v2.json \
  --imgsz 1280 --confidence 0.25 --match_radius 50 --class_id 0
```

Optional final model: once settings are fixed, rebuild with
`build_yolo_dataset.py ... --val_video "PAC 5.MP4" --train_on_val` so PAC 5 is
also trained on, train with `--patience 0 --epochs <best epoch of v2 + 15>`,
and use `last.pt` (val metrics are meaningless once val is in train).

Use the same Ticket 3 manifest and threshold when comparing against the stock
class-32 baseline. Promote the final weights to team-shared storage only after
recording the source-video split, training command, commit, and metrics.

#### Results so far (Ticket 3 set: PAC 1 + PAC 6, 149 positive / 48 negative frames)

| Model | imgsz | conf | Recall | FP rate | Notes |
| --- | --- | --- | --- | --- | --- |
| stock yolo26n, class 32 | 960 | 0.25 | 7.4% | 4.2% | COCO `sports ball` |
| stock yolo26n, class 14 | 960 | 0.25 | 0.0% | 0.0% | COCO `bird` |
| yolo26n_v1 | 960 | 0.25 | 89.9% | 0.0% | median center error 0.66 px |
| yolo26n_v1 | 1280 | 0.25 | 94.0% | 2.1% | same weights, larger inference size |

Most remaining misses are shuttles held in a hand, lying on the floor at the
bottom edge of the frame, or about 10 px wide. On PAC 5, most false positives
are light glare on the floor.

### Visualizing predictions and real-time factor

`visualize_predictions.py` runs a model on every frame of the held-out videos
and writes an annotated MP4 plus a per-frame CSV. Frames that appear in the
manifest also show the hand-labeled box and a TP/FN/FP/TN badge:

```
python visualize_predictions.py \
  --weights .../weights/best.pt \
  --video ".../raw_videos/PAC 1.MP4" --video ".../raw_videos/PAC 6.MP4" \
  --manifest .../badminton_eval/manifest.jsonl \
  --output_dir .../badminton_eval/videos/yolo26n_v1 --imgsz 960
```

Without `--video`, it renders only the labeled frames as a 2 FPS slideshow per
video; add `--only_errors` to keep just the misses and false alarms.

The script also prints the real-time factor (RTF = processing time / video
duration; below 1 keeps up with the camera) and writes `timing.json`. Measured
on an RTX 4060 with yolo26n_v1, 60 FPS 1080p footage, one frame at a time in
PyTorch:

| imgsz | Model ms/frame | Model RTF | RTF incl. video decode |
| --- | --- | --- | --- |
| 960 | 14.2 (inference 9.9) | 0.85 | 1.10 |
| 1280 | 16.1 (inference 9.9) | 0.97 | 1.21 |
| 1280 FP16 | 16.7 | 1.00 | 1.25 |

GPU time is the same at both sizes, so the nano model is limited by per-call
overhead rather than compute. A 30 FPS camera halves the RTF. TensorRT
(`export_trt.py`) and decoding on a separate thread are the next steps if it
must keep up with 60 FPS.

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

Legacy helper for auto-generated `.txt` labels. It splits randomly by frame and
must not be used for the manual-label Ticket 4 workflow. Use
`build_yolo_dataset.py`, which splits by source video, instead.

```
python split_dataset.py --labeled_dir data/labeled --frames_dir data/frames --out_dir data/dataset --val_frac 0.15
```

- Shuffles with a fixed seed (42), so results are reproducible.
- Output: `images/{train,val}`, `labels/{train,val}`, and `shuttlecock.yaml`.

### 4. `train.py`

Fine-tunes a pretrained YOLO26n on the dataset, then prints mAP50 and mAP50-95 on the validation set.

```
python train.py --data data/dataset/shuttlecock.yaml --project <external runs dir> --name yolo26n_v2
```

- Starts from COCO-pretrained weights (downloaded automatically).
- Defaults: 200 epochs, imgsz 1280, batch 8, early stopping after 50 epochs without improvement, mosaic off for the last 15 epochs, `scale 0.3`. See Ticket 4 for the reasoning.
- The weights you want are `<project>/<name>/weights/best.pt`.

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
- The annotation tool stores one box per frame. If a second shuttle is visible (for example on another court), it is unlabeled and trains as background.
- `export_trt.py` and `benchmark_latency.py` are untested and out of date (see above).