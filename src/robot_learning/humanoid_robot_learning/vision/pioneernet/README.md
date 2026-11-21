# PioneerNet v13

An RGB-D visual encoder for robot policies: a DINOv3 ViT-B/16 (86M parameters) distilled from DINOv2-S (colour) and
DeFM-S (depth), with real-photo backgrounds composited in during distillation. It is the encoder we benchmarked for
Pioneer-1. Use it **frozen**.

## Results (simulation, 8-DoF bimanual peg insertion)

Every encoder frozen under the same policy (transformer BC, d_model 512, 30k steps), 3 policy seeds each, success rate
in %, 100 episodes clean, 50 per shift. Unseen environments are scenes no encoder or policy was trained in.

| | Clean | Lighting | Colour | Shadows | Clutter | Lens | Background | **Unseen environments (4)** |
|---|---|---|---|---|---|---|---|---|
| **PioneerNet v13** | **87** | **83** | 68 | **75** | **72** | 39 | 30 | **61** |
| PioneerNet-B (v10) | 85 | 66 | **74** | 13 | 54 | **65** | **44** | 9 |
| PioneerNet v5 (12M ResNet-18) | 81 | 61 | 66 | 13 | 35 | 60 | 19 | 0 |
| DINOv3 ViT-B/16 (Pioneer-1's default) | 77 | 0 | 1 | 2 | 4 | 9 | 1 | 0 |

On fresh validation layouts v13 gets 60% in the unseen environments. Its weak spots are a wet lens, a new background
and recoloured objects. Not yet tested: other tasks, other policy heads (e.g. Pioneer-1's flow matching), the real robot.

## Use

```python
from humanoid_robot_learning.vision.pioneernet import load_pioneernet

enc = load_pioneernet("pioneernet_v13.pt", device="cuda")   # or "hf:<repo_id>/pioneernet_v13.pt"; frozen, eval mode
cls, grid = enc.encode(rgb, depth)       # rgb (N, 3, H, W) in [0, 1]; depth (N, H, W) metres, 0 = no reading
# cls (N, 768); grid (N, 768, 16, 20)
tokens = enc.tokens(enc.preprocess(rgb, depth))   # (N, 5 + 320, 768): class + 4 registers + 16x20 patches
cls, grid = enc.encode(rgb, None)        # camera without depth: the depth input is blanked
```

- Input: any size; it is resized to 256x320 (4:3 sensors map onto it without distortion). Depth must be **metric**
  (float metres). The encoder converts it to DeFM's log encoding itself.
- Output layout matches a Hugging Face DINOv3 ViT's `last_hidden_state`: 1 class + 4 register tokens, then patches.
  The patch grid is **16x20, not square**.
- `depth=None` blanks the depth input (it saw that in 10% of training samples). Robustness to lighting and colour comes
  largely from depth, and RGB-only use is **not benchmarked**.
- Requirements: `torch`, `timm` (>= 1.0, for `vit_base_patch16_dinov3.lvd1689m`), `huggingface_hub` for `hf:` paths.
  The DINOv3 architecture is built locally: no gated download is needed to load a checkpoint.
- The checkpoint (`pioneernet_v13.pt`, 350 MB) is not in git. Its loader checks the state dict loads exactly; the outputs
  were checked bit-identical to the benchmark code that produced the results above.

## Notes for plugging it into a policy (e.g. Pioneer-1)

Pioneer-1 itself is unchanged on this branch. Things to handle when wiring it in, found while trying it:

1. **Frozen weights.** Pioneer-1 fine-tunes its vision encoder at `vision_lr_scale` 0.1; v13 should stay frozen (and in
   eval mode), with its parameters left out of the optimizer.
2. **Non-square grid.** Pioneer-1 pools the wrist cameras' patch grid with `math.isqrt`, which assumes a square grid;
   v13's is 16x20, so pool with the real (h, w).
3. **Aspect ratio.** Pioneer-1 resizes every camera to a square (224x224). Give v13 256x320 (or the raw 4:3 frame) and
   crop augmentations with ratio 320/256.
4. **Depth.** Pioneer-1's datasets have no metric depth today: `sim_recorder.py` saves depth only as an 8-bit preview
   video. Recording float depth per camera is needed for the RGB-D mode. When depth is augmented, apply the same crop
   and rotation as the RGB (nearest interpolation, 0 = no reading), resize back to the frame size, no colour jitter.
5. Normalisation is done inside `preprocess`: feed RGB in [0, 1], not ImageNet-normalised pixels.

## How it was trained (`training/`)

`training/` is a verbatim snapshot of the benchmark code that produced v13 (research code, not a library). Two
changes: the repo root and the data directory are found from the package location, and outputs go to
`$PIONEERNET_WORK` (default `~/pioneernet_work`, outside the repo).

Recipe: the student sees a corrupted RGB-D frame, the frozen teachers the clean one. Student = DINOv3 ViT-B/16 plus a
zero-initialised depth patch embedding (so step 0 is exactly DINOv3 on RGB). Losses: match DINOv2-S patch and class
features from the colour path, DeFM-S features from the depth path, and a small geometry head (peg / hole positions).
Corruptions: photometric jitter (no hue mixing, so object colours keep their identity), depth sensor noise, camera
jitter, colour blanked in 30% / depth in 10% of samples (depth patch embedding at 10x the learning rate), and **real
photos composited into the background**: in 60% of samples the background pixels become a random real RGB-D frame
(TUM, NYU, SUN RGB-D), in 50% the table gets a real photo texture. That last step is what makes it work in new scenes
(B2a, the same recipe without it: 1% in the unseen environments; v13: 61%).

Data and code it was run on:
- Simulator: this repo at commit `ca66cb70` (later commits changed `TeleopSim` and the gripper). Check it out as a git
  worktree and set `BENCH_SRC=<worktree>/src`.
- Robot: `BENCH_ROBOT=8dof` (both arms of `pioneer_bimanual_arm_8dof`), scene `peg_insert`.
- Real backgrounds: 15 TUM RGB-D sequences (freiburg1 360 / desk / desk2 / plant / room / teddy, freiburg2 desk /
  pioneer_360, freiburg3 cabinet / large_cabinet / long_office_household / nostructure_texture_near_withloop /
  structure_texture_far / structure_texture_near / teddy), 12 NYU Depth v2 shards (`sayakpaul/nyu_depth_v2`, train
  000000-000011, every 3rd frame) and SUN RGB-D (`SUNRGBD.zip`), each converted to 320x240 npz with `training/real_rgbd/`.
- Teachers: `timm` DINOv2 (`vit_small_patch14_dinov2.lvd142m`) and DeFM (`pip install defm`, `defm_vit_s14`).

```bash
cd src/robot_learning/humanoid_robot_learning/vision/pioneernet/training
export BENCH_ROBOT=8dof BENCH_SRC=<worktree at ca66cb70>/src PIONEERNET_WORK=~/pioneernet_work
R=$PIONEERNET_WORK/real_rgbd
# 1. simulated demos (160 layouts), the same layouts with camera jitter, and per-pixel background / table masks
python gen_data.py --split train --seeds 0:160 --workers 10
python gen_data.py --split train_camjit --seeds 0:160 --variants camjit --workers 10
python gen_data.py --split train --seeds 0:160 --masks
python gen_data.py --split train_camjit --seeds 0:160 --masks
# 2. real RGB-D frames for the backgrounds
python -I real_rgbd/convert_tum.py <TUM sequences dir> $R/tum_320x240.npz --every 2
curl -sL <NYU shard url> | python -I real_rgbd/convert_nyu.py $R/nyu/train-000000.npz --every 3   # x12 shards
python -I real_rgbd/convert_sun.py SUNRGBD.zip $R/sun_320x240.npz
# 3. distil (~3.3 h on an RTX 5070 Ti with a 16 GB GPU, ~50 GB RAM; checkpoints to resume every 5000 steps)
python distill.py --rgb_teacher dinov2_s14 --depth_teacher defm_s14 --backbone dinov3_vitb16 --steps 20000 --bs 64 \
  --lr 1e-4 --moddrop_rgb 0.3 --moddrop_depth 0.1 --depth_lr_mult 10 --hw 256 320 --geo 1.0 --cam_aug train_camjit \
  --hue_mix 0 --scene_keep_clean --bg_real "$R/tum_320x240.npz,$R/nyu/train-*.npz,$R/sun_320x240.npz" --bg_n 8000 \
  --bg_p 0.6 --table_p 0.5 --out $PIONEERNET_WORK/results_8dof/pioneernet_v13.pt
```

Ablations worth knowing (same policy, 1 seed unless noted): PioneerNet-B's large teachers with the same additions (v14)
were worse on lighting (52) and unseen environments (41); adding synthetic box/panel scene corruption on top of the
real backgrounds (v15) was worse almost everywhere (unseen environments 27).
