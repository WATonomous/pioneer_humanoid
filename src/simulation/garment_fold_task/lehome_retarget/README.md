# LeHome retarget

Replays LeHome fold demos on the 8-DOF arm and records the folds that succeed.

```bash
KIND=pant_short OUTNAME=pant_short_wc EXTRA="--workcell --shift_y 0.05 --hold_s 10" ./run_all.sh
```

## Post-fold creep

| Problem | Solution |
|---|---|
| The folded cloth kept sliding 2-6 mm/s after release (LeHome: ~0.8). The arm laid the patch down at 8 mm, pressing it into the layer below. | Lay it down higher and slower: `LAY_Z, LAY_FRAMES = 0.015, 16` (12 and 20 mm were worse). |
| The first reset after startup lands the cloth up to 9 cm off. | A warm-up reset before the first episode. |
| Some folds still slide (outcomes are chaotic run to run). | `SLIDE_MAX`: reject folds sliding over 1 mm/s 8-10 s after the fold (needs `--hold_s 10`). |

Result: accepted folds settle at 0.94 mm/s vs LeHome's 0.82 (15-20 s after the fold).
Physics, scene and arm were ruled out by loading one sim's end-of-fold cloth into the other.
The hold is not recorded; episodes end when the arm withdraws, like LeHome's.
