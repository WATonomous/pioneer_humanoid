# LeHome retarget

Replays LeHome fold demos on the 8-DOF arm and records the folds that succeed.

```bash
KIND=pant_short OUTNAME=pant_short_wc EXTRA="--workcell --shift_y 0.05 --hold_s 10" ./run_all.sh
```

## Post-fold creep

Cause: the arm laid the folded patch down at 8 mm, pressing it into the layer below, and the fold kept sliding (2-6 mm/s).
Physics, scene and arm were ruled out by loading one sim's end-of-fold cloth into the other.

Fix:
- `LAY_Z, LAY_FRAMES = 0.015, 16`: lay it down higher and slower (12 and 20 mm were worse).
- A warm-up reset before the first episode: the first reset after startup lands the cloth up to 9 cm off.
- `SLIDE_MAX`: reject folds still sliding over 1 mm/s 8-10 s after the fold (needs `--hold_s 10`).

Accepted folds now settle at 0.94 mm/s vs LeHome's 0.82 (15-20 s after the fold).
The hold is not recorded; episodes end when the arm withdraws, like LeHome's.
