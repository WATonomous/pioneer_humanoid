"""Retargeted LAFAN1 CSVs -> SuperMimic motion npz for Pioneer: 5 Hz zero-phase smoothing (per-frame IK jitter), played
SLOW times slower (Pioneer's AKH70-48 hip/knee motors do 3.67 rad/s, human motion needs ~7), resampled to 50 Hz.
Clips with flight phases (jumps, runs, sprints) are left out when slowed: that gives physically impossible air time.

  SM_ROBOT=pioneer PIONEER_XML=<train.xml> <supermimic venv>/python convert_for_supermimic.py <csv dir> <npz out dir> <slow>

slow = 1 plays the motion at recorded speed (the 77-clip run); 2 halves it.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from supermimic.data.motion_convert import convert_csv_to_npz, load_g1_model

FLIGHT = ("jumps", "run", "sprint")
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
slow = float(sys.argv[3])
model = load_g1_model()  # Pioneer under SM_ROBOT=pioneer
b, a = butter(2, 5.0 / 15.0)
total = 0.0
for csv in sorted(src.glob("*.csv")):
    if csv.stem.startswith(FLIGHT) and slow != 1.0:
        continue
    q = np.loadtxt(csv, delimiter=",")
    for i in range(1, len(q)):  # one quaternion hemisphere, so the filter does not cross a sign flip
        if np.dot(q[i, 3:7], q[i - 1, 3:7]) < 0:
            q[i, 3:7] *= -1
    q = filtfilt(b, a, q, axis=0)
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        np.savetxt(f, q, delimiter=",", fmt="%.6f")
    info = convert_csv_to_npz(f.name, dst / f"{csv.stem}.npz", input_fps=30.0 / slow, output_fps=50.0, model=model)
    os.unlink(f.name)
    total += info["duration_s"]
    print(f"{csv.stem}: {info['duration_s']:.0f} s")
print(f"{total / 3600:.2f} h of motion in {dst}")
