"""Chance level for the probe: predict the training-set mean of every target (and the most common phase) on the test
set, scored with probe.metrics. Written to <results>/chance.json; the report's bars are scaled to it.

    .venv/bin/python chance.py      # BENCH_ROBOT=8dof (default) or armv2
"""
from __future__ import annotations

import json

import numpy as np

import common as C
import dataset as D
import probe as P


def main():
    tr = D.load_split("train", rgb=False, depth=False)
    te = D.load_split("test", rgb=False, depth=False)
    ytr, yte = P.stack_targets(tr), P.stack_targets(te)
    pred = np.repeat(ytr.mean(0, keepdims=True), len(yte), 0)
    hist = np.bincount(tr["phase"], minlength=5)
    logits = np.zeros((len(yte), 5))
    logits[:, hist.argmax()] = 1   # always the most common phase
    out = P.metrics(pred, logits, yte, te["phase"])
    out = {k: v for k, v in out.items() if not k.startswith("peg_hole_")}
    out["phase_acc"] = float(np.mean(te["phase"] == hist.argmax()))
    out["phase_hist_train"] = hist.tolist()
    json.dump(out, open(C.OUT / "chance.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
