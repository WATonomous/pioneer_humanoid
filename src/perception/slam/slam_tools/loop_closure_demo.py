#!/usr/bin/env python3
"""See loop closure work (or fail) on a synthetic world, no camera or GPU needed.

A robot walks a building twice plus two side hallways that are IDENTICAL copies of each other. Odometry drifts.
Three detectors run on the same data, each followed by the same pose graph optimization:

  A  prototype      what vo.py does today: old keyframes within 1 m of the (drifted) position, first one to verify wins
  B  appearance     place recognition + verification only. This is "just plug in DBoW2": no false-loop defenses
  C  full module    slam_core.loop_closure: retrieval, verification, odometry gate, confirmation, consistency

The plot shows the true path (grey), raw odometry (red), the optimized result (blue), accepted true loops (green)
and accepted FALSE loops (thick magenta). The look-alike hallways are labelled H1 and H2.

    pip install numpy scipy matplotlib
    python slam_tools/loop_closure_demo.py                    # moderate drift, writes loop_closure_demo.png
    python slam_tools/loop_closure_demo.py --drift heavy --show
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "slam_core"))

try:
    import numpy as np  # noqa: E402
    import scipy  # noqa: E402,F401
    import matplotlib  # noqa: E402,F401
except ImportError as e:
    sys.exit(f"Missing package ({e.name}). Install the demo's packages with the same Python you ran this with:\n"
             f"    {sys.executable} -m pip install numpy scipy matplotlib")

from slam_core.loop_closure import LoopClosureDetector, LoopParams, sim  # noqa: E402

DRIFT = {"moderate": dict(bias=0.0012, rot_sigma=0.01), "heavy": dict(bias=0.003, rot_sigma=0.02)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", choices=DRIFT, default="moderate", help="how fast odometry drifts")
    ap.add_argument("--out", default="loop_closure_demo.png")
    ap.add_argument("--show", action="store_true", help="open the plot window")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = sim.make_scenario(seed=args.seed, **DRIFT[args.drift])

    def verify():                         # a fresh simulated verifier per detector
        return sim.make_verifier(sc)

    p = dict(min_inliers=10)    # the simulator sees ~20-30 landmarks, real XFeat matches are far more numerous
    variants = [
        ("A  prototype (1 m radius)", sim.PrototypeDetector(verify(), min_inliers=10)),
        ("B  appearance only", LoopClosureDetector(verify(), LoopParams(gate=False, confirm_count=1, pcm=False, **p))),
        ("C  full module", LoopClosureDetector(verify(), LoopParams(**p))),
    ]

    results = []
    for name, det in variants:
        t = time.time()
        results.append((name, sim.run(sc, det)))
        print(f"{name:28s} {time.time() - t:4.0f}s")

    print(f"\n{len(sc.kfs)} keyframes, {args.drift} drift, raw odometry error (RMS) {results[0][1]['ate_raw']:.2f} m\n")
    print(f"{'detector':28s} {'true loops':>10s} {'false loops':>11s} {'long-range':>10s} {'error after (m)':>16s}")
    for name, r in results:
        print(f"{name:28s} {r['n_true']:10d} {r['n_false']:11d} {r['n_long']:10d} {r['ate_opt']:16.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4), sharex=True, sharey=True)
    for ax, (name, r) in zip(axes, results):
        ax.plot(sc.gt[:, 0], sc.gt[:, 1], color="0.75", lw=6, label="true path", solid_capstyle="round")
        ax.plot(r["x0"][:, 0], r["x0"][:, 1], color="tab:red", lw=1.2, label="raw odometry")
        ax.plot(r["opt"][:, 0], r["opt"][:, 1], color="tab:blue", lw=1.6, label="after loop closure")
        for L, ok in zip(r["loops"], r["true"]):
            a, b = r["opt"][L.old_id, :2], r["opt"][L.new_id, :2]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:green" if ok else "magenta", lw=0.7 if ok else 3, alpha=0.8 if ok else 1)
        ax.annotate("H1", (-6, 8.6), color="k", weight="bold")
        ax.annotate("H2", (29, 8.6), color="k", weight="bold")
        ax.set_title(f"{name}\n{r['n_true']} true / {r['n_false']} false loops, error {r['ate_raw']:.2f} -> {r['ate_opt']:.2f} m")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left")
    fig.suptitle("Loop closure on a synthetic building: H1 and H2 are identical hallways 36 m apart (magenta = accepted false loop)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nwrote {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
