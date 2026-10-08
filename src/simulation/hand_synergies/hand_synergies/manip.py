"""Manipulation synergies: PCA of the joint targets MPC commands while spinning the cube, compared with
the grasp synergies.

    python -m hand_synergies.mpc --space joint --seconds 15 --log out/mpc_joint_log0.npz
    python -m hand_synergies.manip --logs out/mpc_joint_log0.npz [...] --out out

Writes out/manip_synergies.npz (same keys as synergies.npz, so ``mpc --synergies`` can plan in it) and
prints how much of the manipulation motion each k-dim grasp subspace captures against a random k-dim
subspace (k/20), plus the principal angles between the two top-3 subspaces.

Caveat: the logged targets include the planner's own sampling noise, which is isotropic in joint space.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .synergies import fit_pca


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--max-seconds", type=float, nargs="*", default=None,
                   help="per log: use only the first N s (e.g. up to a drop)")
    p.add_argument("--out", default="out")
    args = p.parse_args()
    out = Path(args.out)

    runs = []
    for i, path in enumerate(args.logs):
        log = np.load(path)
        ctrl = log["ctrl"]
        if args.max_seconds:
            ctrl = ctrl[: int(args.max_seconds[i] / float(log["control_dt"]))]
        runs.append(ctrl)
    x = np.vstack(runs)
    grasp = dict(np.load(out / "synergies.npz"))
    manip = fit_pca(x)
    np.savez(out / "manip_synergies.npz", joint_names=grasp["joint_names"], **manip)

    xc = x - x.mean(axis=0)
    total = (xc**2).sum()
    cum = np.cumsum(manip["variance_ratio"])
    print(f"{len(x)} control steps; manipulation PCA cumulative: "
          + ", ".join(f"{k}:{cum[k - 1]:.0%}" for k in (1, 2, 3, 5, 10)))
    for k in (1, 2, 3, 5, 10):
        captured = ((xc @ grasp["components"][:k].T) ** 2).sum() / total
        print(f"  grasp PCs 1-{k:<2d} capture {captured:4.0%} of the manipulation motion (random {k}-D: {k / 20:.0%})")
    s = np.linalg.svd(manip["components"][:3] @ grasp["components"][:3].T, compute_uv=False)
    print("principal angles, top-3 manipulation vs top-3 grasp subspace (deg):",
          np.degrees(np.arccos(np.clip(s, -1, 1))).round(0))


if __name__ == "__main__":
    main()
