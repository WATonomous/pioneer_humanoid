"""Do synergies help find grasps on new objects? Random pre-shape search, same budget, three samplers.

For each test problem (a new random object and pose under the palm) every sampler draws pre-shapes and
runs the usual power autograsp + shake test from each:

  prior       the generator's own random pre-shape (grasp_gen.sample_preshape)
  gauss-20    Gaussian over all 20 joints fitted to the stable grasps (every PC, at its own sd)
  synergy-k   Gaussian over the first k PCs only: q = mean + sum_i a_i PC_i, a_i ~ N(0, sd_i)

The fitted samplers draw grasp postures, not pre-shapes: their flexion is scaled down by OPEN_SCALE to the
prior's openness (keeping the synergy's pattern), then opened further if the hand still overlaps the object. Reports per-attempt success and the fraction of problems solved within the budget.

    python -m hand_synergies.search --problems 150 --budget 16 --out out
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

from .synergies import BLUE, INK, MUTED, OPEN_STEPS, RED, SURFACE, _style

_STATE = {}
# Grasp postures are ~41% curled, the generator's pre-shapes ~13%: scale flexion by this to match.
OPEN_SCALE = 0.3
PROBLEM_SEED = 900_000_000  # far from grasp_gen's trial seeds: test objects are new


def _gen():
    from .grasp_gen import GraspGen

    return _STATE.get("gen") or _STATE.setdefault("gen", GraspGen())


def _sample(sampler: str, rng: np.random.Generator, pca: dict, gen) -> np.ndarray:
    if sampler == "prior":
        return gen.sample_preshape(rng)
    k = 20 if sampler == "gauss-20" else int(sampler.split("-")[1])
    a = rng.normal(size=k) * pca["stds"][:k]
    q = pca["mean"] + a @ pca["components"][:k]
    return np.clip(q, *gen.ranges.T)


def _clear_preshape(gen, q: np.ndarray, kind, size, pos, quat) -> np.ndarray | None:
    """Turn a grasp posture into a pre-shape: scale its flexion down by OPEN_SCALE (keeping the pattern),
    then open further by OPEN_STEPS until the hand clears the object; None if it never does."""
    from .scene import set_object

    m, d, idx = gen.model, gen.data, gen.idx
    q = q.copy()
    q[gen.close_j] *= OPEN_SCALE
    for delta in (0.0, *OPEN_STEPS):
        q0 = q.copy()
        q0[gen.close_j] -= gen.close_sign * delta
        q0 = np.clip(q0, *gen.ranges.T)
        set_object(m, idx, kind, size)
        mujoco.mj_resetData(m, d)
        d.qpos[idx.qpos] = q0
        d.qpos[idx.obj_qpos:idx.obj_qpos + 3] = pos
        d.qpos[idx.obj_qpos + 3:idx.obj_qpos + 7] = quat
        mujoco.mj_forward(m, d)
        if not any(c.dist < 0 for c in d.contact[: d.ncon]):
            return q0
    return None


def _attempt(args):
    problem, sampler, attempt, pca = args
    from .grasp_gen import GRASP_TYPES

    gen = _gen()
    kind, size, pos, quat = gen.sample_object(np.random.default_rng(PROBLEM_SEED + problem))
    rng = np.random.default_rng([problem, attempt, hash(sampler) % 2**31])
    q = _sample(sampler, rng, pca, gen)
    if sampler != "prior":
        q = _clear_preshape(gen, q, kind, size, pos, quat)
        if q is None:
            return problem, sampler, False
    ok = gen.grasp(q, kind, size, pos, quat, GRASP_TYPES["power"][0]) is not None
    return problem, sampler, ok


def plot(results: dict, budget: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    samplers = list(results)
    colors = {"prior": MUTED, "gauss-20": RED}
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    n = np.arange(1, budget + 1)
    for s in samplers:
        hits = results[s]  # (problems, budget) bool
        solved = np.cumsum(hits, axis=1) > 0
        curve = solved.mean(axis=0) * 100
        ax.plot(n, curve, linewidth=2, color=colors.get(s, BLUE),
                linestyle="-" if s.startswith("synergy") or s in colors else "--",
                alpha=1.0 if s in ("prior", "gauss-20", "synergy-3") else 0.55,
                label=f"{s}  ({hits.mean() * 100:.0f}% per attempt)")
        ax.annotate(f"{curve[-1]:.0f}%", (budget, curve[-1]), textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=INK)
    ax.set_xlim(1, budget + 1.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("pre-shapes tried", color=INK)
    ax.set_ylabel("% of new objects grasped", color=INK)
    ax.set_title(f"Random pre-shape search on {len(next(iter(results.values())))} new objects (power grasp)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--problems", type=int, default=150)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--samplers", default="prior,gauss-20,synergy-1,synergy-2,synergy-3,synergy-5")
    p.add_argument("--out", default="out")
    args = p.parse_args()
    out = Path(args.out)
    pca = dict(np.load(out / "synergies.npz"))
    samplers = args.samplers.split(",")
    jobs = [(pr, s, a, pca) for pr in range(args.problems) for s in samplers for a in range(args.budget)]
    results = {s: np.zeros((args.problems, args.budget), bool) for s in samplers}
    counters = {(pr, s): 0 for pr in range(args.problems) for s in samplers}
    with mp.Pool() as pool:
        for pr, s, ok in pool.imap_unordered(_attempt, jobs, chunksize=8):
            results[s][pr, counters[pr, s]] = ok
            counters[pr, s] += 1
    np.savez(out / "search.npz", **results)
    plot(results, args.budget, out / "search.png")
    for s, hits in results.items():
        print(f"{s:10s} per-attempt {hits.mean():5.1%}   solved within {args.budget}: {(hits.any(1)).mean():5.1%}")


if __name__ == "__main__":
    main()
