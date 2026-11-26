"""Open-loop replay: play each recorded demo's actions back through bc.EvalEnv (the exact action path every policy
uses, linear interpolation between 25 Hz actions) and count insertions. This is the success ceiling for BC, and the
end-to-end check that recorded actions, ArmIO and the eval env agree.

    .venv/bin/python replay_check.py --n 20        # BENCH_ROBOT=8dof (default) or armv2
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

import common as C
from bc import EvalEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    C.contract_check()
    import humanoid_mujoco_scenes.peg_insert.scene as S
    ok = []
    for f in sorted(glob.glob(str(C.DATA / args.split / "ep_*.npz")))[:args.n]:
        seed = int(f.split("_")[-1][:-4])
        z = np.load(f)
        A = z["action"]
        e = EvalEnv(seed, set(), renderer=None)
        assert A.shape[1] == e.io.dim, f"{f}: action dim {A.shape[1]} != ArmIO dim {e.io.dim}"
        s0 = e.io.state(e.d.qpos)
        start_err = float(np.abs(s0 - z["state"][0]).max())
        for i in range(len(A)):
            e.act(A[i], A[min(i + 1, len(A) - 1)])
            if e.done:
                break
        for _ in range(50):   # settle with the last action held
            if e.done:
                break
            e.act(A[-1], A[-1])
        ok.append(e.success)
        if e.peg_task:
            peg, hole = e.d.xpos[e.peg], e.d.xpos[e.hole]
            print(f"seed {seed}: {'inserted' if e.success else 'NOT inserted'}  peg bottom above table "
                  f"{(peg[2] - S.PEG_HEIGHT / 2 - S.TABLE_TOP_Z) * 1000:.1f} mm, xy off {np.round((peg[:2] - hole[:2]) * 1000, 1)} mm, "
                  f"start-state mismatch {start_err:.4f}", flush=True)
        else:
            print(f"seed {seed}: {'success' if e.success else 'FAILED'}  start-state mismatch {start_err:.4f}", flush=True)
        e.close()
    print(f"[{C.ROBOT} {C.SCENE}] open-loop replay success {np.mean(ok):.2f} over {len(ok)} demos")


if __name__ == "__main__":
    main()
