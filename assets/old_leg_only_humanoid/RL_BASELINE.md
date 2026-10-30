# Frozen rough-locomotion research baseline

The rough-no-stairs selective-knee recipe and stairs scaffold use a separate
robot baseline so their existing research configuration does not overwrite the
latest upstream robot definition.

| Robot definition | Users |
| --- | --- |
| `old_leg_only_humanoid.urdf`, `usd/`, and `pioneer_humanoid/whole_body.py` | Upstream robot and legacy Flat/Rough tasks; unchanged by this contribution |
| `rl_baseline_v1.urdf`, `usd/rl_baseline_v1/`, and `pioneer_humanoid/rough_baseline.py` | Rough-no-stairs selective-knee recipe and stairs scaffold only |

The isolated URDF and cached USD layers preserve the research baseline from
this PR before the upstream asset merge. They retain its knee axes and limits,
nominal joint stance, 0.84 m spawn height, and actuator settings. The thirteen
shared STL meshes are identical to those used by that baseline and are reused
rather than duplicated. The selective-collision task writes its generated
conversion cache under ignored `outputs/isaac/rough_baseline_v1/`, not into the
upstream checked-in USD directory.

This is **not a claim that the research baseline matches the latest hardware**.
Its geometry and dynamics have not been validated on the physical robot. The
stairs environment is an untrained scaffold. Source and wiring tests do not
establish stair-climbing performance or hardware safety.

No checkpoints, policy exports, recordings, or local experiment results are
included. A policy must be paired with its training robot definition,
observation normalizers, joint/action ordering, and action scale/default-offset
mapping. Neither unchanged task names nor matching tensor dimensions make
policies trained on different robot geometry interchangeable.

Keep future robot-definition changes explicit and separately validate their
effect on a trained policy. Do not silently replace the frozen baseline with
the upstream URDF/USD or update its joint parameters as a side effect of a
hardware asset change. The configuration and runtime smoke checks should be
rerun after this path-isolation change before claiming current simulator
compatibility; no new locomotion or hardware result is implied by relocation.

Training and handoff notes:

- [Rough-no-stairs recipe](../../src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/locomotion.md#rough-no-stairs-source-only-recipe)
- [Stairs scaffold](../../src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/STAIRS.md)
