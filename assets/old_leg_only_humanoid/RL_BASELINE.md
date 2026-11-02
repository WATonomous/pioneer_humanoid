# Frozen rough-locomotion research baseline

The rough-no-stairs selective-knee recipe and stairs scaffold use a separate
robot baseline so their existing research configuration does not overwrite the
latest upstream robot definition.

| Robot definition | Users |
| --- | --- |
| `old_leg_only_humanoid.urdf`, `usd/`, and `pioneer_humanoid/old_leg_only.py` | Upstream robot and legacy Flat/Rough tasks; unchanged by this contribution |
| `rl_baseline_v1.urdf` and `pioneer_humanoid/rough_baseline.py` | Rough-no-stairs selective-knee recipe and stairs scaffold only |

The isolated URDF and robot configuration preserve the research baseline from
this PR before the upstream asset merge. They retain its knee axes and limits,
nominal joint stance, 0.84 m spawn height, and actuator settings. The thirteen
shared STL meshes are identical to those used by that baseline and are reused
rather than duplicated. Generated USD layers are not checked in: Isaac converts
the URDF on first launch and caches the result under ignored
`outputs/isaac/rough_baseline_v1/`. Both the baseline configuration and selective
self-collision task use this location, without rewriting upstream assets.
Subsequent launches reuse the cache when its conversion settings and source
hash match. Use the pinned team Isaac Lab 2.3.2 / Isaac Sim 5.1 stack; conversion
under another simulator version is not a policy-compatibility guarantee.

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
hardware asset change. Validate cache-generation changes with the configuration
and runtime smoke checks in the pinned image, starting without an existing
`outputs/isaac/rough_baseline_v1/` cache so stale conversion output cannot hide a
problem. No new locomotion or hardware result is implied by cache relocation.

Training and handoff notes:

- [Rough-no-stairs recipe](../../src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/locomotion.md#rough-no-stairs-source-only-recipe)
- [Stairs scaffold](../../src/simulation/humanoid_rl_tasks/humanoid_rl_tasks/locomotion/STAIRS.md)
