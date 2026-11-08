# Article-based control rebuild

Reviewed on 2026-11-06 against the complete
[article](https://aurelarnold.xyz/blog/vr-teleoperation-stack/#control-that-feels-natural)
and its author's [source](https://github.com/Dream-Machines-Robotics/vr-teleop-kit/tree/8d37f3d6bd44c8ce646f7bc110ac98e802e0ab57),
specifically `core/pose_mapping.py`, `ik/decoupled_ik.py`, and `ik/model.py`.

The single always-on launcher uses `ArticleTarget` and `PioneerArticleIK` in
`article_control.py`. Wrist-pivot calibration and Pioneer-specific orientation
adaptations remain. No grip button is required to move an arm.

## Mapping and sequential solve

1. Engage at measured FK and the controller pose, capturing heading once.
2. Read position at the calibrated anatomical wrist pivot, not the grip origin.
3. Map translation increments with the configured gain. Map world-frame rotation
   increments by conjugation, then left-multiply the accepted target.
4. Apply the article's slipping-clutch lead limits to both channels. Excess
   movement is absorbed and reversing responds immediately.
5. Warm-start from the previous command. Solve three-joint position DLS.
6. Bound the position step by physical/rate limits and recompute FK.
7. Solve separate three-joint orientation DLS from the updated pose, using an
   SO(3) error and rotational Jacobian expressed in the same base frame.
8. Apply physical/rate bounds. Each subproblem has independent determinant-based
   adaptive damping. An orientation error beyond 2.2 rad parks that group.

These are fresh implementations of the equations, sharing only existing URDF
geometry and rotation utilities, without a dependency on the DK1 MuJoCo model.

The lead policy intentionally restores the article's behavior. The earlier
fast-out/slow-return example losing alignment is a consequence of absorbed
motion, explicitly acknowledged by the author, rather than proof that this
policy is an implementation error. The intervening persistent-target change
departed from that policy. Reengaging realigns after absorbed movement; continuous
exact correspondence at a joint or reach limit is not promised.

## Required Pioneer geometry adaptation

DK1 uses position joints 1/2/3 and orientation joints 4/5/6. On Pioneer, joints
4 and 6 are parallel at home, making that orientation block rank deficient;
the literal position block is also poorly conditioned there.

| Task | Pioneer joints | Home determinant magnitude, left / right |
| --- | --- | --- |
| Position | 1, 2, 4 | about 0.0153 / 0.00974 cubic metres |
| Orientation | 3, 5, 6 | 1 / 1 |

Position targets joint5's origin; orientation targets link6's frame. Joint 3
also changes this wrist anchor's position. The next position step corrects that
movement, so the tasks are not exactly geometrically decoupled. Transient
position errors and necessary proximal movement remain, particularly near
limits. All three rotation directions are tracked instead of ignoring one.

The reference source itself accepts residual tool-position error from a
non-spherical wrist. Pioneer's coupling differs and can be larger. This is a
methodology port, not an identical joint-index split or equivalent hardware.
The mapper directly targets the wrist anchor, so no separate tool-tip-to-anchor
position conversion is needed.

## Parameters and deliberate differences

Position DLS uses baseline 0.05, additional damping 0.15, and a Pioneer-scaled
manipulability threshold 0.015. Orientation uses the reference baseline 0.05,
additional damping 0.4, and threshold 0.5. Pioneer's singularities are measured
from its own Jacobian, not DK1's robot-specific cosine identity.

A weak position rest bias uses mu=0.002 when position error exceeds 1 mm.
The smaller bias and error gate avoid pulling a stationary holding pose toward
home. No orientation/home spring is applied. These are Pioneer tuning choices.
Position bounds are applied before the orientation FK, whereas the reference
applies its final bounds later. Both physical joint limits and existing command
caps remain: 0.9 rad/s on joints 1–4 and 3.6 rad/s on joints 5–6.

Camera/heading calibration, pivot fitting, left-Y reset, precision, both scenes,
and always-on activation remain. No motor settings or robot assets changed.

## Verification and remaining uncertainty

Offline checks cover both arms tracking ±20 degrees about each controller axis,
then holding still without returning toward home; reachable mixed FK-generated
poses; physical/rate bounds; stops; near-straight-arm damping; antipode parking;
frame conjugation; precision; pivot calibration; reengagement; absorbed excess;
and immediate reversal without a hidden backlog. Tested reachable cases converge
within 1 mm and 0.1 degrees. Activation/tracking/reset regressions exercise the always-on article solver.
Legacy mathematical solver tests remain for comparison.

All twelve arm joints' USD frame translations, rotations and axis signs are
checked against the URDF. That check passes with the local USD bindings.
These checks do not measure live controller noise, Isaac motor/contact behavior,
or headset feel. No simulator/container was launched. Live wrist behavior still needs verification in Isaac with the Quest.
