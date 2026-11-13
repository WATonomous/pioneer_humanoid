# Import provenance

The original six-joint Isaac stack was imported from local `vr-teleop-v2`
commit `4d0fce67` (its controller subtree matched remote `91bb0021`). Its two
`dataset_schema_wato_arm_v2_*` recording configs remain available.

## Wilson's November 12 update

Compatible improvements were ported from `origin/main` at
`2f3c83777fa866a40baf04721cc52bae2762cb41`. The latest teleop change there is
`782eedf887cc872306fdeb2ca6f9f4f8feaf917f` (PR #556); the implementation includes
Wilson's input-filter, latency, anchored-target and browser-tab fixes from
PRs #535, #540 and #549–554. [UPSTREAM_REFRESH.json](UPSTREAM_REFRESH.json)
records upstream file hashes.

- `input_filter.py` is imported unchanged, with its tests adapted to the
  existing Isaac driver and six-joint arm imports.
- The upstream `ArticleTarget`/`AnchoredTarget` mappings and the driver's
  filter-reset, velocity-prediction and tracking-loss handling are ported.
- The upstream pose transport and its regressions are imported with the cargo
  camera endpoint retained. Only Meta Browser owns control; idle tabs cannot
  block the tab streaming VR poses.
- The browser retains the cargo camera panels, HUD and HG-DAgger instructions,
  while adopting click-anywhere VR entry and reconnect handling. Upstream's
  current frontend displays a dark tracking-only view; that removal is not
  applied to this headset-based cargo workflow.

Upstream now uses `pioneer_bimanual_arm_8dof`, a seven-joint arm plus gripper,
MuJoCo and mink IK. This cargo checkpoint was trained with the six-joint Isaac
arm plus gripper. Its robot, IK geometry, seven-value policy action contract,
cameras and recordings remain compatible; the upstream seven-joint model,
mink solver, standalone launcher and new recording schema are not substituted.
The Isaac driver opts into the new mapping/filter/prediction for cargo only;
its standalone legacy entry point retains its existing defaults.

Cargo integration lives in
`src/robot_learning/humanoid_robot_learning/sim/cargo_vr.py`. Startup waits for
tracked head/controllers with both grips released; right grip still selects
human authority and releasing it returns to policy without saving. Right B
explicitly saves corrections and resets; left Y discards the unsaved attempt and resets.
See [NASA cargo VR HG-DAgger](../../../assets/nasa_cargo/VR_HG_DAGGER.md).
