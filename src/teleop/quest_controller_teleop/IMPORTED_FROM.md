# Import provenance

Imported from local branch `vr-teleop-v2`, commit
`4d0fce67` (the controller subtree matches `origin/vr-teleop-v2` at `91bb0021`).
The import includes `src/teleop/quest_controller_teleop` and its two
`dataset_schema_wato_arm_v2_*` recording configs; unrelated Docker,
robot, scene and hardware changes from that branch were not imported.

The controller/transport/IK implementation and its tests are retained. The
standalone entry point's recorder imports use this branch's package layout. The page
also accepts optional application-specific control instructions in feedback.
NASA cargo HG-DAgger integration lives in
`src/robot_learning/humanoid_robot_learning/sim/cargo_vr.py`, with the existing
policy scene, cameras, recorder and human-authority gate.

See [NASA cargo VR HG-DAgger](../../../assets/nasa_cargo/VR_HG_DAGGER.md).
