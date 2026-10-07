# `assets/`

Robot descriptions (URDF / USD / meshes) and shared scene props. Kept at the repo
root, **not** inside a sim package, so the same robot can be consumed by any
backend (Isaac Lab, mjlab, …) without importing that backend's Python.

Code resolves these by repo-relative path; nothing here is shipped by `pip install`.

```
assets/
├── pioneer_bimanual_arm/   # the bimanual arm — urdf, usd, meshes  (BIMANUAL_ARM_CFG)
├── pioneer_hand/           # 20-DOF hand — urdf, usd, meshes        (HAND_CFG)
├── whole_body_humanoid/    # legged humanoid — urdf, usd, meshes    (WHOLE_BODY_HUMANOID_CFG)
├── props/                 # block.usd, box.usd, table.usd, lightbox.usd — scene props owned by no robot
└── lerobot/               # SO101 arm + vial-task USDs + HDRIs — synced from an external repo
```

Isaac Lab articulation configs for the pioneer robots live in
`src/pioneer_humanoid/`. Task-specific assets (e.g. push-block DR textures) stay
with their task package.
