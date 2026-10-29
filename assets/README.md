# `assets/`

Robot descriptions (URDF / USD / meshes) and shared scene props. Kept at the repo
root, **not** inside a sim package, so the same robot can be consumed by any
backend (Isaac Lab, mjlab, …) without importing that backend's Python.

Code resolves these by repo-relative path; nothing here is shipped by `pip install`.

```
assets/
├── pioneer_bimanual_arm/   # updated Arm V2: forearms + dated wrist/grippers, keyboard MuJoCo UI; USD generated from URDF
├── pioneer_hand/           # 20-DOF hand — urdf, usd, meshes        (HAND_CFG)
├── old_leg_only_humanoid/  # the old legs-only humanoid (placeholder upper body) — urdf, usd, meshes (WHOLE_BODY_HUMANOID_CFG)
├── pioneer_whole_body_humanoid/  # the full robot from CAD (head, arms, hands, legs; 69 joints), and a gripper version (the bimanual arm's joints; 31) — urdf, MuJoCo xml, meshes
├── pioneer_leg_only_humanoid/    # the same robot without arms, for walking first (12 leg joints; shares the whole-body meshes)
├── props/                 # block.usd, box.usd, table.usd, lightbox.usd — scene props owned by no robot
└── lerobot/               # SO101 arm + vial-task USDs + HDRIs — synced from an external repo
```

Isaac Lab articulation configs for the pioneer robots live in
`src/pioneer_humanoid/`. Task-specific assets (e.g. push-block DR textures) stay
with their task package.
