#!/bin/bash
# The Oct 24 whole-body model (before #468's new forearm and gripper) and its arm-swing URDF, as the policy was trained on.
set -e
cd "$(dirname "$0")"
git -C "$(git rev-parse --show-toplevel)" archive d104f229 assets/pioneer_whole_body_humanoid | tar -x
mkdir -p assets/pioneer_leg_only_humanoid pioneer_leg_only_humanoid
python3 gen_isaac_urdf.py
rmdir assets/pioneer_leg_only_humanoid pioneer_leg_only_humanoid
