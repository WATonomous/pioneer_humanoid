"""Visualize lidar rays and hits at default 20m offset on spiral stairs.
Run: DISPLAY=:10 isaaclab.sh -p tools/visualize_height_scanner_20m.py
"""
import argparse
import math

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--radius", type=float, default=1.515)
parser.add_argument("--angle", type=float, default=-30.8)
parser.add_argument("--height", type=float, default=0.84)
parser.add_argument("--max-dist", type=float, default=50.0)
args = parser.parse_args()

app = AppLauncher({"headless": False}).app

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdGeom, Sdf
import omni.physx

# Spiral terrain USD path
SPIRAL_USD = "/workspace/humanoid/assets/staircase_onshape/spiral_terrain.usd"

# Create world
world = World(stage_units_in_meters=1.0, physics_dt=1/200, rendering_dt=1/60)
stage = world.stage

# Load the spiral terrain
add_reference_to_stage(SPIRAL_USD, "/World/ground")

# Lights
dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
dome_light.GetAttribute("inputs:intensity").Set(500.0)
dome_light.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0, 1.0, 1.0))

distant_light = stage.DefinePrim("/World/DistantLight", "DistantLight")
distant_light.GetAttribute("inputs:intensity").Set(500.0)
distant_light.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0, 1.0, 0.9))
distant_xform = UsdGeom.Xform(distant_light)
distant_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

world.reset()

# Get PhysX scene query interface
physx_scene_query = omni.physx.get_physx_scene_query_interface()

# Scanner position - use DEFAULT 20m offset (base config)
theta = math.radians(args.angle)
x = args.radius * math.cos(theta)
y = args.radius * math.sin(theta)
z = args.height + 20.0  # DEFAULT 20m offset from base config
origin = Gf.Vec3f(x, y, z)
direction = Gf.Vec3f(0, 0, -1)  # straight down

print(f"\n=== Lidar visualization at r={args.radius:.3f}, angle={args.angle:.1f}° ===")
print(f"Scanner origin: ({x:.3f}, {y:.3f}, {z:.3f})  (base + 20m)")
print(f"Ray direction: down (-Z)")
print(f"Max distance: {args.max_dist}m")

# Add camera
cam_prim = stage.DefinePrim("/World/camera", "Camera")
cam = UsdGeom.Camera(cam_prim)
cam_xform = UsdGeom.Xform(cam_prim)
cam_xform.AddTranslateOp().Set(Gf.Vec3d(x + 3.0, y + 3.0, 4.0))
cam_xform.AddOrientOp().Set(Gf.Quatf(0.5, -0.5, 0.5, -0.5))
cam.CreateFocalLengthAttr().Set(24.0)
cam.CreateFocusDistanceAttr().Set(5.0)
cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.1, 100.0))

# Scanner origin marker (green)
ref_prim = stage.DefinePrim("/World/scanner_ref", "Xform")
ref_xform = UsdGeom.Xform(ref_prim)
sphere_prim = stage.DefinePrim("/World/scanner_ref/origin_sphere", "Sphere")
sphere_prim.GetAttribute("radius").Set(0.05)
xform = UsdGeom.Xform(sphere_prim)
xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
UsdGeom.Gprim(sphere_prim).CreateDisplayColorAttr().Set([(0.0, 1.0, 0.0)])

# Lidar hits parent
stage.DefinePrim("/World/lidar_hits", "Xform")

# Fire rays in a grid pattern (16x10 like the height scanner)
n_x, n_y = 16, 10
size_x, size_y = 1.6, 1.0
hits_data = []

for iy in range(n_y):
    for ix in range(n_x):
        offset_x = (ix - (n_x - 1) / 2) * (size_x / (n_x - 1))
        offset_y = (iy - (n_y - 1) / 2) * (size_y / (n_y - 1))
        
        ray_origin = Gf.Vec3f(x + offset_x, y + offset_y, z)
        hit = physx_scene_query.raycast_closest(ray_origin, direction, args.max_dist)
        
        if hit["hit"]:
            hit_pos = hit["position"]
            dist = hit["distance"]
            
            # Red sphere at hit point
            sphere_path = f"/World/lidar_hits/hit_{ix}_{iy}"
            sphere_prim = stage.DefinePrim(sphere_path, "Sphere")
            sphere_prim.GetAttribute("radius").Set(0.02)
            xform = UsdGeom.Xform(sphere_prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(hit_pos[0], hit_pos[1], hit_pos[2]))
            UsdGeom.Gprim(sphere_prim).CreateDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
            
            # Ray line
            line_path = f"/World/lidar_rays/ray_{ix}_{iy}"
            line = UsdGeom.BasisCurves.Define(stage, line_path)
            line.CreatePointsAttr([(ray_origin[0], ray_origin[1], ray_origin[2]), 
                                    (hit_pos[0], hit_pos[1], hit_pos[2])])
            line.CreateCurveVertexCountsAttr([2])
            line.CreateWidthsAttr([0.003])
            line.CreateDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
        else:
            # Miss - gray line
            end_pos = Gf.Vec3f(ray_origin[0], ray_origin[1], ray_origin[2] - args.max_dist)
            line_path = f"/World/lidar_rays/ray_{ix}_{iy}"
            line = UsdGeom.BasisCurves.Define(stage, line_path)
            line.CreatePointsAttr([(ray_origin[0], ray_origin[1], ray_origin[2]), 
                                    (end_pos[0], end_pos[1], end_pos[2])])
            line.CreateCurveVertexCountsAttr([2])
            line.CreateWidthsAttr([0.003])
            line.CreateDisplayColorAttr().Set([(0.5, 0.5, 0.5)])

print(f"Total rays: {n_x * n_y}")
print("Running until Ctrl+C...")

try:
    while True:
        world.step(render=True)
except KeyboardInterrupt:
    pass