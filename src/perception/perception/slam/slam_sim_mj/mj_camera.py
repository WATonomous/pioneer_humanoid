"""MuJoCo offscreen RGB-D camera with real RealSense D455 intrinsics.

Loads a scene MJCF and attaches a mocap-driven camera to it, so the scene file
stays a plain room that can be swapped for a downloaded one without edits.

Frame convention, which is the easiest thing here to get silently backwards:

  * The mocap body IS the ROS optical frame  (+Z boresight, +X right, +Y down),
    matching what orbit_path.py produces and what the images are stamped with.
  * MuJoCo's camera frame is OpenGL's (-Z boresight, +X right, +Y up), so the
    camera hangs off that body with a fixed 180-degree rotation about X.

Doing it this way means poses go straight from orbit_path into mocap_quat with no
conversion at the call site -- there is exactly one place the convention lives.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# Intrinsics read off the real demo's rtabmap database (RTAB-Map DB viewer, Calib
# row): 640x480, fx=383.682 fy=383.165. That is a D455's aligned colour stream, so
# the simulated camera is literally the camera from that recording.
#
# The real calib also has cx=326.864 cy=242.252 -- a ~7px offset from centre. That is
# NOT reproduced here: the principal point is left at the exact centre. What matters
# to SLAM is that CameraInfo agrees with the image it describes, and it does, because
# both come from the same model. Chasing 7px of a specific physical unit's factory
# calibration buys nothing and adds a sign convention to get wrong.
D455_REF_W, D455_REF_H = 640, 480
D455_640_FX = 383.682
D455_640_FY = 383.165

# Real D455 depth limits. Outside these the sensor reports nothing, and RTAB-Map's
# convention for "no reading" is 0 -- see the publisher.
D455_MIN_RANGE = 0.4
D455_MAX_RANGE = 20.0

# Fixed rotation optical -> MuJoCo camera: 180 degrees about X, as (w, x, y, z).
_OPTICAL_TO_MJCAM = "0 1 0 0"


class MjCamera:
    def __init__(
        self,
        scene: Path,
        width: int = 640,
        height: int = 480,
        fx: float | None = None,
        fy: float | None = None,
    ) -> None:
        if not Path(scene).exists():
            raise FileNotFoundError(
                f"no scene at {scene}. The room is generated, not committed -- run:\n"
                f"    python3 {Path(__file__).parent / 'scene' / 'build_room.py'}"
            )
        self.width, self.height = width, height

        # Focal length in PIXELS scales with resolution, so that changing --width
        # changes how much detail is sampled and NOT how wide the lens is. Passing
        # the 640x480 figure straight through at 848x480 would quietly narrow the
        # field of view instead of widening it, which is the opposite of what a real
        # D455 does when you ask it for a wider mode.
        if fx is None:
            fx = D455_640_FX * width / D455_REF_W
        if fy is None:
            fy = D455_640_FY * height / D455_REF_H

        # focalpixel needs a sensorsize to convert against; any consistent choice
        # works, so pick a 0.01 "mm" pixel pitch and let MuJoCo do the arithmetic.
        sensor_w, sensor_h = width * 0.01, height * 0.01
        rig = f"""<mujoco model="slam_rig">
  <include file="{scene.resolve()}"/>
  <visual><global offwidth="{width}" offheight="{height}"/></visual>
  <worldbody>
    <body name="camera_rig" mocap="true" pos="0 0 1.5">
      <camera name="d455" mode="fixed" quat="{_OPTICAL_TO_MJCAM}"
              resolution="{width} {height}"
              sensorsize="{sensor_w} {sensor_h}"
              focalpixel="{fx} {fy}"/>
    </body>
  </worldbody>
</mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(rig)

        # znear/zfar are FRACTIONS OF model.stat.extent, not metres. Setting them to
        # metres directly clips the near plane to several metres out: two thirds of
        # every frame comes back invalid and the corridor walls -- the only thing
        # there is to map -- vanish. It looks like a broken scene, not a bad number.
        self.model.vis.map.znear = D455_MIN_RANGE / 2.0 / self.model.stat.extent
        self.model.vis.map.zfar = D455_MAX_RANGE * 2.0 / self.model.stat.extent

        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)

    def set_pose(self, position: np.ndarray, quat_wxyz: np.ndarray) -> None:
        """Place the optical frame. `quat_wxyz` is orbit_path's ordering."""
        self.data.mocap_pos[0] = position
        self.data.mocap_quat[0] = quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """(rgb uint8 HxWx3, depth float32 HxW in metres).

        MuJoCo's depth is distance to the IMAGE PLANE (verified: a wall normal to the
        boresight renders a constant value edge to edge), which is what RTAB-Map
        wants. Radial range would bow the map outward at the image edges.
        """
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera="d455")
        rgb = self.renderer.render()

        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="d455")
        depth = self.renderer.render().astype(np.float32)
        self.renderer.disable_depth_rendering()

        # Clamp to what the real sensor can see. Beyond zfar MuJoCo returns the far
        # plane rather than "nothing", which would otherwise plant a phantom wall at
        # a fixed range across the whole image.
        depth[(depth < D455_MIN_RANGE) | (depth > D455_MAX_RANGE)] = 0.0
        depth[~np.isfinite(depth)] = 0.0
        return rgb, depth

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """K, derived from the model rather than from the constructor arguments, so
        published CameraInfo cannot drift from what was actually rendered."""
        fx_mm, fy_mm = self.model.cam_intrinsic[0][:2]
        sw, sh = self.model.cam_sensorsize[0]
        fx = fx_mm / sw * self.width
        fy = fy_mm / sh * self.height
        return np.array([
            [fx, 0.0, self.width / 2.0],
            [0.0, fy, self.height / 2.0],
            [0.0, 0.0, 1.0],
        ])
