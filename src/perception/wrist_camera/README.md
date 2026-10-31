# wrist_camera

ROS 2 node for the real wrist camera. It opens the camera at 1920x1080, rectifies each frame with
`wrist_cam.rectify()` to the 640x480 pinhole the sim wrist cameras render, and publishes it.

| Topic | Type | |
|---|---|---|
| `/wrist_cam/image_rect` | `sensor_msgs/Image`, rgb8, 640x480 | what policies and recordings use; the dataset schema's `wrist_left` reads it |
| `/wrist_cam/camera_info` | `sensor_msgs/CameraInfo` | the pinhole: f = 723 px, centred, zero distortion |

Both topics are reliable QoS, since the recorder subscribes reliable. How the calibration and the
pinhole are made: [camera_calibration/README.md](../camera_calibration/README.md).

## Run in Docker (perception module)

It's the `wrist_camera` service in `modules/docker-compose.perception.yaml`: the perception image,
with host networking and `/dev`. Add `perception` to `ACTIVE_MODULES` in
`watod-config.local.sh`, then:

```bash
ls /dev/v4l/by-id/                      # find the camera: usb-Sonix_..._Webcam_...-video-index0
WRIST_CAM_DEVICE=/dev/v4l/by-id/usb-Sonix_...-video-index0 ./watod up wrist_camera
```

`WRIST_CAM_DEVICE` defaults to `/dev/video0`. Rebuild the image after changing the node's code; the
`wrist_cam.py` and `pioneer_humanoid` sources are mounted, so calibration updates apply on restart.

## Setup outside Docker (once, in the ROS 2 environment that runs the node)

```bash
sudo apt install python3-opencv            # if the image doesn't already have it
pip install -e src/pioneer_humanoid -e src/perception/camera_calibration   # rectify() + the calibration YAMLs
colcon build --packages-select wrist_camera && source install/setup.bash
```

## Run

Find the camera's stable path (`/dev/videoN` numbers can change after a reboot or replug):

```bash
ls /dev/v4l/by-id/          # usb-Sonix_..._Webcam_...-video-index0
```

```bash
ros2 launch wrist_camera wrist_camera.launch.py device:=/dev/v4l/by-id/usb-Sonix_...-video-index0
```

The node exits at startup if the camera can't deliver 1920x1080, because the calibration is only
valid for that mode. It asks for MJPG, since raw 1080p doesn't fit through USB 2.0 at 30 fps.

## Watch it (e.g. during teleop)

From any machine on the same network and `ROS_DOMAIN_ID`:

```bash
ros2 run rqt_image_view rqt_image_view /wrist_cam/image_rect
```

Foxglove works too. Without ROS, on the machine the camera is plugged into,
`python3 src/perception/camera_calibration/wrist_cam.py --camera <device>` shows raw and rectified
side by side.

## Notes

- Run it with host networking (as the interfacing containers do), or its topics won't reach nodes
  in other containers.
- Frames are stamped when `read()` returns, so the stamp includes the camera's capture and USB
  latency. Time sync with joint states is still a to-do.
