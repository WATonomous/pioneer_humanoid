# Wrist Stereo Depth - Complete Explainer

This document explains everything about the wrist stereo depth system in plain English. Use this to understand what we built and why, or to explain it to someone else.

---

## The Problem We're Solving

Pioneer (the humanoid robot) has a depth camera on its chest - an Intel RealSense D455. This camera can see how far away objects are, which is essential for tasks like picking things up.

**But there's a problem:** When the robot does close-range manipulation work (like inserting a vial into a rack, or threading a needle), the chest camera:

1. Is too far away from the action
2. Is at a bad viewing angle (looking down at an angle instead of straight at the task)
3. Often can't see the workspace because the robot's own arms are in the way

So we need depth sensing from the wrists - right where the action is.

---

## The Hardware Situation

The robot has two small cameras called **HBVCAMs** mounted on the wrists - one on each arm. These are cheap, lightweight cameras that are good for this application.

**The catch:** These are just regular RGB cameras. They see color images but have no built-in depth sensing. Unlike the RealSense on the chest, they can't directly tell you how far away something is.

---

## The Solution: Stereo Vision

Humans see depth using two eyes. Your brain compares what your left eye sees versus your right eye, and from the slight differences, it figures out distance. This is called **stereo vision**.

We do the same thing with the two wrist cameras:

1. Take an image from the left camera
2. Take an image from the right camera
3. Find matching points between them (e.g., "this corner in the left image corresponds to this corner in the right image")
4. Measure how much those points are shifted between the two images (this shift is called **disparity**)
5. Use geometry to convert disparity into depth

**The key insight:** If you know exactly where both cameras are in 3D space, and you know how far apart they are (the **baseline**), then the math is straightforward:

```
depth = baseline × focal_length / disparity
```

---

## The Complication: Dynamic Baseline

Normal stereo camera systems (like the RealSense) have two cameras bolted together at a fixed distance. You calibrate it once and you're done forever.

Our setup is weird: the cameras are on **different arms**. As the robot moves, the distance and angle between the cameras changes constantly. Every single frame could have a different baseline.

This is called **dynamic baseline stereo**, and it's harder than fixed stereo because:

1. You can't pre-compute the stereo geometry
2. You need to know where each camera is in 3D space at every moment
3. You have to recompute the stereo math on the fly

---

## How We Know Where the Cameras Are

The robot knows where its arms are. Every joint has a sensor (encoder) that reports its current angle. The robot also has a **URDF file** - essentially a blueprint that describes the robot's geometry (how long each link is, how joints connect them, etc.).

Using the joint angles plus the URDF, we can do **forward kinematics (FK)** - math that answers: "Given these joint angles, where is the wrist in 3D space?"

So at runtime:
1. Read joint angles from the robot
2. Run FK to compute where each wrist is
3. Apply a known offset from wrist to camera (the **hand-eye calibration**)
4. Now we know where both cameras are in 3D space

---

## The Two Calibrations We Need

Before the system can work, we need to calibrate two things for each camera:

### 1. Intrinsic Calibration (Camera Internals)

Every camera lens is slightly different. We need to measure:

- **Focal length** - How "zoomed in" the camera is. Affects the relationship between pixels and real-world angles.
- **Principal point** - Where the optical center of the lens is in the image. Usually close to the center, but not exactly.
- **Distortion** - How much the lens warps the image. Cheap lenses have more distortion (straight lines appear curved).

**How we measure it:** Point the camera at a checkerboard pattern. Capture 15-20 images from different angles. The calibration software knows the checkerboard is a flat grid of known size, so by seeing how it appears distorted in images, it can calculate the lens properties.

**Output:** A file like `left_intrinsics.yaml` containing the camera matrix (K) and distortion coefficients (D).

### 2. Hand-Eye Calibration (Camera Mounting Position)

We need to know exactly where each camera is mounted relative to its wrist. Even if someone tells you "it's bolted to the side of the wrist," that's not precise enough - we need millimeter-level accuracy.

**How we measure it:** 

1. Fix a checkerboard somewhere in the workspace (tape it to the table)
2. Move the robot's wrist to different poses while keeping the checkerboard in view
3. At each pose, we know two things:
   - Where the checkerboard is relative to the camera (from image detection)
   - Where the wrist is relative to the robot base (from FK)
4. With enough poses, we can solve for the unknown: where is the camera relative to the wrist?

This is a classic problem called **AX = XB hand-eye calibration**. The math is well-established.

**Output:** A file like `left_hand_eye.yaml` containing the camera-to-wrist transform.

---

## The Runtime Pipeline

Once calibrated, here's what happens every frame:

### Step 1: Get Inputs
- Receive images from both wrist cameras
- Receive joint angles from the robot

### Step 2: Compute Camera Poses
- Run FK to get wrist poses from joint angles
- Apply hand-eye calibration to get camera poses

### Step 3: Compute Stereo Geometry
- Calculate the relative pose between the two cameras (rotation and translation)
- Extract the baseline (distance between cameras)

### Step 4: Rectify Images
- Stereo matching assumes corresponding points are on the same horizontal row in both images
- Real cameras aren't perfectly aligned, so we warp both images to make this true
- This warping is called **rectification**

### Step 5: Stereo Matching (SGBM)
- For each pixel in the left image, search for the matching pixel in the right image
- The algorithm (SGBM = Semi-Global Block Matching) looks at small patches of pixels and finds the best match
- Output: a **disparity map** - how many pixels each point shifted between left and right images

### Step 6: Convert to Depth
- Use the formula: `depth = baseline × focal_length / disparity`
- Filter out invalid values (too close, too far, failed matches)

### Step 7: Publish
- Publish the depth image on `/wrist_stereo/depth`
- Optionally publish a 3D point cloud on `/wrist_stereo/points`

---

## Rolling Shutter Problem

The HBVCAM uses an **OV3660 sensor**, which has a **rolling shutter**. This means it doesn't capture the entire image at once - it scans from top to bottom over time (a few milliseconds).

**Why this matters:** If the arm is moving while the image is being captured, the top and bottom of the image are from slightly different moments in time. The image looks "smeared" or distorted. This breaks stereo matching.

**Our solution:** The node monitors joint velocities. It only processes frames when:
1. The arm has been stationary (velocity below threshold) 
2. For at least 50ms (configurable)

This means depth is only computed during brief pauses. For manipulation tasks, this is fine - you typically pause before grasping anyway. The parameter `require_stationary` can be set to `false` if you want to disable this protection.

---

## What Each File Does

```
wrist_stereo/
├── package.xml                 # ROS 2 package metadata (dependencies, etc.)
├── setup.py                    # Python installation config
├── README.md                   # Quick-start usage guide
├── EXPLAINER.md                # This document
│
├── config/
│   └── stereo_config.yaml      # Tunable parameters (topics, thresholds, etc.)
│
├── launch/
│   └── stereo_depth.launch.py  # Launch file for easy startup
│
└── wrist_stereo/
    ├── calibrate_intrinsics.py # Tool: measure camera lens properties
    ├── calibrate_hand_eye.py   # Tool: measure camera mounting position
    └── stereo_depth_node.py    # Runtime: images + joints → depth
```

---

## How to Use It (Wednesday Checklist)

### Before You Start
- [ ] Both wrist cameras are connected and publishing images
- [ ] Joint states are being published
- [ ] You have a checkerboard (print one if needed - 9x6 or similar, known square size)

### Step 1: Calibrate Left Camera Intrinsics (~10 min)
```bash
ros2 run wrist_stereo calibrate_intrinsics --ros-args \
  -p camera_topic:=/wrist_left/image_raw \
  -p output_file:=config/left_intrinsics.yaml
```
- Point checkerboard at camera from different angles
- Press SPACE to capture (~15-20 images)
- Press C to calibrate
- Press S to save

### Step 2: Calibrate Right Camera Intrinsics (~10 min)
Same as above but for the right camera.

### Step 3: Calibrate Left Hand-Eye (~15 min)
```bash
ros2 run wrist_stereo calibrate_hand_eye --ros-args \
  -p camera_topic:=/wrist_left/image_raw \
  -p intrinsics_file:=config/left_intrinsics.yaml \
  -p output_file:=config/left_hand_eye.yaml \
  -p urdf_file:=/path/to/pioneer_bimanual_arm.urdf \
  -p wrist_link:=link6l
```
- Tape checkerboard to the table (don't move it)
- Move the arm to different poses viewing the checkerboard
- Press SPACE at each pose (~10-15 poses)
- Press C to calibrate
- Press S to save

### Step 4: Calibrate Right Hand-Eye (~15 min)
Same as above but for right camera with `wrist_link:=link6`.

### Step 5: Run Stereo Depth
```bash
ros2 launch wrist_stereo stereo_depth.launch.py \
  urdf_file:=/path/to/pioneer_bimanual_arm.urdf
```

### Step 6: Verify
```bash
ros2 topic echo /wrist_stereo/depth  # Should see depth messages
rviz2  # Visualize the depth image or point cloud
```

---

## Future Improvements

The current system uses **SGBM** (Semi-Global Block Matching), a classical algorithm from 2008. It's fast and works reasonably well, but:

- Struggles with textureless surfaces (white walls, blank objects)
- Struggles with reflective/transparent surfaces
- Can be noisy

**Phase 2** (mentioned by spiffy) is to replace SGBM with a **learned model** like:

- **MASt3R** - A transformer-based model from Meta that's very accurate
- **RAFT-Stereo** - An optical flow network adapted for stereo

These learned models are more accurate but require a GPU and are slower. We start with SGBM to get something working, then upgrade if needed.

---

## Key Terms Glossary

| Term | Meaning |
|------|---------|
| **Stereo vision** | Computing depth by comparing two camera views |
| **Baseline** | Distance between the two cameras |
| **Disparity** | How much a point shifts between left and right images (in pixels) |
| **Rectification** | Warping images so matching points are on the same row |
| **SGBM** | Semi-Global Block Matching - the stereo algorithm we use |
| **Forward kinematics (FK)** | Computing where links are in 3D space from joint angles |
| **Hand-eye calibration** | Finding where a camera is mounted on a robot link |
| **Intrinsic calibration** | Finding a camera's internal properties (focal length, distortion) |
| **Rolling shutter** | Sensor that captures image row-by-row (causes blur if moving) |
| **URDF** | Robot description file (geometry, joints, links) |

---

## If Someone Asks...

**"What is this?"**
> A system that gives us depth sensing from the wrist cameras so the robot can see close-up manipulation tasks that the chest camera can't see.

**"Why not just use the chest camera?"**
> The chest camera is too far away and at a bad angle for close-range work like inserting vials. The arms often block its view too.

**"Why not just buy depth cameras for the wrists?"**
> Depth cameras (like RealSense) are heavier and bulkier. The HBVCAMs are tiny and light, which is better for wrist mounting. Plus stereo from two separate cameras gives us a wider baseline when the arms are spread apart, which means more accurate depth.

**"How does it work in one sentence?"**
> We use the robot's joint angles to figure out where each wrist camera is in space, then do stereo matching between the two camera images to compute depth.

**"What's the hard part?"**
> The cameras are on different arms, so the distance between them changes constantly. We have to recompute the stereo geometry every frame using forward kinematics.

---

*Document last updated: November 2026*
