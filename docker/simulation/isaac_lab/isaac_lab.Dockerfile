# ── Stage 1: Build ROS 2 Humble against Python 3.11 ──────────────────────────
# Isaac Sim 5.1 uses Python 3.11. Standard apt rclpy is compiled for Python 3.10
# and will not load in Isaac Sim's Python. We build from source here so the
# C extension ABI tag matches (cpython-311).
FROM ubuntu:24.04 AS ros2_builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /ros2_ws

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential curl wget gnupg2 lsb-release \
    software-properties-common ca-certificates \
    pkg-config libasio-dev libtinyxml2-dev libcunit1-dev \
    libacl1-dev libpython3-dev \
    libboost-system-dev libssl-dev nlohmann-json3-dev

# Python 3.11 from deadsnakes (same minor version as Isaac Sim → same cpython-311 ABI)
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# All Python build tools via pip for Python 3.11.
# rosdep, rosinstall_generator, vcstool all installed here so their shebangs
# resolve to python3.11 (not whatever system python Noble provides).
RUN python3.11 -m pip install \
        empy==3.3.4 numpy pybind11 PyYAML lark \
        "pybind11[global]" setuptools==70.0.0 \
        colcon-common-extensions netifaces \
        rosinstall_generator vcstool rosdep && \
    ln -sf /usr/include/python3.11 /usr/include/python3

# Fetch minimal ROS 2 source packages (only what the teleop pipeline needs)
RUN mkdir -p src && \
    rosinstall_generator --deps --rosdistro humble \
        rcutils rcl rmw rclpy rclcpp \
        geometry_msgs std_msgs common_interfaces \
        rosidl_default_generators rosidl_default_runtime \
        ros2cli ros2run ros2topic ros2node ros2pkg \
    > ros2.humble.rosinstall && \
    vcs import --shallow src < ros2.humble.rosinstall

# Patch rclpy CMakeLists to hard-code Python 3.11 paths
RUN find /ros2_ws/src -name rclpy -type d | xargs -I{} /bin/bash -c \
    'if [ -f {}/CMakeLists.txt ]; then \
        sed -i "s|include_directories(\${PYTHON_INCLUDE_DIRS})|include_directories(/usr/include/python3.11)|" {}/CMakeLists.txt; \
        sed -i "s|\${PYTHON_LIBRARY}|python3.11|" {}/CMakeLists.txt; \
    fi'

RUN rosdep init && rosdep update

# Build everything and install directly into /opt/ros/humble
# (paths baked into setup.bash will match the COPY destination in the final stage)
RUN colcon build \
    --install-base /opt/ros/humble \
    --merge-install \
    --cmake-args \
        "-DPython3_EXECUTABLE=/usr/bin/python3.11" \
        "-DPYTHON_EXECUTABLE=/usr/bin/python3.11" \
        "-DPYTHON_INCLUDE_DIR=/usr/include/python3.11" \
        "-DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.11.so"

# Bundle system libs that the Isaac Sim image may not have
RUN cp /usr/lib/x86_64-linux-gnu/libtinyxml2.so*  /opt/ros/humble/lib/ 2>/dev/null || true && \
    cp /usr/lib/x86_64-linux-gnu/libssl.so*        /opt/ros/humble/lib/ 2>/dev/null || true && \
    cp /usr/lib/x86_64-linux-gnu/libcrypto.so*     /opt/ros/humble/lib/ 2>/dev/null || true

# ── Stage 2: Isaac Sim + IL stack ────────────────────────────────────────────
# SO101 sim IL: NVIDIA workshop stack (Isaac Lab 2.3.2 / Sim 5.1 / Python 3.11 / torch 2.7).
# Launch via watod: ACTIVE_MODULES="simulation_isaac" ./watod up -d && ./watod -t simulation_isaac
FROM nvcr.io/nvidia/isaac-lab:2.3.2

ENV PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
ENV PATH="/workspace/isaaclab/_isaac_sim/kit/python/bin:$PATH"
ENV ISAACLAB_PATH=/workspace/isaaclab
ENV HUMANOID_ROOT=/workspace/humanoid

# ── LeRobot (workshop coexist pattern: no-deps + constraints) ─────────────────
WORKDIR /workspace
RUN git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && git checkout e670ac5daf9b76

RUN $PYTHON -m pip install --no-deps -e /workspace/lerobot

RUN printf '%s\n' \
    "packaging==23.0" \
    "numpy==1.26.0" \
    "lxml==4.9.4" \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    "imageio==2.37.0" \
    > /tmp/constraints.txt

RUN $PYTHON -m pip install -c /tmp/constraints.txt \
    "datasets>=4.0.0,<4.2.0" \
    "diffusers>=0.27.2,<0.36.0" \
    "huggingface-hub[hf-transfer,cli]>=0.34.2,<0.36.0" \
    "accelerate>=1.10.0,<2.0.0" \
    "cmake>=3.29.0.1,<4.2.0" \
    "av>=15.0.0,<16.0.0" \
    "jsonlines>=4.0.0,<5.0.0" \
    "pynput>=1.7.7,<1.9.0" \
    "pyserial>=3.5,<4.0" \
    "wandb>=0.20.0,<0.22.0" \
    "torchcodec>=0.2.1,<0.6.0" \
    "draccus==0.10.0" \
    "deepdiff>=7.0.1,<9.0.0" \
    "feetech-servo-sdk>=1.0.0,<2.0.0"

# pi0 / pi0.5 policies (src/robot_learning/humanoid_il/pi0.5). lerobot[pi] needs HF's openpi
# transformers fork, not a PyPI release -- pinned to the commit the pinned lerobot above was
# developed against (fix/lerobot_openpi, reports 4.53.3). The base PaliGemma tokenizer is gated:
# `huggingface-cli login` on the host (HF_HOME is mounted).
RUN $PYTHON -m pip install -c /tmp/constraints.txt \
    "transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0" \
    "scipy>=1.11,<1.16"

RUN curl --proto "=https" --tlsv1.2 -sSf -L -o /tmp/ffmpeg.tar.xz \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.1-latest-linux64-lgpl-shared-7.1.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /usr/local --strip-components=1 && \
    ldconfig && \
    rm /tmp/ffmpeg.tar.xz

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 \
    libxkbcommon0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

RUN $PYTHON -m pip install --no-deps "rerun-sdk>=0.24.0,<0.27.0" && \
    $PYTHON -m pip install pyzmq && \
    $PYTHON -m pip install --upgrade pip

# ── Humanoid packages (editable; repo bind-mounted at runtime) ────────────────
COPY src/robot_learning ${HUMANOID_ROOT}/src/robot_learning
COPY src/simulation/so101_vial_task ${HUMANOID_ROOT}/src/simulation/so101_vial_task
COPY src/simulation/isaac_scenes ${HUMANOID_ROOT}/src/simulation/isaac_scenes
COPY src/simulation/humanoid_rl ${HUMANOID_ROOT}/src/simulation/humanoid_rl
COPY src/simulation/humanoid_rl_tasks ${HUMANOID_ROOT}/src/simulation/humanoid_rl_tasks
COPY src/pioneer_humanoid ${HUMANOID_ROOT}/src/pioneer_humanoid
COPY src/simulation/garment_fold_task ${HUMANOID_ROOT}/src/simulation/garment_fold_task

# Humanoid packages: --no-deps (never [sim]/[lerobot] extras — they pull torch/lerobot with deps)
# and --no-build-isolation (use the base image's setuptools; pip's PEP-517 isolated build env
# can't reach an index for setuptools>=61 in this builder).
RUN $PYTHON -m pip install --no-deps --no-build-isolation -e "${HUMANOID_ROOT}/src/robot_learning" && \
    $PYTHON -m pip install -c /tmp/constraints.txt psutil omegaconf && \
    $PYTHON -m pip install --no-deps --no-build-isolation -e "${HUMANOID_ROOT}/src/simulation/so101_vial_task" && \
    $PYTHON -m pip install --no-deps --no-build-isolation -e "${HUMANOID_ROOT}/src/simulation/isaac_scenes" && \
    $PYTHON -m pip install --no-deps --no-build-isolation -e "${HUMANOID_ROOT}/src/simulation/humanoid_rl" && \
    $PYTHON -m pip install --no-deps --no-build-isolation -e "${HUMANOID_ROOT}/src/simulation/humanoid_rl_tasks" && \
    $PYTHON -m pip install --no-deps --no-build-isolation -e "${HUMANOID_ROOT}/src/simulation/garment_fold_task" && \
    $PYTHON -m pip install --no-deps --no-build-isolation -e "${HUMANOID_ROOT}/src/pioneer_humanoid"

RUN mkdir -p /tmp/pycache && chmod 1777 /tmp/pycache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache

# ── ROS 2 Humble (pre-built for Python 3.11 in ros2_builder stage) ───────────
COPY --from=ros2_builder /opt/ros/humble /opt/ros/humble

# Build tools for custom packages (common_msgs, quest_teleop).
# We install deadsnakes Python 3.11 here — the same version used in ros2_builder —
# so that cmake export paths baked into std_msgs/geometry_msgs resolve correctly.
# Those .cmake files bake in the numpy include path from the ros2_builder stage:
#   /usr/local/lib/python3.11/dist-packages/numpy/_core/include   (numpy >= 2.x layout)
# numpy is intentionally UNPINNED here to stay in sync with ros2_builder (also unpinned),
# so both stages always resolve to the same major-version layout.
# NEVER pip-install anything into Isaac Sim's own bundled Python ($PYTHON) for ROS2 purposes —
# that interpreter uses pip_prebundle paths that have no C headers, and polluting it
# causes packaging-version conflicts with lerobot/isaaclab-rl.
# This Python is ONLY used for colcon builds; Isaac Sim uses its own bundled Python.
# The resulting .so files share the cpython-311 ABI tag and load in Isaac Sim fine.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential software-properties-common \
    libboost-system-dev libssl-dev nlohmann-json3-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends python3.11 python3.11-dev && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | /usr/bin/python3.11 && \
    /usr/bin/python3.11 -m pip install --upgrade pip setuptools==70.0.0 wheel && \
    /usr/bin/python3.11 -m pip install numpy "empy==3.3.4" lark catkin_pkg \
        colcon-common-extensions netifaces && \
    rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=0
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH}

# ament_ws: symlinks into bind-mounted repo, resolved at runtime.
RUN mkdir -p /root/ament_ws/src && \
    ln -s /workspace/humanoid/src/common_msgs /root/ament_ws/src/common_msgs && \
    ln -s /workspace/humanoid/src/teleop /root/ament_ws/src/teleop

COPY docker/simulation/isaac_lab/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN cat >> /root/.bashrc <<'EOF'
export ISAACLAB=/workspace/isaaclab
export HUMANOID_ROOT=/workspace/humanoid
export TASK_ROOT=/workspace/humanoid/src/simulation/so101_vial_task
export RL_RUNNERS=/workspace/humanoid/src/simulation/humanoid_rl/humanoid_rl/scripts
alias train-policy='$PYTHON -m lerobot.scripts.lerobot_train'
alias record-demos='cd $TASK_ROOT && PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p scripts/lerobot_agent.py'
alias eval-policy='cd $TASK_ROOT && PYTHONPATH=$(pwd) $ISAACLAB/isaaclab.sh -p scripts/lerobot_eval.py'
# humanoid_rl / humanoid_rl_tasks are pip-installed (editable). Checkpoints go to
# $HUMANOID_ROOT/outputs/rl/ regardless of cwd.
alias rl-train='cd $HUMANOID_ROOT && $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/train.py'
alias rl-play='cd $HUMANOID_ROOT && $ISAACLAB/isaaclab.sh -p $RL_RUNNERS/play.py'
EOF

RUN cat >> /root/.bashrc <<'EOF'
if [ -f /opt/ros/humble/setup.bash ]; then
  source /opt/ros/humble/setup.bash
  # Build common_msgs and quest_teleop using deadsnakes Python 3.11.
  # We use /usr/bin/python3.11 (not Isaac Sim's Python) because cmake export files
  # for std_msgs/geometry_msgs reference numpy headers at the deadsnakes pip path.
  # The cpython-311 ABI tag is identical so .so files load in Isaac Sim's Python.
  cd /root/ament_ws && colcon build --packages-select common_msgs quest_teleop \
      --cmake-args \
          "-DPython3_EXECUTABLE=/usr/bin/python3.11" \
          "-DPYTHON_EXECUTABLE=/usr/bin/python3.11" \
      2>/dev/null || true
  source /root/ament_ws/install/setup.bash 2>/dev/null || true
  cd - > /dev/null
fi
EOF

WORKDIR ${HUMANOID_ROOT}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
