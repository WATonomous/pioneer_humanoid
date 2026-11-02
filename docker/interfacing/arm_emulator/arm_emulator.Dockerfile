# The arm emulator: real joint_command + MuJoCo. No CAN tools and no can_node, on purpose.
#   target core:  emulator, joint_command, tests
#   target full:  core + LeRobot and the leader's servo SDK, for teleop, recording and policies
ARG BASE_IMAGE=ros:humble-ros-base-jammy

################################ Core ################################
FROM ${BASE_IMAGE} AS core

ENV AMENT_WS=/root/ament_ws
ENV WATONOMOUS_INSTALL=/opt/watonomous
WORKDIR ${AMENT_WS}/src
COPY src/interfacing/joint_command joint_command
COPY src/common_msgs common_msgs

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake g++ libyaml-cpp-dev python3-pip python3-yaml \
      libgl1 libegl1 libosmesa6 libglfw3 libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*
# Same pins as mjlabs.Dockerfile.
RUN pip3 install --no-cache-dir mujoco==3.15.0 numpy==2.2.6 scipy

WORKDIR ${AMENT_WS}
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF --install-base ${WATONOMOUS_INSTALL} && \
    rm -rf src build log

# Written at container start by sim_config (entrypoint), readable by the host user compose runs as.
RUN mkdir -p /calibration /opt/joint_command_config && \
    chmod -R a+rwX /calibration /opt/joint_command_config ${WATONOMOUS_INSTALL}/joint_command/share/joint_command/config && \
    chmod 755 /root
COPY docker/interfacing/arm_emulator/entrypoint.sh /usr/local/bin/arm_emulator_entrypoint.sh
# A Windows checkout can leave CRLF line ends, which break the shebang.
RUN sed -i 's/$//' /usr/local/bin/arm_emulator_entrypoint.sh && chmod +x /usr/local/bin/arm_emulator_entrypoint.sh
ENV ROS_HOME=/tmp/ros
ENV HOME=/tmp/home
ENV PYTHONPATH=/workspace/humanoid/src/interfacing/arm_emulator:/workspace/humanoid/src/interfacing/arm_emulator/scenarios
WORKDIR /workspace/humanoid
ENTRYPOINT ["/usr/local/bin/arm_emulator_entrypoint.sh"]

################################ Full ################################
FROM core AS full

RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils git && rm -rf /var/lib/apt/lists/*
# ffmpeg with SVT-AV1 for LeRobot videos, CPU torch, LeRobot at the commit the repo pins, the servo SDK: as mjlabs.Dockerfile.
RUN curl --proto "=https" --tlsv1.2 -sSf -L -o /tmp/ffmpeg.tar.xz \
      https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-lgpl-shared-7.1.tar.xz && \
    echo "f5f0ad52c6ee28a222eb10838c231469a10ad325f84063d3bc0aadf08164b3ec  /tmp/ffmpeg.tar.xz" | sha256sum -c - && \
    tar -xf /tmp/ffmpeg.tar.xz -C /usr/local --strip-components=1 && ldconfig && rm /tmp/ffmpeg.tar.xz
RUN python3 -m pip install --no-cache-dir pip==26.2.1 && \
    pip3 install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
      "lerobot @ git+https://github.com/huggingface/lerobot.git@e670ac5daf9b76" \
      "feetech-servo-sdk>=1.0.0,<2.0.0" pynput pillow tqdm
ENV PYTHONPATH=/workspace/humanoid/src/interfacing/arm_emulator:/workspace/humanoid/src/interfacing/arm_emulator/scenarios:/workspace/humanoid/src/robot_learning
