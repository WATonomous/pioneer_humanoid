# Use the official, lightweight ROS 2 Humble image instead of a massive NVIDIA base
ARG BASE_IMAGE=ros:humble-ros-base-jammy

################################ Source ################################
FROM ${BASE_IMAGE} AS source

# Set up the workspace correctly
ENV AMENT_WS=/root/ament_ws
WORKDIR ${AMENT_WS}/src

# Copy in the required WATO source code (matching the Isaac architecture)
COPY src/common_msgs common_msgs

# Scan for rosdeps
RUN apt-get -qq update && rosdep update && \
    rosdep install --from-paths . --ignore-src -r -s \
        | { grep 'apt-get install' || true; } \
        | awk '{print $3}' \
        | sort  > /tmp/colcon_install_list || true && \
    touch /tmp/colcon_install_list

################################# Dependencies ################################
FROM ${BASE_IMAGE} AS dependencies

ENV AMENT_WS=/root/ament_ws

# Install Rosdep requirements
COPY --from=source /tmp/colcon_install_list /tmp/colcon_install_list
RUN apt-get update -qq && \
    apt-get install -qq -y --no-install-recommends $(cat /tmp/colcon_install_list) || true

# Copy in source code from source stage
WORKDIR ${AMENT_WS}
COPY --from=source ${AMENT_WS}/src src

# Dependency Cleanup
WORKDIR /
RUN apt-get -qq autoremove -y && apt-get -qq autoclean && apt-get -qq clean && \
    rm -rf /root/* /root/.ros /tmp/* /var/lib/apt/lists/* /usr/share/doc/*

################################ Build ################################
FROM dependencies AS build

ENV AMENT_WS=/root/ament_ws

# Build ROS2 packages
WORKDIR ${AMENT_WS}
RUN if [ -n "$ROS_DISTRO" ] && [ -f "/opt/ros/$ROS_DISTRO/setup.sh" ]; then \
        . /opt/ros/$ROS_DISTRO/setup.sh && \
        colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --install-base ${WATONOMOUS_INSTALL:-/opt/watonomous}; \
    fi

# Source and Build Artifact Cleanup
RUN rm -rf src/* build/* devel/* install/* log/*

# ── MjLab Physics & Web Viewer ────────────────────────────────────────────────
# No symlinks or custom python binaries needed. Just standard system pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir mujoco mjviser numpy jax[cuda12] brax flax optax

# ── Pioneer leader-arm teleop + recording (pioneer_leader_arm_teleop.py --target mujoco) ───────
# GL/X libs for the MuJoCo viewer and offscreen camera rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl xz-utils git libgl1 libegl1 libosmesa6 \
    libx11-6 libxcursor1 libxrandr2 libxi6 libxinerama1 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*
# ffmpeg with SVT-AV1 for LeRobot videos (same build as the Isaac image).
RUN curl --proto "=https" --tlsv1.2 -sSf -L -o /tmp/ffmpeg.tar.xz \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-lgpl-shared-8.1.tar.xz && \
    tar -xf /tmp/ffmpeg.tar.xz -C /usr/local --strip-components=1 && \
    ldconfig && \
    rm /tmp/ffmpeg.tar.xz
# CPU torch (LeRobot at this commit needs <2.8), then LeRobot at the Isaac image's commit, the
# leader's servo SDK and the episode keys. Repo packages are not installed: the scripts add
# src/ paths themselves (repo mounted at /workspace/humanoid).
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir \
    "lerobot @ git+https://github.com/huggingface/lerobot.git@e670ac5daf9b76" \
    "feetech-servo-sdk>=1.0.0,<2.0.0" pynput pillow tqdm

# ── ROS Networking & Entrypoint ──────────────────────────────────────────────
ENV ROS_DOMAIN_ID=0
ENV FASTDDS_BUILTIN_TRANSPORTS=UDPv4

# Append setup sourcing to bashrc
RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'cd /root/ament_ws && colcon build --packages-select common_msgs 2>/dev/null || true' >> /root/.bashrc && \
    echo 'source /root/ament_ws/install/setup.bash 2>/dev/null || true' >> /root/.bashrc

# Hook into the team's shared entrypoint script
COPY docker/wato_ros_entrypoint.sh ${AMENT_WS}/wato_ros_entrypoint.sh
RUN chmod +x ${AMENT_WS}/wato_ros_entrypoint.sh
ENTRYPOINT ["./wato_ros_entrypoint.sh"]



################################ Develop ################################
# Run as the host user so bind-mounted files aren't root-owned.
FROM build AS develop
ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=dev
RUN getent group "${USER_GID}" >/dev/null || groupadd --gid "${USER_GID}" "${USERNAME}"; \
    id -u "${USERNAME}" >/dev/null 2>&1 || useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" --shell /bin/bash; \
    usermod -aG dialout "${USERNAME}"; \
    apt-get update && apt-get install -y --no-install-recommends sudo; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"; \
    chmod 711 /root; \
    chown -R "${USER_UID}:${USER_GID}" /root/ament_ws; \
    rm -rf /var/lib/apt/lists/*
USER ${USERNAME}
WORKDIR /root/ament_ws
