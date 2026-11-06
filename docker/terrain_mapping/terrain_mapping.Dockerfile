# elevation_mapping_cupy on ROS 2 Humble with CUDA 12.8.
# robot_base has no CUDA, so this image starts from NVIDIA's CUDA image (the recipe from
# iit-DLSLab/elevation_mapping_gpu_ros2) and recreates the robot_base conventions watod uses.
# CUDA must be >= 12.8 for Blackwell GPUs (RTX 50xx, sm_120).
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04

################################ Base ################################
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Etc/UTC \
    ROS_DISTRO=humble \
    AMENT_WS=/home/bolty/ament_ws \
    WATONOMOUS_INSTALL=/opt/watonomous/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg2 lsb-release software-properties-common && \
    add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      python3-colcon-common-extensions \
      python3-pip \
      build-essential \
      git && \
    rm -rf /var/lib/apt/lists/*

################################ Dependencies ################################
FROM base AS dependencies

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-grid-map-msgs \
      ros-humble-grid-map-rviz-plugin \
      ros-humble-rtabmap-ros \
      ros-humble-imu-filter-madgwick \
      ros-humble-depth-image-proc \
      ros-humble-realsense2-camera-msgs \
      ros-humble-tf-transformations \
      ros-humble-rviz2 \
      ros-humble-cv-bridge \
      ros-humble-message-filters \
      ros-humble-sensor-msgs-py \
      ros-humble-std-srvs \
      ros-humble-rosbag2-storage-mcap \
      ros-humble-ament-cmake-pytest \
      ros-humble-ament-cmake-ros \
      ros-humble-ament-lint-auto \
      ros-humble-ament-lint-common \
      ros-humble-launch-testing-ament-cmake \
      ros-humble-launch-testing-ros \
      python3-opencv \
      python3-pytest \
      python3-ruamel.yaml \
      python3-shapely && \
    rm -rf /var/lib/apt/lists/*

# numpy<2 and scipy<1.16 for Humble's apt Python stack. transforms3d from pip because
# jammy's 0.3.1 breaks on numpy >= 1.24. torch backs upstream's traversability filter.
RUN python3 -m pip install --no-cache-dir \
      numpy==1.26.4 \
      scipy==1.15.3 \
      transforms3d==0.4.2 \
      simple-parsing==0.1.9 \
      cupy-cuda12x==13.6.0 && \
    python3 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.9.1+cu128

# ros2_numpy is not released for Humble; build its humble branch at a pinned commit.
RUN git clone --filter=blob:none https://github.com/Box-Robotics/ros2_numpy.git ${AMENT_WS}/src/ros2_numpy && \
    git -C ${AMENT_WS}/src/ros2_numpy checkout -q d665add22e6a9070e88d9514bbf0dcbcb122a3dc && \
    rm -rf ${AMENT_WS}/src/ros2_numpy/.git

################################ Build ################################
FROM dependencies AS build

WORKDIR ${AMENT_WS}
COPY src/terrain_mapping src/terrain_mapping

RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
        --install-base ${WATONOMOUS_INSTALL} && \
    rm -rf build log install

# `watod -t` opens a plain bash, which skips the entrypoint, so source the install from bashrc too.
RUN printf '%s\n' \
      "source /opt/ros/${ROS_DISTRO}/setup.bash" \
      "source ${WATONOMOUS_INSTALL}setup.bash" \
      "[ -f ${AMENT_WS}/install/setup.bash ] && source ${AMENT_WS}/install/setup.bash" \
      >> /etc/bash.bashrc

COPY docker/terrain_mapping/fastdds_shm.xml /opt/fastdds_shm.xml
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/opt/fastdds_shm.xml

COPY docker/wato_ros_entrypoint.sh ${AMENT_WS}/wato_ros_entrypoint.sh
ENTRYPOINT ["./wato_ros_entrypoint.sh"]

################################ Develop ################################
# Run as the host user so bind-mounted files aren't root-owned.
FROM build AS develop
ARG USER_UID=1000
ARG USER_GID=1000
ARG USERNAME=dev
RUN old=$(getent passwd "${USER_UID}" | cut -d: -f1 || true); \
    if [ -n "$old" ] && [ "$old" != "${USERNAME}" ]; then \
        groupmod -n "${USERNAME}" "$(getent group "${USER_GID}" | cut -d: -f1)" 2>/dev/null || true; \
        usermod  -l "${USERNAME}" -d "/home/${USERNAME}" "$old"; \
        mkdir -p "/home/${USERNAME}" && cp -rT /etc/skel "/home/${USERNAME}"; \
    fi; \
    id -u "${USERNAME}" >/dev/null 2>&1 || { \
        getent group "${USER_GID}" >/dev/null || groupadd --gid "${USER_GID}" "${USERNAME}"; \
        useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}" --shell /bin/bash; }; \
    apt-get update && apt-get install -y --no-install-recommends sudo; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"; \
    chown -R "${USER_UID}:${USER_GID}" "${AMENT_WS}" "/home/${USERNAME}"; \
    rm -rf /var/lib/apt/lists/*
USER ${USERNAME}
WORKDIR ${AMENT_WS}
