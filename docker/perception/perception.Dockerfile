ARG BASE_IMAGE=ghcr.io/watonomous/robot_base/base:humble-ubuntu22.04

################################ Source ################################
FROM ${BASE_IMAGE} AS source

WORKDIR ${AMENT_WS}/src

# Copy in source code 
COPY src/perception/perception perception
COPY src/common_msgs common_msgs

# Install rosdep if not present, update package lists
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-rosdep && \
    rm -rf /var/lib/apt/lists/*

# Update rosdep database (safe in containers)
RUN rosdep update

# Generate dependency list (simulated install → extract apt packages)
RUN rosdep install \
    --from-paths . \
    --ignore-src \
    --rosdistro $ROS_DISTRO \
    -y \
    --simulate | \
    grep "apt-get install" | \
    sed 's/apt-get install -y //' > /tmp/colcon_install_list || true

################################# Dependencies ################################
FROM ${BASE_IMAGE} AS dependencies

# Install Rosdep requirements
COPY --from=source /tmp/colcon_install_list /tmp/colcon_install_list

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros2-latest-archive-keyring.gpg
RUN apt-get update && \
    apt-fast install -qq -y --no-install-recommends $(cat /tmp/colcon_install_list)

# Dependency Cleanup
WORKDIR /
RUN apt-get -qq autoremove -y && apt-get -qq autoclean && apt-get -qq clean && \
    rm -rf /root/* /root/.ros /tmp/* /var/lib/apt/lists/* /usr/share/doc/*

# Essential build & Python tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      cmake \
      ninja-build \
      python3 \
      python3-pip \
      python3-dev \
      python3-setuptools \
      curl \
      ca-certificates \
      gnupg2 \
      libgl1-mesa-glx \
      lsb-release \
      libssl-dev \
      usbutils \
      libusb-1.0-0-dev \
      pkg-config \
      libgtk-3-dev \
      wget

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-$ROS_DISTRO-librealsense2* \
      ros-$ROS_DISTRO-realsense2-camera* && \
      rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir \
      pccm>=0.4.16 \
      ccimport>=0.4.4 \
      pybind11>=2.6.0 \
      cv_bridge \
      numpy==1.26.4 \
      fire \
      opencv-python

# Install PyTorch CPU
RUN pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install mmcv built for torch2.1.0
RUN pip install mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html

# Install mmpose and dependencies
RUN pip install --no-cache-dir mmpose==1.3.2 --no-deps && \
    pip install --no-cache-dir mmdet==3.3.0 --no-deps && \
    pip install --no-cache-dir \
      scipy \
      pycocotools \
      shapely \
      terminaltables \
      munkres \
      json_tricks \
      tqdm \
      setuptools==65.5.0

# Pin numpy to 1.x for cv_bridge compatibility
RUN pip install numpy==1.26.4 --force-reinstall

# Patch xtcocotools references to use pycocotools
RUN find /usr/local/lib/python3.10/dist-packages/mmpose/ \
    -name "*.py" \
    -exec sed -i \
    's/from xtcocotools/from pycocotools/g;s/import xtcocotools/import pycocotools/g' {} +

# Dependency Cleanup
WORKDIR /
RUN apt-get -qq autoremove -y && apt-get -qq autoclean && apt-get -qq clean && \
    rm -rf /root/.ros /tmp/* /var/lib/apt/lists/* /usr/share/doc/*

# Download RTMPose model
RUN mkdir -p /root/models && \
    wget -q https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth \
    -O /root/models/rtmpose-s.pth

# Foxglove visualization support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ros-humble-foxglove-bridge \
      ros-humble-rosbag2-storage-mcap && \
    rm -rf /var/lib/apt/lists/*

################################ Build ################################
FROM dependencies AS build
COPY --from=source ${AMENT_WS}/src ${AMENT_WS}/src

# Build ROS2 packages
WORKDIR ${AMENT_WS}

RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
        --install-base ${WATONOMOUS_INSTALL}

# Source and Build Artifact Cleanup 
RUN rm -rf build/* devel/* install/* log/*

# Entrypoint will run before any CMD on launch. Sources ~/opt/<ROS_DISTRO>/setup.bash and ~/ament_ws/install/setup.bash
COPY docker/wato_ros_entrypoint.sh ${AMENT_WS}/wato_ros_entrypoint.sh
ENTRYPOINT ["./wato_ros_entrypoint.sh"]



################################ Develop ################################
# Run as the host user so bind-mounted files aren't root-owned. The base image
# ships a `bolty` user at uid 1000; remap it to the host user (or make a new one).
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
