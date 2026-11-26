ARG BASE_IMAGE=ghcr.io/watonomous/robot_base/base:humble-ubuntu22.04

################################ Source ################################
FROM ${BASE_IMAGE} AS source

WORKDIR ${AMENT_WS}/src

COPY src/interfacing/joint_command joint_command
COPY src/common_msgs common_msgs

RUN apt-get -qq update && rosdep update && echo "" > /tmp/colcon_install_list

################################ Dependencies ################################
FROM ${BASE_IMAGE} AS dependencies

COPY --from=source /tmp/colcon_install_list /tmp/colcon_install_list
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libyaml-cpp-dev \
    $(cat /tmp/colcon_install_list) && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${AMENT_WS}
COPY --from=source ${AMENT_WS}/src src

################################ Build ################################
FROM dependencies AS build

WORKDIR ${AMENT_WS}
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --install-base ${WATONOMOUS_INSTALL}

RUN rm -rf src build log

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
