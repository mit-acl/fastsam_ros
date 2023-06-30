FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Parker Lusk <plusk@mit.edu>"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y git curl cmake build-essential wget gnupg2 lsb-release ca-certificates python3-pip
 # && apt-get clean && rm -rf /var/lib/apt/lists/*
 #ffmpeg

# install ROS Noetic
ARG ROS_DISTRO=noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-${ROS_DISTRO}-ros-base python3-rosdep python3-catkin-tools \
        ros-${ROS_DISTRO}-cv-bridge ros-${ROS_DISTRO}-vision-msgs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip && \
    python3 -m pip install torch torchvision torchaudio

RUN mkdir -p /catkin_ws/src && cd /catkin_ws && \
    catkin init && catkin config --extend /opt/ros/${ROS_DISTRO} && \
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release

COPY ./fastsam_ros/src/fastsam_ros/FastSAM/requirements.txt /
RUN python3 -m pip install -r /requirements.txt
RUN python3 -m pip install git+https://github.com/openai/CLIP.git

COPY . /catkin_ws/src/
RUN cd /catkin_ws/src && \
    catkin build

COPY ./ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
WORKDIR /catkin_ws