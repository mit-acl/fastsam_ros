#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/noetic/setup.bash"
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# source our ws
source /catkin_ws/devel/setup.bash
echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

exec "$@"