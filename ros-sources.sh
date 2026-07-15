#!/bin/bash
####################################################################################
# This is just meant to consolidate calls to sourcing ros setup files and exporting
# environment variables. Include "source /home/root/ros-sources.sh" inside bash 
# scripts meant to run inside the container so you can apply updates to a single file
# (e.g. when changing the values of ROS_DOMAIN_ID, ROS_DISCOVERY_SERVER etc...)
####################################################################################
source /opt/ros/humble/setup.bash
source /home/root/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=10
# export GZ_SIM_RESOURCE_PATH="/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models":"/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/worlds"
export GZ_VERSION=harmonic
SENTINEL_VISION_PATH="/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models/sentinel_vision/model.sdf"
