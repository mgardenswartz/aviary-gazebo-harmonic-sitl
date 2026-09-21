#!/bin/bash 

WORLD_FILE="depot.sdf"

# export path to gazebo world/model files
export GZ_SIM_RESOURCE_PATH="/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models":"/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/worlds"
export GZ_VERSION=harmonic

# depot.sdf references filename="LocalizedWindPlugin" (aviary::LocalizedWindPlugin).
# This launch path does NOT source ros-sources.sh, so export the plugin search path here
# too or gz silently loads the world with no wind. Built by:
#   cd ros2_ws && colcon build --packages-select aviary_wind_plugin
export GZ_SIM_SYSTEM_PLUGIN_PATH="/home/root/ros2_ws/install/aviary_wind_plugin/lib${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"

# start up the gazebo sim environment
# gz sim -r -s -v 4 $WORLD_FILE
gz sim -r -v 4 $WORLD_FILE
