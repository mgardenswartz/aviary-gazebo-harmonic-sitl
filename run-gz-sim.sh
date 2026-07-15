#!/bin/bash 

WORLD_FILE="max-custom-aviary.sdf"

# export path to gazebo world/model files
export GZ_SIM_RESOURCE_PATH=export GZ_SIM_RESOURCE_PATH="/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models":"/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/worlds"
export GZ_VERSION=harmonic

# start up the gazebo sim environment
gz sim -r -s -v 4 $WORLD_FILE
# gz sim -r -v 4 $WORLD_FILE
