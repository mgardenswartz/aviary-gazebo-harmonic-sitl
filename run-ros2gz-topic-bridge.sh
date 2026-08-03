#!/bin/bash
# Note - this will bridge gazebo and ros topics identified in the yaml file $CONFIG_FILE
# This script uses the gz_odom.yaml file to bridge gazebo odom to ros by default
# To bridge different topics between gazebo and ros, write a new yaml file with the topic information and set CONFIG_FILE=<new_file_name>.yaml
source /home/root/ros-sources.sh
CONFIG_FILE=gz_odom.yaml
CONFIG_PATH="/home/root/"$CONFIG_FILE
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=$CONFIG_PATH