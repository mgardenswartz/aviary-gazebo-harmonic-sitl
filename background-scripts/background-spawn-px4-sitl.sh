#!/bin/bash
# Note - This script is intended to be run outside of the docker container
CONTAINER_NAME=px4-sitl-gz 
# Prompt user for command line arguments needed to spawn vehicle
echo
echo "Preparing to start px4_sitl in standalone mode"
echo "Note - You must start gazebo, open a world and spawn model before running this script."
read -p "Press enter when the gazebo world and model are ready"
read -p "Enter the model name to attach this sitl instance to (e.g. px4_1, px4_2 ...): " vehicle_name
read -p "Enter the instance number (should match vehicle id, e.g. i=1 for px4_1, i=2 for px4_2): " instance_number
echo "Preparing to run px4_sitl in standalone mode for model $vehicle_name"

# make sure the container is running before we attempt to connect
if [ ! "$(sudo docker ps -a | grep "$CONTAINER_NAME")" ]; then
	echo "Warning: container "$CONTAINER_NAME" is not running. PX4-SITL set up failed"
else
	echo "Found container "$CONTAINER_NAME"."
    echo "Preparing to run px4_sitl in standalone mode for model $vehicle_name"
    echo 
    # only log the first 100 lines of output so we don't end up with massive log files
	sudo docker exec -d $CONTAINER_NAME bash -c "source /home/root/ros-sources.sh; cd /home/root/voxl-px4/px4-firmware; PX4_SYS_AUTOSTART=4101 PX4_GZ_MODEL_NAME=$vehicle_name PX4_GZ_STANDALONE=1 ./build/px4_sitl_default/bin/px4 -i $instance_number >/dev/null 2>&1"
	sleep 3
fi
