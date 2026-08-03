#!/bin/bash
# Note - This script is intended to be run outside of the docker container
CONTAINER_NAME=px4-sitl-gz 

# make sure the container is running before we attempt to connect
if [ ! "$(sudo docker ps -a | grep "$CONTAINER_NAME")" ]; then
	echo "Warning: container "$CONTAINER_NAME" is not running. MicroXRCEAgent set up failed"
else
	echo "Found container "$CONTAINER_NAME"."
    echo "Starting MicroXRCEAgent"
    echo 
    # only log the first 200 lines of output so we don't end up with massive log files
	sudo docker exec -d $CONTAINER_NAME bash -c "source /home/root/ros-sources.sh; MicroXRCEAgent udp4 -p 8888 2>&1 | tee >(head -n 200 > /tmp/xrce-output.log) > /dev/null "
	sleep 10
	sudo docker exec -it $CONTAINER_NAME bash -c "cat /tmp/xrce-output.log"
fi
