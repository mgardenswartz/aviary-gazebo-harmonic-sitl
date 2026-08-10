#!/bin/bash
# Note - This script is intended to be run outside of the docker container
# Usage: background-run-vision-odometry-noise.sh [NOISE_AND_DELAYS_ON]  (default: true)
CONTAINER_NAME=px4-sitl-gz
NOISE_AND_DELAYS_ON="${1:-true}"

# make sure the container is running before we attempt to connect
if [ ! "$(sudo docker ps -a | grep "$CONTAINER_NAME")" ]; then
	echo "Warning: container "$CONTAINER_NAME" is not running. vision_odometry_noise node set up failed"
else
	echo "Found container "$CONTAINER_NAME"."
    echo "Starting vision_odometry_noise_node (NOISE_AND_DELAYS_ON=$NOISE_AND_DELAYS_ON)"
    echo
    # Unlike the other background-run-*.sh scripts, this one does NOT cap the log at 200
    # lines -- this node prints ongoing [DIAG] latency/rate lines every 2s for the life of the
    # run, and a truncated log would silently stop capturing them a few minutes in. Watch it
    # live with: sudo docker exec -it $CONTAINER_NAME bash -c "tail -f /tmp/vision-odometry-noise-output.log"
	sudo docker exec -d $CONTAINER_NAME bash -c "NOISE_AND_DELAYS_ON=$NOISE_AND_DELAYS_ON ./run-vision-odometry-noise-node.sh > /tmp/vision-odometry-noise-output.log 2>&1"
	sleep 3
	sudo docker exec -it $CONTAINER_NAME bash -c "cat /tmp/vision-odometry-noise-output.log"
fi
