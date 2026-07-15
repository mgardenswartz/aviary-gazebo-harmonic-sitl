#!/bin/bash
# Note - This script is intended to be run outside of the docker container
CONTAINER_NAME=px4-sitl-gz 

# Prompt user for command line arguments needed to spawn vehicle
use_simulator="True"
echo
read -p "Enter 'True' to start the gazebo server (just press enter spawn models in a already running world): " standalone
if [[ -z "$standalone" ]]; then
    echo "Running in standalone mode"
    use_simulator="False"
else
    echo "Running in server mode"
fi
echo 
echo "If spawning multiple turtlebots in the same environment, use distinct vehicle names to avoid topic conflict"
read -p "Please enter a vehicle name (e.g. tb1, tb2, ...): " vehicle_name
echo "Setting vehicle name to $vehicle_name"
echo
echo "Please entered the desired spawn position in NED frame"
read -p "x [m]: " x_pos
read -p "y [m]: " y_pos 
read -p "z [m]: " z_pos 
if [[ -z "$x_pos" ]]; then
    echo "No x spawn position given. Setting to 0.0"
    x_pos=0.0
fi
if [[ -z "$y_pos" ]]; then
    echo "No y spawn position given. Setting to 0.0"
    y_pos=0.0
fi
if [[ -z "$z_pos" ]]; then
    echo "No z spawn position given. Setting to 0.0"
    z_pos=0.0
fi
echo "Setting vehicle spawn position to ($x_pos, $y_pos, $z_pos) NED"
echo
echo "Note - gazebo uses ENU frame and will spawn vehicles aligned with the East axis if no yaw offset is given"
echo "A default yaw offset of 1.5708 radians will be used to align the vehicle with the north axis if no user input is given"
read -p "Enter desired yaw offset (press enter to skip): " yaw_offset
if [[ -z "$yaw_offset" ]]; then
    yaw_offset=1.5708
    echo "Using default yaw offset = $yaw_offset"
else
    echo "Using yaw offset = $yaw_offset radians"
fi
if (( $(echo "$z_pos < 0" | bc -l) )) || (( $(echo "$z_pos > 0" | bc -l) )); then
    z_pos=$(echo "$z_pos * -1" | bc -l)
fi


# make sure the container is running before we attempt to connect
if [ ! "$(sudo docker ps -a | grep "$CONTAINER_NAME")" ]; then
	echo "Warning: container "$CONTAINER_NAME" is not running. Gazebo set up failed"
else
	echo "Found container "$CONTAINER_NAME"."
    echo "Preparing to run gazebo"
    echo 
    # only log the first 200 lines of output so we don't end up with massive log files
	sudo docker exec -d $CONTAINER_NAME bash -c "source /home/root/ros-sources.sh; ros2 launch nav2_minimal_tb4_sim simulation.launch.py namespace:=$vehicle_name robot_name:=$vehicle_name use_rviz:=False use_simulator:=$use_simulator x_pose:=$y_pos y_pose:=$x_pos z_pose:=$z_pos yaw:=$yaw_offset 2>&1 | tee >(head -n 200 > /tmp/tb-gz-sim-output.log) > /dev/null"
	sleep 5
	sudo docker exec -it $CONTAINER_NAME bash -c "cat /tmp/tb-gz-sim-output.log"
fi

