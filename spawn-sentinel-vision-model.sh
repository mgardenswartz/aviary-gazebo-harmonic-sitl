#!/bin/bash

# Make sure gazebo server is running before trying to spawn
GZPID=$(pgrep -f "gz sim")
if [[ $GZPID ]]; then
    echo "Found running gazebo server. Checking for active worlds..."
    WORLD_TOPIC=$(gz topic -l | grep -E "/world/.+/pose/info" | head -n 1)

    if [ -n "$WORLD_TOPIC" ]; then
        # Extract the world name from the topic string (e.g., from "/world/default/pose/info" get "default")
        WORLD_NAME=$(echo "$WORLD_TOPIC" | cut -d'/' -f3)
        echo "Running Gazebo world: $WORLD_NAME"
    else
        echo "Server is running, but could not find an active world. Please open a world in gazebo"
        exit 1
    fi
else
    echo "Error - Gazebo server is not running. Please start gazebo before attempting to spawn a vehicle"
    exit 1
fi

# Prompt user for command line arguments needed to spawn vehicle
echo
echo "Preparing to spawn sentinel_vision model"
echo 
echo "Vehicle name must match px4_sitl naming convention (px4_<instance_number>)"
read -p "Please enter a vehicle name (e.g. px4_1, px4_2 ...): " vehicle_name
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

# spawn vehicle w/ user args
source /home/root/ros-sources.sh
# Note - need to swap y and x, sign flip z to go from ENU (gazebo) to NED
if (( $(echo "$z_pos < 0" | bc -l) )) || (( $(echo "$z_pos > 0" | bc -l) )); then
    z_pos=$(echo "$z_pos * -1" | bc -l)
fi
echo "Spawning vehicle $vehicle_name in $WORLD_NAME world at ($y_pos,$x_pos,$z_pos) ENU"
ros2 run ros_gz_sim create -file $SENTINEL_VISION_PATH -name $vehicle_name -allow_renaming true -x $y_pos -y $x_pos -z $z_pos -Y $yaw_offset

# hardcoded example
# ros2 run ros_gz_sim create -file "/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models/sentinel_vision/model.sdf" -name px4_1 -allow_renaming true -x 2 -y 2 -z 0 

# ros2 run ros_gz_sim create -file "/home/root/turtlebot4_ws/install/nav2_minimal_tb4_description/share/nav2_minimal_tb4_description/urdf/standard/turtlebot4.urdf.xacro" 