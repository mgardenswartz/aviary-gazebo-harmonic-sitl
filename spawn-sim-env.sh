#!/bin/bash
# Note - This script is intended to be run outside of the docker container
source spawn-locations.env

echo "Number of turtlebots to spawn: $N_TB"
echo "Number of quads to spawn: $N_QUAD"
echo "TB spawn locations:"
for i in $(seq 1 $N_TB); do
    echo "TB_"$i": ${TB_SPAWN_LOCATIONS[i-1]}"
done
echo "Quad spawn locations:"
for i in $(seq 1 $N_QUAD); do
    echo "QUAD_"$i": ${QUAD_SPAWN_LOCATIONS[i-1]}"
done

# Run gazebo (tb)
use_simulator="True"
vehicle_name="tb"
x_pos=0.0
y_pos=0.0
z_pos=0.0
yaw_offset=1.5708
SENTINEL_VISION_PATH="/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models/sentinel_vision/model.sdf"

# make sure the container is running before we attempt to connect
if [ ! "$(sudo docker ps -a | grep "$CONTAINER_NAME")" ]; then
	echo "Warning: container "$CONTAINER_NAME" is not running. Gazebo set up failed"
else
	echo "Found container "$CONTAINER_NAME"."
    echo "Preparing to run simulation"
    
    # need to sleep longer after first spinning up gazebo before we attempt to load in all the other models
    sleep_time=10
    # Start gazebo and load in all of the turtlebots
    for i in $(seq 1 $N_TB); do
        #if this is the first turtlebot to spawn we need to start gazebo
        if [ "$i" -lt 2 ]; then
            echo "Starting gazebo and spawning tb1"
        else
            use_simulator="False"
            sleep_time=3
            echo "Spawning tb"$i
        fi
        IFS="," read -r -a spawn_position <<< ${TB_SPAWN_LOCATIONS[i-1]}
        x_pos=${spawn_position[0]}
        y_pos=${spawn_position[1]}
        z_pos=${spawn_position[2]}
        if (( $(echo "$z_pos < 0" | bc -l) )) || (( $(echo "$z_pos > 0" | bc -l) )); then
            z_pos=$(echo "$z_pos * -1" | bc -l)
        fi
        echo "tb"$i" spawn position = ("$x_pos","$y_pos","$z_pos")"
        # only log the first 100 lines of output so we don't end up with massive log files
        sudo docker exec -d $CONTAINER_NAME bash -c "source /home/root/ros-sources.sh; ros2 launch nav2_minimal_tb4_sim simulation.launch.py namespace:=$vehicle_name$i robot_name:=$vehicle_name$i use_rviz:=False use_simulator:=$use_simulator x_pose:=$y_pos y_pose:=$x_pos z_pose:=$z_pos yaw:=$yaw_offset 2>&1 | tee >(head -n 200 > /tmp/$vehicle_name$i-output.log) > /dev/null"
        # give gazebo time to start up/load models before we load another
        sleep $sleep_time
        sudo docker exec -it $CONTAINER_NAME bash -c "cat /tmp/$vehicle_name$i-output.log"
    done

    # if we're not using the turtlebot simulator package we'll need to start gazebo separately
    if [ "$N_TB" -lt 1 ]; then
        ./background-scripts/background-run-gz.sh
    fi

    # Now load in the quad models
    vehicle_name="px4_"
    sleep_time=3
    for i in $(seq 1 $N_QUAD); do
        IFS="," read -r -a spawn_position <<< ${QUAD_SPAWN_LOCATIONS[i-1]}
        x_pos=${spawn_position[0]}
        y_pos=${spawn_position[1]}
        z_pos=${spawn_position[2]}
        if (( $(echo "$z_pos < 0" | bc -l) )) || (( $(echo "$z_pos > 0" | bc -l) )); then
            z_pos=$(echo "$z_pos * -1" | bc -l)
        fi
        echo "px4_"$i" spawn position = ("$x_pos","$y_pos","$z_pos")"
        echo "Spawning sentinel vision model px4_"$i
        # only log the first 100 lines of output so we don't end up with massive log files
        sudo docker exec -it $CONTAINER_NAME bash -c "source /home/root/ros-sources.sh; ros2 run ros_gz_sim create -file $SENTINEL_VISION_PATH -name $vehicle_name$i -allow_renaming true -x $y_pos -y $x_pos -z $z_pos -Y $yaw_offset"
        sleep $sleep_time
    done

    # Now start px4_sitl instances for each quad loaded, give px4 a little longer to load
    sleep_time=10
    for i in $(seq 1 $N_QUAD); do
        echo "Starting px4_sitl instance for sentinel vision model px4_"$i
        # don't log px4 output, creates too large of files as blinking cursor is read as output for some reason
        sudo docker exec -d $CONTAINER_NAME bash -c "source /home/root/ros-sources.sh; PX4_SYS_AUTOSTART=4101 PX4_GZ_MODEL_NAME=$vehicle_name$i PX4_GZ_STANDALONE=1 /home/root/voxl-px4/px4-firmware/build/px4_sitl_default/bin/px4 -i $i >/dev/null 2>&1"
        # give gazebo time to start up/load models before we load another
        sleep $sleep_time
    done

    # Now start MicroXRCEAgent to bridge px4 topics to the ros domain
    # echo "Current Directory: "
    # pwd
    # only start if we're spawning a quad
    if [ "$N_QUAD" -gt 0 ]; then
        ./background-scripts/background-start-xrce-agent.sh
    fi
    

    # Optional - start ros-gz param bridge for quad odom topic
fi

# /opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
