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

# Slung-load ablation (see spawn-locations.env for SLUNG_MASS_KG). Unlike NOISE_AND_DELAYS_ON,
# this needs an actual file regenerated (model.sdf has the mass/inertia baked in as numbers, not
# read at runtime) and redeployed into the container's live SITL tree before it can be spawned --
# both done here automatically so editing SLUNG_MASS_KG in spawn-locations.env is still the only
# thing you touch between runs, same as the noise tests.
SLUNG_MASS_KG="${SLUNG_MASS_KG:-0}"
if (( $(echo "$SLUNG_MASS_KG > 0" | bc -l) )); then
    SENTINEL_VISION_PATH="/home/root/voxl-px4/px4-firmware/Tools/simulation/gz/models/sentinel_vision_slung/model.sdf"
    echo "SLUNG_MASS_KG=$SLUNG_MASS_KG -- generating and deploying sentinel_vision_slung model.sdf..."
    python3 generate-slung-load-model.py "$SLUNG_MASS_KG"
    # Targeted equivalent of update-px4-files.sh's "Upload models" step only -- deliberately not
    # invoking the full script here, since that also touches firmware source/ROMFS files that
    # this ablation has nothing to do with and might stomp on other in-progress edits.
    sudo docker exec $CONTAINER_NAME bash -c "cp -r /home/root/px4-updates/models /home/root/voxl-px4/px4-firmware/Tools/simulation/gz/"
    IFS="," read -r -a quad1_check <<< "${QUAD_SPAWN_LOCATIONS[0]}"
    quad1_z_ned="${quad1_check[2]}"
    if (( $(echo "$quad1_z_ned >= -0.3" | bc -l) )); then
        echo "NOTE: QUAD1_LOCATION z=$quad1_z_ned (NED) means the payload will render below the" \
             "ground plane at the bottom of its swing (rod+payload hang ~0.72m below base_link," \
             "base_link only clears ~0.02m of ground at spawn) -- purely visual, not a physics" \
             "problem, since neither the rod nor the payload has collision geometry. Only bump" \
             "QUAD1_LOCATION's z (e.g. -0.8, NED so negative = up) if that clipping bothers you" \
             "in the GUI."
    fi
fi

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

    # Start the ROS<->Gazebo ground-truth odometry bridge (gz_odom.yaml) now that the quad
    # model(s) exist and are publishing /model/<name>/odometry. Needed by
    # vision_odometry_noise_node below. NOTE: gz_odom.yaml only bridges px4_1 by default --
    # uncomment the px4_2..px4_5 blocks there (and extend the loop below) for N_QUAD > 1.
    if [ "$N_QUAD" -gt 0 ]; then
        ./background-scripts/background-run-ros2gz-bridge.sh
    fi

    # Now start px4_sitl instances for each quad loaded, give px4 a little longer to load
    sleep_time=10
    for i in $(seq 1 $N_QUAD); do
        echo "Starting px4_sitl instance for sentinel vision model px4_"$i
        # don't log px4 output, creates too large of files as blinking cursor is read as output for some reason
        sudo docker exec -d $CONTAINER_NAME bash -c "source /home/root/ros-sources.sh; PX4_SYS_AUTOSTART=4101 PX4_GZ_MODEL_NAME=$vehicle_name$i PX4_GZ_STANDALONE=1 NOISE_AND_DELAYS_ON=$NOISE_AND_DELAYS_ON /home/root/voxl-px4/px4-firmware/build/px4_sitl_default/bin/px4 -i $i >/dev/null 2>&1"
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

    # Now start vision_odometry_noise_node so PX4's EKF2 has vision aiding to fuse -- without
    # this running, the vision-only airframe config (4101_gz_sentinel) never clears its
    # preflight EKF2 checks. NOTE: single-vehicle only for now (defaults to px4_1, matching
    # QUAD_SPAWN_LOCATIONS[0]) -- extend to a loop alongside the bridge above for N_QUAD > 1.
    if [ "$N_QUAD" -gt 0 ]; then
        ./background-scripts/background-run-vision-odometry-noise.sh "$NOISE_AND_DELAYS_ON"
    fi
fi

# /opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin
