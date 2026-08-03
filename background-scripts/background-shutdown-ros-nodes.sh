#!/bin/bash
ROS_PIDS=$(pgrep -f "/opt/ros/humble" | grep -v "pgrep")
if [ -z "$ROS_PIDS" ]; then 
    echo "No ros processes running"
else
    echo "Found following PIDs for ros processes: $ROS_PIDS"
    for PID in $ROS_PIDS; do 
        echo "Terminating process $PID..."
        kill "$PID"
        
        # Max Gardenswartz added
        sleep 1
        if ps -p "$PID" > /dev/null; then
            echo "ROS process $PID refused to die. Sending SIGKILL..."
            kill -9 "$PID"
        fi
    done
    echo "All ros processes have been terminated."
fi 
source /opt/ros/humble/setup.bash
ros2 daemon stop
