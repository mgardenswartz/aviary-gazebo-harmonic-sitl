#!/bin/bash
GZ_CMD="gz sim"
GZ_PIDS=$(pgrep -f "$GZ_CMD" | grep -v "pgrep")
if [ -z "$GZ_PIDS" ]; then 
    echo "Gazebo is not running"
else
    echo "Found following PIDs for command 'gz sim': $GZ_PIDS"
    for PID in $GZ_PIDS; do 
        echo "Terminating process $PID..."
        kill "$PID"

        # Max Gardenswartz added:
        sleep 1
        if ps -p "$PID" > /dev/null; then
            echo "Gazebo process $PID refused to die. Sending SIGKILL..."
            kill -9 "$PID"
        fi

    done
    echo "Gazebo has been terminated."
fi 

RSP_CMD="robot_state_publisher"
RSP_PIDS=$(pgrep -f "$RSP_CMD" | grep -v "pgrep")
if [ -z "$RSP_PIDS" ]; then 
    echo "Robot state publisher is not running"
else
    echo "Found following PIDS for command $RSP_CMD: $RSP_PIDS"
    for PID in $RSP_PIDS; do
        echo "Terminating process $PID"
        kill "$PID"

        # Max Gardenswartz added:
        sleep 1
        if ps -p "$PID" > /dev/null; then
            echo "Robot state publisher process $PID refused to die. Sending SIGKILL..."
            kill -9 "$PID"
        fi
    done
fi
