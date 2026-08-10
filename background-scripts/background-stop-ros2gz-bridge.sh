#!/bin/bash
BRIDGE_CMD="parameter_bridge"
BRIDGE_PIDS=$(pgrep -f "$BRIDGE_CMD" | grep -v "pgrep")
if [ -z "$BRIDGE_PIDS" ]; then
    echo "ROS<->Gazebo topic bridge is not running"
else
    echo "Found following PIDs for command $BRIDGE_CMD: $BRIDGE_PIDS"
    for PID in $BRIDGE_PIDS; do
        echo "Terminating process $PID..."
        kill "$PID"

        sleep 1
        if ps -p "$PID" > /dev/null; then
            echo "Process $PID refused to die. Sending SIGKILL..."
            kill -9 "$PID"
        fi
    done
    echo "ROS<->Gazebo topic bridge has been terminated."
fi
