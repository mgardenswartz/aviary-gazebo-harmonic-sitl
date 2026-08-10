#!/bin/bash
NOISE_NODE_CMD="vision_odometry_noise_node"
NOISE_NODE_PIDS=$(pgrep -f "$NOISE_NODE_CMD" | grep -v "pgrep")
if [ -z "$NOISE_NODE_PIDS" ]; then
    echo "vision_odometry_noise_node is not running"
else
    echo "Found following PIDs for command $NOISE_NODE_CMD: $NOISE_NODE_PIDS"
    for PID in $NOISE_NODE_PIDS; do
        echo "Terminating process $PID..."
        kill "$PID"

        sleep 1
        if ps -p "$PID" > /dev/null; then
            echo "Process $PID refused to die. Sending SIGKILL..."
            kill -9 "$PID"
        fi
    done
    echo "vision_odometry_noise_node has been terminated."
fi
