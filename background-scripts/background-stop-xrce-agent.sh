#!/bin/bash
XRCE_CMD="MicroXRCEAgent"
XRCE_PID=$(pgrep -f "$XRCE_CMD" | grep -v "pgrep")
if [ -z "$XRCE_PID" ]; then 
    echo "MicroXRCEAgent is not running "
else
    echo "Found following PIDs for command $XRCE_CMD: $XRCE_PID"
    for PID in $XRCE_PID; do 
        echo "Terminating process $PID..."
        kill "$PID"

        # Max Gardenswartz added:
        sleep 1
        if ps -p "$PID" > /dev/null; then
            echo "Process $PID refused to die. Sending SIGKILL..."
            kill -9 "$PID"
        fi
    done
    echo "MicroXRCEAgent has been terminated."
fi 
