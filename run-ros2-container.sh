#!/bin/bash

# allow docker to use display to run guis
xhost +local:docker
SITL_DIRECTORY_NAME="voxl-px4-sitl"
SIM_DIRECTORY=$HOME"/"$SITL_DIRECTORY_NAME"/"

if [ ! -d "$SIM_DIRECTORY" ]; then
    echo "Warning - Directory $SIM_DIRECTORY does not exist. Make sure cloned repository name matches "$SITL_DIRECTORY_NAME" as defined in this script and was cloned in the user's home directory "$HOME
else
    sudo docker run --rm -it --net=host --ipc=host --pid=host --privileged -v /dev/shm:/dev/shm -e DISPLAY=$DISPLAY -v /dev/input:/dev/input:rw -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v $SIM_DIRECTORY:/home/root:rw -w /home/root --name=ros2humble osrf/ros:humble-desktop /bin/bash -l

fi