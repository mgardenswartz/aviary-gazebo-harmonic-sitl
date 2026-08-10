#!/bin/bash

# allow docker to use display to run guis
xhost +local:docker
SITL_DIRECTORY_NAME="voxl-px4-sitl"
SIM_DIRECTORY=$HOME"/"$SITL_DIRECTORY_NAME"/"
IMAGE_NAME=voxl-px4-sitl
IMAGE_TAG=sdk1.4-v1
CONTAINER_NAME=px4-sitl-gz 

if [ ! -d "$SIM_DIRECTORY" ]; then
    echo "Warning - Directory $SIM_DIRECTORY does not exist. Make sure cloned repository name matches "$SITL_DIRECTORY_NAME" as defined in this script and was cloned in the user's home directory "$HOME
else
    # HOME=/home/root points root's home at the bind-mounted (persistent) directory instead of
    # the container's ephemeral /root -- otherwise things like `git config --global` (e.g. the
    # safe.directory exception PX4's build needs) get silently wiped every time this --rm
    # container is recreated.
    sudo docker run --rm -it --net=host --ipc=host --pid=host --privileged -v /dev/shm:/dev/shm -e DISPLAY=$DISPLAY -e HOME=/home/root -v /dev/input:/dev/input:rw -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v $SIM_DIRECTORY:/home/root:rw -w /home/root --name=$CONTAINER_NAME $IMAGE_NAME:$IMAGE_TAG  /bin/bash -l

fi
