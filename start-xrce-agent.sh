#!/bin/bash
# source setup file so we get the right domain id
source /home/root/ros-sources.sh
# need to set transport method to ipv4 udp and discovery port to 8888 (same used by client)
MicroXRCEAgent udp4 -p 8888
