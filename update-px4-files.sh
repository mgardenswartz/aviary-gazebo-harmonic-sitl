#!/bin/bash

PX4_DIRECTORY_NAME="voxl-px4/px4-firmware"
PX4_DIRECTORY="/home/root/"$PX4_DIRECTORY_NAME"/"

cd /home/root/

if [ ! -d "$PX4_DIRECTORY" ]; then
    echo "Warning - Directory $PX4_DIRECTORY does not exist."
    echo "Make sure that voxl-px4 is cloned, the submodules are updated and it is installed in the correct location before running this script"
else
    echo "Found px4-firmware.  Updating files..."
    #####################################################################
    # Apply Bug fixes

    # board_common.h
    echo "Overwriting px4-firmware/platforms/common/include/px4_platform_common/board_common.h"

    cp /home/root/px4-updates/sitl-build-fix/board_common.h /home/root/voxl-px4/px4-firmware/platforms/common/include/px4_platform_common/

    # main.cpp
    echo "Overwriting px4-firmware/platforms/posix/src/px4/common/main.cpp"

    cp /home/root/px4-updates/sitl-build-fix/main.cpp /home/root/voxl-px4/px4-firmware/platforms/posix/src/px4/common/

    # LoadMon.cpp
    echo "Overwriting px4-firmware/src/modules/load_mon/LoadMon.cpp"
    
    cp /home/root/px4-updates/sitl-build-fix/LoadMon.cpp /home/root/voxl-px4/px4-firmware/src/modules/load_mon/

    #GZBridge.cpp
    echo "Overwriting px4-firmware/src/modules/simulation/gz_bridge/GZBridge.cpp"
    
    cp /home/root/px4-updates/sitl-build-fix/GZBridge.cpp /home/root/voxl-px4/px4-firmware/src/modules/simulation/gz_bridge/
    #####################################################################
    # Upload models 
    echo "Copying updated model files to px4-firmware/Tools/simulation/gz/"

    cp -r /home/root/px4-updates/models /home/root/voxl-px4/px4-firmware/Tools/simulation/gz/

    #####################################################################
    # Startup scripts
    echo "Copying updated startup scripts to px4-firmware/ROMFS/px4fmu_common/init.d-posix"

    cp -r /home/root/px4-updates/init.d-posix /home/root/voxl-px4/px4-firmware/ROMFS/px4fmu_common/

    #####################################################################
    # dds topics 
    echo "Overwriting px4-firmware/src/modules/microdds_client/dds_topics.yaml"

    cp /home/root/px4-updates/dds-client/dds_topics.yaml /home/root/voxl-px4/px4-firmware/src/modules/microdds_client/

    #####################################################################
    # Gazebo garden to harmonic
    echo "Overwriting /home/root/voxl-px4/px4-firmware/src/modules/simulation/gz_bridge/CMakeLists.txt"

    cp /home/root/px4-updates/gz-garden-to-harmonic/gz_bridge/CMakeLists.txt /home/root/voxl-px4/px4-firmware/src/modules/simulation/gz_bridge/

    echo "Overwriting /home/root/voxl-px4/px4-firmware/src/modules/simulation/gz_bridge/GZMixingInterfaceESC.hpp"

    cp /home/root/px4-updates/gz-garden-to-harmonic/gz_bridge/GZMixingInterfaceESC.hpp /home/root/voxl-px4/px4-firmware/src/modules/simulation/gz_bridge/

    # Done!
    echo "Finished updating PX4 files, ready to build"


fi