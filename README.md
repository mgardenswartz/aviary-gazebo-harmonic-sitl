# Instructions

Author: Joe Lullo

> [!'NOTE']
> These instructions require a Docker container with Ubuntu 22.04 and ROS 2 Humble installed.
> These instructions include scripts that run outside Docker that are Linux only. Mac and Windows are not supported (unless you want to write the bash scripts over again).

## Building the sim environment

1. Copy "voxl-px4-sitl" folder to home directory (required for setup scripts to work)

2. If you don't have a ros2 docker image to start with, install it with the following command:

```
sudo docker pull osrf/ros:humble-desktop
```

If using an existing ros2 docker image as your base image, update "run-ros2-container.sh" script as follows:

```
sudo docker run --rm -it --net=host --ipc=host --pid=host --privileged -v /dev/shm:/dev/shm -e DISPLAY=$DISPLAY -v /dev/input:/dev/input:rw -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v $SIM_DIRECTORY:/home/root:rw -w /home/root --name=ros2humble <the name of your ros2 docker image> /bin/bash -l
```

3. Open a terminal, navigate to the voxl-px4-sitl directory, make shell scripts executable and open a container with the ros2 base image

```
cd ~/voxl-px4-sitl
chmod +x background-scripts/*
chmod +x run*
chmod +x spawn*
chmod +x shutdown*
chmod +x start*
chmod +x update*
chmod +x install*
chmod +x connect*    
./run-ros2-container.sh
```

4. From within the docker container, clone modalai's fork of PX4 (same firmware running on Sentinel vehicles) and update the submodules

```
git clone https://gitlab.com/voxl-public/voxl-sdk/services/voxl-px4.git -b sdk-1.4.0
cd voxl-px4
git submodule update --init --recursive
```

5. Install PX4 dependencies (without installing simulation tools)

```
cd px4-firmware
./Tools/setup/ubuntu.sh --no-sim-tools
```

6. Return to the mounted home directory and run the helper script to install Gazebo Harmonic separately

```
cd /home/root
./install_gz_harmonic.sh
```

7. Clone and build the appropriate DDS-Agent firmware (used to bridge PX4 UORB topics to ROS domain), this will take a few minutes

```
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```

8. Return to the mounted home directory and run the helper script to apply bug-fixes to px4 firmware

```
cd /home/root
./update-px4-files.sh
```

9. Navigate to the px4 firmware and build the SITL binaries.  If the build process was performed successfully, this will open a Gazebo instance and spawn a model tied to a px4_sitl instance.  Exit w/ CTRL+C to close the simulation

```
cd /home/root/voxl-px4/px4-firmware 
make px4_sitl gz_sentinel_vision
```

10. Build the px4_msgs ROS package required to write ROS packages that can interface with px4_sitl instances

```
source /opt/ros/humble/setup.bash
cd /home/root/ros2_ws
colcon build --packages-select px4_msgs --symlink-install
```

11. Install ros packages for interfacing with gazebo

```
apt install ros-humble-ros-gzharmonic
```

12. Install dependencies and build ros minimal turtlebot simulation package 

```
apt-get install ros-humble-joint-state-publisher
cd /home/root/tb_sim_custom
colcon build --symlink-install
```

*****************
### Optional
Before committing the new image, it can be helpful to add the following line to the end of the .bashrc file:

```
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
```

This enables tab completion when building ros packages.  To edit files within the docker container that aren't mounted from your pc's filesystem, install vim (apt install vim) or another command line text editor and edit the file (~/.bashrc) from the terminal running the container
*****************

13. The build is now complete.  Commit the running docker container to a new image.  Do not exit from the docker container.  Open a new terminal and commit the new docker image

```
sudo docker commit ros2humble voxl-px4-sitl:sdk1.4-v1
```

14. To confirm the image was successfully committed, navigate to the voxl-px4-sitl directory and use the helper script to open a container with the new image

```
cd ~/voxl-px4-sitl
./run-px4-sim-container.sh
```

If the new image was successfully committed, exit both containers.

## Using the sim environment

1. Open a terminal, navigate to the sim environment directory and run the docker container

```
cd ~/voxl-px4-sitl
./run-px4-sim-container.sh
```

2. Edit the file ~/voxl-px4-sitl/spawn-locations.env to set the number of each vehicle type you'd like to spawn within the sim environment and their spawn locations

3. Open another terminal, navigate to the sim environment directory and run the bringup script for the sim environment.  This will start a gazebo instance, spawn the number of models selected, run a px4_sitl instance for each spawned quad model, and start the DDS agent to bridge PX4 UORB topics to ROS

```
cd ~/voxl-px4-sitl
./spawn-sim-env.sh
```

4. From the terminal that is running the docker container, source the ros-sources file and run a ros package to interact with the models in gazebo

```
source ros-sources.sh
ros2 launch your_package your_launch_file.launch.py
```