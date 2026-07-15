#!/bin/bash

echo
echo "Installing PX4 simulation dependencies"

# General simulation dependencies
sudo DEBIAN_FRONTEND=noninteractive apt-get -y --quiet --no-install-recommends install \
	bc \
	;

# Gazebo / Gazebo classic installation

# Expects Ubuntu 22.04 > by default
echo "[ubuntu.sh] Gazebo (Harmonic) will be installed"
echo "[ubuntu.sh] Earlier versions will be removed"
# Add Gazebo binary repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update -y --quiet
# Install Gazebo
gazebo_packages="gz-harmonic libunwind-dev"

sudo DEBIAN_FRONTEND=noninteractive apt-get -y --quiet --no-install-recommends install \
	dmidecode \
	$gazebo_packages \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-base \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-libav \
	libeigen3-dev \
	libgstreamer-plugins-base1.0-dev \
	libimage-exiftool-perl \
	libopencv-dev \
	libxml2-utils \
	pkg-config \
	protobuf-compiler \
	;

if sudo dmidecode -t system | grep -q "Manufacturer: VMware, Inc." ; then
	# fix VMWare 3D graphics acceleration for gazebo
	echo "export SVGA_VGPU10=0" >> ~/.profile
fi