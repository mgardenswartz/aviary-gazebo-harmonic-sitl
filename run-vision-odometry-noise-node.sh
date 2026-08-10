#!/bin/bash
# Note - this runs the vision_odometry_noise ROS 2 node, which reads Gazebo ground-truth
# odometry (bridged in by run-ros2gz-topic-bridge.sh) and publishes vehicle_visual_odometry
# to PX4 over the uXRCE-DDS agent. In this vision-only SITL config it's PX4's only source of
# position/velocity aiding, so it must be running before EKF2 will pass its arming checks.
#
# NOISE_AND_DELAYS_ON (env var, default true) is the same ablation flag read by
# 4100_gz_sentinel for EKF2_EVV_NOISE/EKF2_EV_DELAY -- true: realistic noisy/jittered vision.
# false: true ground truth, published synchronously with zero added noise/latency.
source /home/root/ros-sources.sh
NOISE_AND_DELAYS_ON=${NOISE_AND_DELAYS_ON:-true}
# Python fully-buffers stdout by default when it isn't a tty (i.e. whenever this is piped to
# a log file, as background-run-vision-odometry-noise.sh does) -- without this, the [DIAG]
# lines only show up in bursts whenever the buffer happens to fill, not every 2s as printed.
export PYTHONUNBUFFERED=1
ros2 run vision_odometry_noise vision_odometry_noise_node --ros-args \
	--params-file /home/root/ros2_ws/install/vision_odometry_noise/share/vision_odometry_noise/params/vision_odometry_noise.yaml \
	-p noise_and_delays_enabled:=$NOISE_AND_DELAYS_ON
