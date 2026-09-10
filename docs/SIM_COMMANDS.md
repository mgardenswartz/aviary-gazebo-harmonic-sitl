
BUILDING PX4 (inside Docker)
bash /home/root/update-px4-files.sh 
cd /home/root/voxl-px4/px4-firmware 
git config --global --add safe.directory '*'
make px4_sitl gz_sentinel_vision

HOST PYTHON
pyenv local 3.13.14
python3 -m venv venv_host
source venv_host/bin/activate
pip install -e .
pip install --upgrade pip

MAKING PARAMS (outside Docker)
GAZEBO=true
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type pid --out ros2_ws/src/aviary_rise_controller/param/pid_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type integrated_resnet --out ros2_ws/src/aviary_rise_controller/param/integrated_resnet_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type resnet --out ros2_ws/src/aviary_rise_controller/param/resnet_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type baseline --out ros2_ws/src/aviary_rise_controller/param/baseline_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1

RUNNING THE SIM 
Terminal 1 (outside docker)
./shutdown-background-services.sh && ./spawn-sim-env.sh 

Terminal 2 (inside Docker)
cd /home/root
apt install python3.10-venv -y
git clone --branch v1.2.1 https://github.com/mgardenswartz/resnet.git
python3 -m venv venv_docker --system-site-packages
source venv_docker/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install "setuptools==58.2.0"
python3 -m pip install numpy jax pandas ./resnet

cd /home/root/ros2_ws
colcon build --symlink-install --packages-select debugging_node vision_odometry_noise
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/home/root/venv_docker/bin/python3 --packages-select aviary_rise_controller
sed -i '1s|^.*$|#!/home/root/venv_docker/bin/python3|' install/aviary_rise_controller/lib/aviary_rise_controller/aviary_rise_controller
source /home/root/ros-sources.sh

ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file /home/root/ros2_ws/src/aviary_rise_controller/param/params.yaml

DESIRED_TRAJECTORY=1 && ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/ros2_ws/src/aviary_rise_controller/param/baseline_params_${DESIRED_TRAJECTORY}.yaml"
DESIRED_TRAJECTORY=1 && ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/ros2_ws/src/aviary_rise_controller/param/resnet_params_${DESIRED_TRAJECTORY}.yaml"
DESIRED_TRAJECTORY=1 && ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/ros2_ws/src/aviary_rise_controller/param/integrated_resnet_params_${DESIRED_TRAJECTORY}.yaml"
DESIRED_TRAJECTORY=1 && ros2 run aviary_rise_controller aviary_rise_controller --ros-args --params-file "/home/root/ros2_ws/src/aviary_rise_controller/param/pid_params_${DESIRED_TRAJECTORY}.yaml"

PLOTTING DATA (Outside docker)
sudo chmod -R 777 plot_data/
python3 scripts/plot_csv_results.py plot_data/resnet/figure_eight/run_1.csv
python3 scripts/plot_csv_results.py plot_data/integrated_resnet/figure_eight/run_1.csv


RUNNING DEBUGGING NODE
source /home/root/ros-sources.sh
ros2 run debugging_node debugging_node --ros-args --params-file /home/root/ros2_ws/src/debugging_node/params/debugging_node.yaml



