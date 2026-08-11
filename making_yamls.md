Set up virtual environment.
``
pyenv local 3.13.14
python3 -m venv venv_host
source venv_host/bin/activate
pip install -e .
pip install --upgrade pip
``

Generate params for the Gazebo.
``
GAZEBO=true
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type pid --out ros2_ws/src/aviary_rise_controller/param/pid_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type integrated_resnet --out ros2_ws/src/aviary_rise_controller/param/integrated_resnet_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type resnet --out ros2_ws/src/aviary_rise_controller/param/resnet_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1
python scripts/generate_hardware_params.py --best_gains best_gains.yaml --controller_type baseline --out ros2_ws/src/aviary_rise_controller/param/baseline_params_1.yaml --gazebo $GAZEBO --desired_trajectory 1

``
