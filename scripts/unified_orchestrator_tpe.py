#!/usr/bin/env python3
import os
import subprocess
import time
import re
import argparse
import optuna
import sys
import yaml
from functools import partial

# Ensure JAX stays on CPU for any script-side ops (if any)
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Add ros2_ws paths if needed for jax_resnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "aviary_rise_controller", "aviary_rise_controller")))
from jax_resnet import get_total_parameters, init_resnet_weights

# --- HYPERPARAMETERS & GLOBALS ---
TRIALS_PHASE_1 = 50 
TRIALS_PHASE_2 = 50
TRIALS_PHASE_3 = 75
TRIALS_PHASE_4 = 50
PATIENCE = 0

SEED = 42

FIXED_X = 2.0
FIXED_Y = 4.0

RETRY_ATTEMPTS = 2

class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self.patience <= 0:
            return
            
        try:
            best_trial = study.best_trial
            # If the current trial number is X, and the best trial was found at X - patience, stop!
            if trial.number - best_trial.number >= self.patience:
                print("\n=========================================================================")
                print(f" [Early Stopping] No improvement in the last {self.patience} trials!")
                print(f" Best trial was #{best_trial.number} with cost {best_trial.value:.4f}.")
                print(" Exiting study prematurely.")
                print("=========================================================================\n")
                study.stop()
        except ValueError:
            # Raised if no trial has completed yet
            pass

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Optuna Orchestrator for Gazebo SITL")
    parser.add_argument(
        "--desired_trajectory",
        type=int,
        required=True,
        choices=[1, 2],
        help="Specify which trajectory to run (1: Warped Figure-Eight, 2: Rose Curve)."
    )
    return parser.parse_args()

def write_spawn_location(x: float, y: float) -> None:
    env_path = "spawn-locations.env"
    env_content = f"""CONTAINER_NAME="px4-sitl-gz"\nN_TB=0\nTB_SPAWN_LOCATIONS=()\nN_QUAD=1\nQUAD1_LOCATION="{x},{y},0.0"\nQUAD_SPAWN_LOCATIONS=($QUAD1_LOCATION)\n"""
    with open(env_path, "w") as f:
        f.write(env_content)

def cleanup_environment() -> None:
    print("[*] Purging environment...")
    subprocess.run(["./shutdown-background-services.sh"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

def wait_for_sim_ready() -> bool:
    print("[*] Polling for PX4 EKF2 and DDS bridge to initialize...")
    for attempt in range(40): 
        cmd = 'docker exec px4-sitl-gz bash -c "source /home/root/ros-sources.sh && timeout 2 ros2 topic echo --once /px4_1/fmu/out/vehicle_odometry"'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            print("[*] EKF2 converged! Odometry is publishing.")
            time.sleep(1)
            return True
        time.sleep(1)
    return False

def write_ros_params(param_dict: dict) -> str:
    """Dynamically generates the params.yaml file."""
    param_dir = os.path.expanduser("~/voxl-px4-sitl/ros2_ws/src/aviary_rise_controller/param")
    os.makedirs(param_dir, exist_ok=True)
    param_file = os.path.join(param_dir, "params.yaml")

    params = {
        'aviary_rise_node': {
            'ros__parameters': param_dict
        }
    }

    with open(param_file, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    return "/home/root/ros2_ws/src/aviary_rise_controller/param/params.yaml"

def execute_single_run(param_dict: dict, desired_trajectory: int, add_hanging_mass: bool):
    cleanup_environment()

    container_param_path = write_ros_params(param_dict)
    write_spawn_location(FIXED_X, FIXED_Y)

    # Inject into a copy of the current environment variables. Doesn't set/select a world
    # file (that's no longer this script's job -- run-gz-sim.sh always loads depot.sdf).
    # SLUNG_MASS_KG is spawn-sim-env.sh's actual hanging-mass mechanism (0 = no load, >0 =
    # generates + deploys sentinel_vision_slung/model.sdf via generate-slung-load-model.py
    # before spawning) -- 0.25 matches the tuned working target established separately.
    run_env = os.environ.copy()
    run_env["GZ_SEED"] = str(SEED)
    run_env["SLUNG_MASS_KG"] = "0.25" if add_hanging_mass else "0"

    # Pass the modified environment to the spawn script
    subprocess.run(["./spawn-sim-env.sh"], env=run_env, stdout=subprocess.DEVNULL)
    
    if not wait_for_sim_ready():
        print("[!] Polling timeout. Simulator failed to boot properly.")
        cleanup_environment()
        return None, None, None, None
    
    ros_cmd = (
        f"source /home/root/ros-sources.sh && "
        f"cd /home/root/ros2_ws && source install/setup.bash && "
        f"ros2 run aviary_rise_controller aviary_rise_controller "
        f"--ros-args --params-file {container_param_path}"
    )
    
    process = subprocess.Popen(
        ["docker", "exec", "px4-sitl-gz", "bash", "-c", ros_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    final_cost = None
    rms_error = None
    rms_control = None
    rms_jerk = None
    for line in process.stdout:
        print(f"  [ROS] {line.strip()}")
        # aviary_rise_node's [RESULT] lines are sentence-case ("Final cost", "RMS
        # error", ...), not the title-case these patterns were originally written
        # against -- IGNORECASE keeps this robust to either.
        if match := re.search(r"\[RESULT\] (?:Final )?Cost = ([\d\.]+)", line, re.IGNORECASE):
            final_cost = float(match.group(1))
        elif match := re.search(r"\[RESULT\] RMS Error = ([\d\.]+)", line, re.IGNORECASE):
            rms_error = float(match.group(1))
        elif match := re.search(r"\[RESULT\] RMS Control Effort = ([\d\.]+)", line, re.IGNORECASE):
            rms_control = float(match.group(1))
        elif match := re.search(r"\[RESULT\] RMS Control Jerk = ([\d\.]+)", line, re.IGNORECASE):
            rms_jerk = float(match.group(1))

    process.wait()
    cleanup_environment()
    return final_cost, rms_error, rms_control, rms_jerk

def evaluate_single(trial: optuna.Trial, param_dict: dict, desired_trajectory: int, add_hanging_mass: bool):
    base_config = get_base_param_dict(param_dict['controller_type'], desired_trajectory)
    full_params = {**base_config, **param_dict}
    
    # For neural networks, initialize weights if needed
    if param_dict['controller_type'] in ['resnet', 'integrated_resnet']:
        if 'initial_weight_scale_factor' in full_params:
            ws = full_params['initial_weight_scale_factor']
            full_params['h_act_func'] = 'swish'
            full_params['o_act_func'] = 'tanh'
            full_params['shortcut_act_func'] = 'swish'
            full_params['theta_bar'] = 1e6
            full_params['theta_dot_bar'] = 1e3
            full_params['d_in'] = 12 if param_dict['controller_type'] == "resnet" else 15
            
            key = jax.random.PRNGKey(SEED)
            initial_weights_jax = ws * init_resnet_weights(key, full_params['d_in'], full_params['hidden_width'], full_params['d_out'], full_params['num_blocks'], full_params['k_0'], full_params['k_i'], 'xavier', 'he')
            full_params['initial_weights'] = [float(w) for w in initial_weights_jax]
    
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        if attempt != 1:
            print(f"\n[*] Execution Attempt {attempt}/{RETRY_ATTEMPTS}")
        
        cost, rms_err, rms_ctrl, rms_jerk = execute_single_run(full_params, desired_trajectory, add_hanging_mass)

        if cost is not None:
            if rms_err is not None: trial.set_user_attr('e_RMS', rms_err)
            if rms_ctrl is not None: trial.set_user_attr('u_RMS', rms_ctrl)
            if rms_jerk is not None: trial.set_user_attr('u_dot_RMS', rms_jerk)
            return cost
            
        print("[!] Trial returned None (System/SITL failure). Retrying...")
        time.sleep(3)
        
    print(f"[!] All {RETRY_ATTEMPTS} attempts failed. Assigning ultimate penalty.")
    return 1e6

def get_base_param_dict(controller_type: str, desired_trajectory: int):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'conf', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['aviary_rise_node']['ros__parameters']
    config['is_gazebo'] = True
    config['vehicle_name'] = 'px4_1'
    config['desired_trajectory'] = desired_trajectory
    config['controller_type'] = controller_type
    return config

def run_stage_rise_baseline(trial: optuna.Trial, desired_trajectory: int, add_hanging_mass: bool) -> float:
    param_dict = {
        'controller_type': 'baseline'
    }
    param_dict['k_1'] = trial.suggest_float("k_1", 0.01, 2.0, log=True)
    param_dict['k_2'] = trial.suggest_float("k_2", 0.01, 8.0, log=True)
    param_dict['k_3'] = trial.suggest_float("k_3", param_dict['k_2'], 8.0)
    param_dict['k_rise'] = trial.suggest_float("k_rise", 0.01, 8.0, log=True)
    return evaluate_single(trial, param_dict, desired_trajectory, add_hanging_mass)

def get_best_params(study_name: str, db_file_path: str) -> dict:
    if os.path.exists(db_file_path):
        try:
            stage_db_url = f"sqlite:///{db_file_path}"
            study = optuna.load_study(study_name=study_name, storage=stage_db_url)
            return study.best_params
        except Exception as e:
            print(f"Warning: Could not load {study_name}. Error: {e}")
    return {}

def run_stage_2(trial: optuna.Trial, db_dir: str, controller_type: str, base_stage: str,
                 desired_trajectory: int, add_hanging_mass: bool) -> float:
    best_base_params = get_best_params(f"stage_{base_stage}_study", os.path.join(db_dir, f"stage_{base_stage}.db"))
    param_dict = {
        'controller_type': controller_type,
        'initial_weight_scale_factor': 0.2,
        'num_blocks': 6,
        'k_0': 2,
        'k_i': 2,
        'hidden_width': 8,
        'gamma': trial.suggest_float("gamma", 0.0, 25.0),
        'sigma_mod': trial.suggest_float("sigma_mod", 0.01, 5.0, log=True),
        **best_base_params
    }
    return evaluate_single(trial, param_dict, desired_trajectory, add_hanging_mass)

def run_stage_3(trial: optuna.Trial, desired_trajectory: int, add_hanging_mass: bool) -> float:
    k_st_1 = trial.suggest_float("k_st_1", 0.01, 10.0, log=True)
    k_st_2 = trial.suggest_float("k_st_2", 0.01, 10.0, log=True)
    k_st_3 = trial.suggest_float("k_st_3", 0.01, 10.0, log=True)

    # Store k_st_* in optuna, but map back to k_1, k_2, k_3 for the sim
    mapped_dict = {
        'controller_type': 'supertwisting',
        'k_1': k_st_1,
        'k_2': k_st_2,
        'k_3': k_st_3
    }
    return evaluate_single(trial, mapped_dict, desired_trajectory, add_hanging_mass)

def run_stage_4(trial: optuna.Trial, desired_trajectory: int, add_hanging_mass: bool) -> float:
    param_dict = {
        'controller_type': 'pid',
        'K_P': trial.suggest_float("K_P", 0.1, 15.0, log=True),
        'K_I': trial.suggest_float("K_I", 0.1, 15.0, log=True),
        'K_D': trial.suggest_float("K_D", 0.1, 15.0, log=True)
    }
    return evaluate_single(trial, param_dict, desired_trajectory, add_hanging_mass)

def extract_and_save_gains(db_dir: str):
    print("\n==========================================")
    print(" Extracting Best Gains...")
    print("==========================================")
    best_gains_path = os.path.join(db_dir, "best_gains.yaml")
    best_gains = {}

    def fetch_best(study_name: str, db_file: str) -> dict:
        full_path = os.path.join(db_dir, db_file)
        if not os.path.exists(full_path): return {}
        try:
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{full_path}")
            e_rms = study.best_trial.user_attrs.get('e_RMS', 'N/A')
            u_rms = study.best_trial.user_attrs.get('u_RMS', 'N/A')
            e_str = f"{e_rms:.4f}" if isinstance(e_rms, float) else e_rms
            u_str = f"{u_rms:.4f}" if isinstance(u_rms, float) else u_rms
            print(f"[{study_name}] Cost: {study.best_value:.4f} | e_RMS: {e_str} | u_RMS: {u_str}")
            return dict(study.best_params)
        except Exception as e:
            return {}

    fixed_arch = {'num_blocks': 6, 'k_0': 2, 'k_i': 2, 'hidden_width': 8}

    rise_params = {
        '1A': fetch_best("stage_1A_study", "stage_1A.db"),
        '1B': fetch_best("stage_1B_study", "stage_1B.db"),
    }
    for base_stage, params in rise_params.items():
        if params:
            best_gains[f'BEST_RISE_{base_stage}'] = {'controller_type': 'baseline', **params}

    # (controller_type, stage_letter, base_stage) for each of the four stage-2 variants
    nn_variants = [
        ('resnet', '2A', '1B'),
        ('resnet', '2C', '1A'),
        ('integrated_resnet', '2B', '1B'),
        ('integrated_resnet', '2D', '1A'),
    ]
    key_prefix = {'resnet': 'BEST_NN', 'integrated_resnet': 'BEST_INN'}
    for controller_type, stage_letter, base_stage in nn_variants:
        nn_params = fetch_best(f"stage_{stage_letter}_study", f"stage_{stage_letter}.db")
        stage2_base = rise_params[base_stage]
        if nn_params and stage2_base:
            best_gains[f'{key_prefix[controller_type]}_{base_stage}'] = {
                'controller_type': controller_type, **stage2_base, **fixed_arch, **nn_params
            }

    st_params = fetch_best("stage_3_study", "stage_3.db")
    if st_params:
        best_gains['BEST_ST'] = {
            'controller_type': 'supertwisting',
            'k_1': st_params.get('k_st_1', 1.0),
            'k_2': st_params.get('k_st_2', 1.0),
            'k_3': st_params.get('k_st_3', 1.0)
        }

    pid_params = fetch_best("stage_4_study", "stage_4.db")
    if pid_params:
        best_gains['BEST_PID'] = {'controller_type': 'pid', **pid_params}

    with open(best_gains_path, 'w') as f:
        yaml.dump(best_gains, f)
    print(f"\n[*] Extracted best gains and saved to '{best_gains_path}'")

def main():
    args = parse_args()
    db_dir = f"output/traj{args.desired_trajectory}"
    os.makedirs(db_dir, exist_ok=True)

    # Each tuple: (stage_name, stage_func, num_trials, add_hanging_mass)
    stages = [
        ('1A', run_stage_rise_baseline, TRIALS_PHASE_1, False),
        ('1B', run_stage_rise_baseline, TRIALS_PHASE_1, True),
        ('2A', partial(run_stage_2, db_dir=db_dir, controller_type='resnet', base_stage='1B'), TRIALS_PHASE_2, True),
        ('2B', partial(run_stage_2, db_dir=db_dir, controller_type='integrated_resnet', base_stage='1B'), TRIALS_PHASE_2, True),
        ('2C', partial(run_stage_2, db_dir=db_dir, controller_type='resnet', base_stage='1A'), TRIALS_PHASE_2, True),
        ('2D', partial(run_stage_2, db_dir=db_dir, controller_type='integrated_resnet', base_stage='1A'), TRIALS_PHASE_2, True),
        #('3', run_stage_3, TRIALS_PHASE_3, True),
        ('4', run_stage_4, TRIALS_PHASE_4, True),
    ]

    early_stopper = EarlyStoppingCallback(patience=PATIENCE)

    for stage_name, stage_func, num_trials, add_hanging_mass in stages:
        print(f"\n==========================================")
        print(f" Starting Stage {stage_name}")
        print(f"==========================================")

        db_file_path = os.path.join(db_dir, f"stage_{stage_name}.db")
        db_url = f"sqlite:///{db_file_path}"
        study_name = f"stage_{stage_name}_study"

        study = optuna.create_study(
            study_name=study_name,
            storage=db_url,
            load_if_exists=True,
            direction="minimize"
        )

        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining = max(0, num_trials - completed)
        if remaining == 0:
            print(f"[*] Stage {stage_name} already has {completed}/{num_trials} completed trials. Skipping.")
            continue

        print(f"[*] Stage {stage_name}: {completed}/{num_trials} trials already completed. Running {remaining} more.")
        bound_func = partial(stage_func, desired_trajectory=args.desired_trajectory, add_hanging_mass=add_hanging_mass)
        study.optimize(bound_func, n_trials=remaining, callbacks=[early_stopper])

    extract_and_save_gains(db_dir)
    print("\n==========================================")
    print(" Pipeline Complete.")
    print("==========================================")

if __name__ == "__main__":
    main()
