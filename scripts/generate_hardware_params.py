#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "aviary_rise_controller", "aviary_rise_controller")))
from jax_resnet import init_resnet_weights

from unified_orchestrator_tpe import SEED

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    default_config_path = os.path.join(os.path.dirname(__file__), '..', 'conf', 'config.yaml')

    parser = argparse.ArgumentParser(description="Generate Hardware Param YAML")
    parser.add_argument("--best_gains", type=str, required=True, help="Path to best_gains.yaml file")
    parser.add_argument("--controller_type", type=str, choices=["baseline", "resnet", "integrated_resnet", "supertwisting", "pid"], required=True)
    parser.add_argument("--desired_trajectory", type=int, choices=[1, 2], required=True, help="Desired trajectory (optional override)")
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to base config.yaml")
    parser.add_argument("--out", type=str, default="hardware_params.yaml", help="Output yaml file path")
    parser.add_argument("--gazebo", type=str2bool, required=True, help="Gazebo or real-world experiment?")
    args = parser.parse_args()

    target_controller_type = args.controller_type

    if not os.path.exists(args.best_gains):
        raise FileNotFoundError(f"Best gains file not found: {args.best_gains}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Base config file not found: {args.config}")

    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
        base_config = full_config['aviary_rise_node']['ros__parameters']

    with open(args.best_gains, 'r') as f:
        best_gains = yaml.safe_load(f)

    # Find the specific controller parameters in best_gains.yaml
    controller_params = None
    for key, params in best_gains.items():
        if params['controller_type'] == target_controller_type:
            controller_params = params
            break

    if controller_params is None:
        raise ValueError(f"Could not find controller_type '{target_controller_type}' in {args.best_gains}")

    # Build the combined parameter dictionary
    param_dict = base_config.copy()
    param_dict.update(controller_params)

    # Allow overriding desired_trajectory via command line
    if args.desired_trajectory is not None:
        param_dict['desired_trajectory'] = args.desired_trajectory

    # Hardware-specific overrides translated from the old script
    param_dict['is_gazebo'] = args.gazebo
    param_dict['save_data'] = True
    param_dict['mpc_acc_vert_max_mps2'] = 6.0
    param_dict['odom_timeout_s'] = 1.0
    param_dict['odom_watchdog_freq_hz'] = 10.0
    param_dict['vehicle_name'] = 'px4_1' if args.gazebo else 'sentinel5'

    # Generate initial neural network weights if applicable. best_gains.yaml only ever
    # carries what Optuna actually tuned (gamma, sigma_mod, ...) plus the fixed
    # architecture dict (num_blocks/k_0/k_i/hidden_width) baked in by
    # unified_orchestrator_tpe.py's run_stage_2 -- theta_bar, theta_dot_bar, d_in, and the
    # runtime activation functions are never tuned, so they're set here the same way
    # unified_orchestrator_tpe.py/run_best_gains_tpe.py set them for every resnet-family run.
    if target_controller_type in ["resnet", "integrated_resnet"]:
        param_dict['d_in'] = 15 if target_controller_type == "integrated_resnet" else 12
        param_dict['theta_bar'] = 1e6
        param_dict['theta_dot_bar'] = 1e3
        param_dict['h_act_func'] = 'swish'
        param_dict['o_act_func'] = 'tanh'
        param_dict['shortcut_act_func'] = 'swish'

        init_scale = param_dict['initial_weight_scale_factor']
        key = jax.random.PRNGKey(SEED)

        # h_method/o_method are init_resnet_weights' own weight-initialization scheme
        # (distinct from h_act_func/o_act_func/shortcut_act_func above, which are the
        # runtime activation functions the node reads as ROS params) -- hardcoded to
        # match every other current call site, not sourced from best_gains.yaml.
        initial_weights_jax = init_scale * init_resnet_weights(
            key=key,
            d_in=param_dict['d_in'],
            hidden_width=param_dict['hidden_width'],
            d_out=param_dict['d_out'],
            b=param_dict['num_blocks'],
            k_0=param_dict['k_0'],
            k_i=param_dict['k_i'],
            h_method='xavier',
            o_method='he'
        )
        param_dict['initial_weights'] = [float(w) for w in initial_weights_jax]

    params = {
        'aviary_rise_node': {
            'ros__parameters': param_dict
        }
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    print(f"[*] Generated hardware parameters for {target_controller_type.upper()} running Trajectory {param_dict['desired_trajectory']}.")
    print(f"[*] Saved to {args.out}")

if __name__ == "__main__":
    main()
