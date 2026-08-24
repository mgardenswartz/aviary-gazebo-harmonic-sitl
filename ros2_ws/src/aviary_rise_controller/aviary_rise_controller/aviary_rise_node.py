import os
import gc
import math
import time
import csv
from functools import partial
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry

import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu") # Force use of CPU since quad has no GPU
jax.config.update("jax_enable_x64", True) # Use 64 bit since all floats to be used are doubles; otherwise XLA recompilation will occur mid-flight
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from jax_resnet import resnet_network
from aviary_rise_controller.proj import discrete_projection, discrete_rate_projection
from aviary_rise_controller.desired_trajectory import TrajectoryGenerator

class ExperimentState:
    STATE_INIT: int = 0
    STATE_TAKEOFF: int = 1
    STATE_FOLLOW_TRAJ: int = 2
    STATE_PAUSED: int = 3

class JaxLatencyError(Exception):
    pass

class OdomTimeoutError(Exception):
    pass

class FailsafeTriggeredError(Exception):
    pass

class BoundaryBreachError(Exception):
    pass

class ExperimentFinished(Exception):
    pass

class ControlLoopOverrunError(Exception):
    pass

# Param names that are only ever fetched for SOME controller_type values (e.g. K_P/K_I/K_D
# are read when controller_type == 'pid' but not otherwise). A single params.yaml commonly
# holds gains for every controller type at once so it's quick to switch controller_type
# without re-adding gains -- so these names are legitimate to leave declared-but-unused for
# any one run and must not trip the unrecognized-parameter check in
# _validate_declared_parameters(). Anything NOT in this set (plus what a given run actually
# fetches) is either a genuine typo/stale key or an always-required name -- both real bugs.
CONTROLLER_CONDITIONAL_PARAM_NAMES: set = {
    'K_P', 'K_I', 'K_D',
    'k_1', 'k_2', 'k_3', 'k_rise',
    'd_in', 'initial_weights', 'gamma', 'sigma_mod', 'theta_bar', 'theta_dot_bar',
    'hidden_width', 'num_blocks', 'k_0', 'k_i', 'h_act_func', 'o_act_func', 'shortcut_act_func',
    'use_sim_time' # TEMP
}

# Param names that are declared in the YAML purely as input to
# scripts/generate_hardware_params.py (consumed there to bake initial_weights before the
# YAML is ever written) and are never read by the node itself at runtime.
GENERATOR_ONLY_PARAM_NAMES: set = {
    'initial_weight_scale_factor',
}

# Fraction of control_period_s a single control_timer_callback tick is allowed to consume
# before it's treated as a real-time violation. Kept as a code constant (not a YAML param):
# it's a property of the control-loop deadline itself, not something a given experiment
# should be tuning per run.
CONTROL_TICK_BUDGET_FRACTION: float = 0.90

class AviaryRiseNode(Node):
    def __init__(self) -> None:
        super().__init__(
            node_name='aviary_rise_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        self._used_param_names: set = set()

        # Basic Parameters of the Experiment
        self.is_gazebo: bool = self._get_param(name='is_gazebo')
        self.desired_trajectory: int = self._get_param(name='desired_trajectory')
        self.vehicle_name: str = self._get_param(name='vehicle_name')
        self.controller_type: str = self._get_param(name='controller_type')
        control_frequency_hz: float = self._get_param(name='control_frequency_hz')
        self.control_period_s: float = 1.0 / control_frequency_hz
        self.save_data: bool = self._get_param(name='save_data')
        self.trial_number: Optional[int] = self._get_param(name='trial_number') if self.has_parameter('trial_number') else None
        self.run_length_s: float = self._get_param(name='run_length_s')
        self.init_tol_m: float = self._get_param(name='init_tol_m')
        self.d_out: int = self._get_param(name='d_out')

        # Desired Trajectory
        if self.desired_trajectory not in [1,2]:
            raise ValueError("INVALID DESIRED TRAJECTORY SELECTED.")
        # Fix: Convert parameters to primitive values for config
        self.config: Dict[str, Any] = {k: v.value for k, v in self.get_parameters_by_prefix(prefix='').items()}
        self.traj_gen: TrajectoryGenerator = TrajectoryGenerator(config=self.config)

        self._used_param_names.update([
            'traj1_center_z_m_ned', 'traj1_period_s', 'traj1_x_amp_m_ned',
            'traj1_y_amp_m_ned', 'traj1_z_amp_m_ned', 'traj1_alpha_warp',
            'traj2_center_z_m_ned', 'traj2_petal_radius_m', 'traj2_target_speed_mps'
        ])

        # Safety
        self.acc_hor_max_mps2: float = self._get_param(name='mpc_acc_hor_max_mps2')
        self.acc_vert_max_mps2: float = self._get_param(name='mpc_acc_vert_max_mps2')
        self.safe_x_min_m_ned: float = self._get_param(name='safe_x_min_m_ned')
        self.safe_x_max_m_ned: float = self._get_param(name='safe_x_max_m_ned')
        self.safe_y_min_m_ned: float = self._get_param(name='safe_y_min_m_ned')
        self.safe_y_max_m_ned: float = self._get_param(name='safe_y_max_m_ned')
        self.safe_z_min_m_ned: float = self._get_param(name='safe_z_min_m_ned')
        self.safe_z_max_m_ned: float = self._get_param(name='safe_z_max_m_ned')
        self.odom_timeout_s: float = self._get_param(name='odom_timeout_s')
        self.init_z_m_ned: float = self._get_param(name='init_z_m_ned')
        self.odom_watchdog_freq_hz: float = self._get_param(name='odom_watchdog_freq_hz')
        self.mode_cmd_retry_period_s: float = self._get_param(name='mode_cmd_retry_period_s')
        self.takeoff_timeout_s: float = self._get_param(name='takeoff_timeout_s')

        self._validate_trajectory_envelope()

        # Cost Function (same formula used for post-hoc gain selection in
        # unified_orchestrator.py's compute_trial_J - no t-weighting on tracking error)
        self.q_e: float = self._get_param(name='q_e')
        self.r_u: float = self._get_param(name='r_u')
        self.r_udot: float = self._get_param(name='r_udot')
        self.w_fail: float = self._get_param(name='w_fail')

        if self.controller_type == "pid":
            self.K_P: float = self._get_param(name='K_P')
            self.K_I: float = self._get_param(name='K_I')
            self.K_D: float = self._get_param(name='K_D')

        elif self.controller_type in ['baseline', 'integrated_resnet', 'resnet', 'supertwisting']:
            self.k_1: float = self._get_param(name='k_1')
            self.k_2: float = self._get_param(name='k_2')
            self.k_3: float = self._get_param(name='k_3')

            if self.controller_type in ['baseline', 'integrated_resnet', 'resnet']:
                self.K_RISE: float = self._get_param(name='k_rise')
                self.K_P: float = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
                self.K_I: float = (self.k_1 * self.k_2 * self.k_3) + self.k_1
                self.K_D: float = self.k_1 + self.k_2 + self.k_3

            if self.controller_type in ["resnet", "integrated_resnet"]:
                self.d_in: int = self._get_param(name='d_in')

                self.theta_hat: jax.Array = jnp.array(object=self._get_param(name='initial_weights'))

                self.gamma_diag: jax.Array = jnp.ones(shape=self.theta_hat.shape[0]) * self._get_param(name='gamma')
                self.sigma_mod: float = self._get_param(name='sigma_mod')
                self.theta_bar: float = self._get_param(name='theta_bar')
                self.theta_dot_bar: float = self._get_param(name='theta_dot_bar')

                self.bound_resnet = jax.jit(partial(
                    resnet_network,
                    d_in=self.d_in,
                    hidden_width=self._get_param(name='hidden_width'),
                    d_out=self.d_out,
                    b=self._get_param(name='num_blocks'),
                    k_0=self._get_param(name='k_0'),
                    k_i=self._get_param(name='k_i'),
                    h_act_func=self._get_param(name='h_act_func'),
                    o_act_func=self._get_param(name='o_act_func'),
                    shortcut_act_func=self._get_param(name='shortcut_act_func'),
                ))

                @jax.jit
                def compiled_update_step(theta_hat: jax.Array, x_vec: jax.Array, r1_vec: jax.Array, dt: float, theta_bar: float, theta_dot_bar: float, gamma_diag: jax.Array, s_mod: float, control_saturated: bool) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                    phi_val, vjp_fn = jax.vjp(lambda t: self.bound_resnet(t, x_vec), has_aux=False, *[theta_hat])
                    grad_term = vjp_fn(r1_vec)[0]
                    theta_dot_unprojected = gamma_diag * (grad_term - s_mod * theta_hat)
                    # theta_hat_dot = sat(proj(nominal_theta_hat_dot)): discrete_projection is
                    # the "proj" stage (ball-constrains the state), discrete_rate_projection is
                    # the "sat" stage (caps the resulting effective rate's 2-norm), applied in
                    # that order -- see proj.py for why the order matters.
                    theta_next_ball, ball_projected = discrete_projection(theta_hat=theta_hat, theta_dot_unprojected=theta_dot_unprojected, dt=dt, theta_bar=theta_bar, gamma_diag=gamma_diag)
                    theta_next_rate_capped, rate_limited = discrete_rate_projection(theta_hat=theta_hat, theta_next=theta_next_ball, dt=dt, theta_dot_bar=theta_dot_bar)
                    # Control-command saturation dominates both projections above: if the
                    # published acceleration was clamped this tick, freeze theta_hat entirely
                    # rather than merely rate-limiting it.
                    final_theta = jax.lax.select(pred=control_saturated, on_true=theta_hat, on_false=theta_next_rate_capped)
                    # Actual applied per-tick derivative (post-projection/saturation/freeze),
                    # not the nominal theta_dot_unprojected above -- this is what
                    # theta_dot_bar actually bounds. dt is a measured (not fixed) period and
                    # can be 0 on the very first FOLLOW_TRAJ tick; final_theta == theta_hat
                    # in that case too (see discrete_projection/discrete_rate_projection), so
                    # the numerator is already 0 and jnp.maximum below just avoids a 0/0 NaN.
                    theta_hat_dot = (final_theta - theta_hat) / jnp.maximum(dt, 1e-12)
                    return final_theta, phi_val, ball_projected, rate_limited, theta_hat_dot

                self.compiled_update_step = compiled_update_step
                self.precompile_jax()

        # For VehicleStatus callback
        self.nav_state: int = 0
        self.vehicle_system_id: int = 1
        self.vehicle_component_id: int = 1

        # Init
        self.is_armed: bool = False
        self.in_offboard_mode: bool = False
        self.landing_command_sent: bool = False
        self.publish_offboard_heartbeat: bool = False
        self.position_mode_requested: bool = False
        self._mode_cmd_seeded: bool = False
        self.cost_started: bool = False
        self.is_control_saturated: bool = False
        self.freeze_int_xy: bool = False
        self.freeze_int_z: bool = False
        self.initial_position_locked: bool = False
        self.latest_odom: Optional[VehicleOdometry] = None

        self.last_odom_ros_time_s: float = 0.0
        self.init_x_m_ned: float = 0.0
        self.init_y_m_ned: float = 0.0
        self.experiment_state: int = ExperimentState.STATE_INIT
        self.t_0: float = 0.0

        self.last_t_s: float = 0.0
        self.last_mode_cmd_time_s: float = 0.0
        self.pause_start_time_s: float = 0.0
        self.takeoff_entry_time_s: float = 0.0
        self.pre_pause_state: int = ExperimentState.STATE_INIT

        self.ticks_without_odom: int = 0
        self.reset_integral()
        self.cost_J: float = 0.0
        self.last_cost_integrand: float = 0.0
        self.error_sq_integral: float = 0.0
        self.last_error_sq: float = 0.0
        self.u_sq_integral: float = 0.0
        self.last_u_sq: float = 0.0
        self.u_dot_sq_integral: float = 0.0
        self.last_u_dot_sq: float = 0.0
        self.last_u: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
        self.time_history: List[float] = []
        self.control_output_norm_history: List[float] = []
        self.control_output_history: List[List[float]] = []
        self.u_dot_history: List[List[float]] = []
        self.error_norm_history: List[float] = []
        self.weight_history: List[List[float]] = []
        self.phi_history: List[List[float]] = []
        self.theta_hat_norm_history: List[float] = []
        self.theta_hat_dot_norm_history: List[float] = []
        self.ball_projected_history: List[bool] = []
        self.rate_limited_history: List[bool] = []
        self.q_history: List[List[float]] = []
        self.qd_history: List[List[float]] = []

        qos_profile: QoSProfile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            msg_type=OffboardControlMode, topic=f'/{self.vehicle_name}/fmu/in/offboard_control_mode', qos_profile=qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            msg_type=TrajectorySetpoint, topic=f'/{self.vehicle_name}/fmu/in/trajectory_setpoint', qos_profile=qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            msg_type=VehicleCommand, topic=f'/{self.vehicle_name}/fmu/in/vehicle_command', qos_profile=qos_profile)

        self.status_sub = self.create_subscription(
            msg_type=VehicleStatus, topic=f'/{self.vehicle_name}/fmu/out/vehicle_status', callback=self.vehicle_status_callback, qos_profile=qos_profile)
        self.odom_sub = self.create_subscription(
            msg_type=VehicleOdometry, topic=f'/{self.vehicle_name}/fmu/out/vehicle_odometry', callback=self.odom_callback, qos_profile=qos_profile)

        self.control_timer = self.create_timer(timer_period_sec=self.control_period_s, callback=self.control_timer_callback)

        self.odom_watchdog_timer = self.create_timer(timer_period_sec=1.0/self.odom_watchdog_freq_hz, callback=self.odom_watchdog_callback)

        self.offboard_heartbeat_timer = self.create_timer(timer_period_sec=self.control_period_s, callback=self.offboard_heartbeat_callback)

        self._validate_declared_parameters()

        self.get_logger().info(f"Node initialized successfully. Controller: {self.controller_type.upper()} | Trajectory: {self.desired_trajectory} | Gazebo mode: {self.is_gazebo}.")

    def _get_param(self, name: str) -> Any:
        self._used_param_names.add(name)
        return self.get_parameter(name=name).value

    def _validate_declared_parameters(self) -> None:
        # The flip side of never falling back to a default: a param name that's declared
        # (present in the YAML) but never fetched anywhere above is either a typo of a real
        # name or a stale key left behind by a rename -- both are bugs we want to catch at
        # startup, not silently ignore. CONTROLLER_CONDITIONAL_PARAM_NAMES is excluded since
        # a single params.yaml legitimately keeps every controller type's gains declared at
        # once so switching controller_type doesn't require re-adding them.
        declared_names: set = set(self.get_parameters_by_prefix(prefix='').keys())
        unrecognized: set = declared_names - self._used_param_names - CONTROLLER_CONDITIONAL_PARAM_NAMES - GENERATOR_ONLY_PARAM_NAMES
        if unrecognized:
            self.get_logger().fatal(f"Unrecognized parameter(s) in YAML, not read by this node: {sorted(unrecognized)}. Check for typos or stale/renamed keys.")
            raise ValueError(f"Unrecognized parameter(s): {sorted(unrecognized)}.")

    def _validate_trajectory_envelope(self) -> None:
        # Catches a correctly-named-but-dangerously-valued config (e.g. a trajectory
        # amplitude that overruns the safety box) at startup instead of discovering it via
        # a live boundary-breach failsafe mid-flight.
        if not (self.safe_z_min_m_ned <= self.init_z_m_ned <= self.safe_z_max_m_ned):
            raise ValueError(f"init_z_m_ned={self.init_z_m_ned} falls outside safe_z bounds [{self.safe_z_min_m_ned}, {self.safe_z_max_m_ned}].")

        num_samples: int = 200
        for i in range(num_samples + 1):
            t: float = self.run_length_s * i / num_samples
            pos, _, _ = self.traj_gen.get_desired_state(t=t)
            if not (self.safe_x_min_m_ned <= pos[0] <= self.safe_x_max_m_ned):
                raise ValueError(f"Trajectory x position {pos[0]:.2f}m at t={t:.2f}s falls outside safe_x bounds [{self.safe_x_min_m_ned}, {self.safe_x_max_m_ned}].")
            if not (self.safe_y_min_m_ned <= pos[1] <= self.safe_y_max_m_ned):
                raise ValueError(f"Trajectory y position {pos[1]:.2f}m at t={t:.2f}s falls outside safe_y bounds [{self.safe_y_min_m_ned}, {self.safe_y_max_m_ned}].")
            if not (self.safe_z_min_m_ned <= pos[2] <= self.safe_z_max_m_ned):
                raise ValueError(f"Trajectory z position {pos[2]:.2f}m at t={t:.2f}s falls outside safe_z bounds [{self.safe_z_min_m_ned}, {self.safe_z_max_m_ned}].")

        self.get_logger().info("Trajectory envelope validated against safety boundaries.")

    def _log_theta_saturation(self, t: float, ball_projected: bool, rate_limited: bool) -> None:
        # Surfaced at INFO (not DEBUG) since these are meant to be visible in a normal
        # run: theta_bar/theta_dot_bar are sized to never bind in practice, so either
        # one firing is itself a signal worth seeing live, not just on request.
        if ball_projected and rate_limited:
            self.get_logger().info(f"theta_bar AND theta_dot_bar saturation both triggered at t={t:.2f}s.")
        elif ball_projected:
            self.get_logger().info(f"theta_bar (weight-norm ball) saturation triggered at t={t:.2f}s.")
        elif rate_limited:
            self.get_logger().info(f"theta_dot_bar (update-rate) saturation triggered at t={t:.2f}s.")

    def precompile_jax(self) -> None:
        dummy_x: jax.Array = jnp.zeros(shape=self.d_in)
        dummy_r1: jax.Array = jnp.zeros(shape=self.d_out)
        self.get_logger().info("Compiling XLA graph on CPU...")

        self.theta_hat, _, _, _, _ = self.compiled_update_step(
            theta_hat=self.theta_hat,
            x_vec=dummy_x,
            r1_vec=dummy_r1,
            dt=self.control_period_s,
            theta_bar=self.theta_bar,
            theta_dot_bar=self.theta_dot_bar,
            gamma_diag=self.gamma_diag,
            s_mod=self.sigma_mod,
            control_saturated=False
        )
        self.theta_hat.block_until_ready()

        start_time: float = time.perf_counter()
        self.theta_hat, _, _, _, _ = self.compiled_update_step(
            theta_hat=self.theta_hat,
            x_vec=dummy_x,
            r1_vec=dummy_r1,
            dt=self.control_period_s,
            theta_bar=self.theta_bar,
            theta_dot_bar=self.theta_dot_bar,
            gamma_diag=self.gamma_diag,
            s_mod=self.sigma_mod,
            control_saturated=False
        )
        self.theta_hat.block_until_ready()
        hot_time: float = time.perf_counter() - start_time

        # Reset the weights back to true initial conditions
        self.theta_hat = jnp.array(object=self._get_param(name='initial_weights'))
        self.theta_hat.block_until_ready()
        self.get_logger().info(f"Neural network latency: {hot_time*1000:.2f}ms.")
        if hot_time > self.control_period_s:
            self.get_logger().fatal(f"Execution time {hot_time:.4f}s exceeds control_period_s={self.control_period_s:.4f}s limit.")
            raise JaxLatencyError("ResNet latency too high for selected control frequency (init).")

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        self.nav_state = msg.nav_state
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_offboard_mode = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.vehicle_system_id = msg.system_id
        self.vehicle_component_id = msg.component_id

    def odom_callback(self, msg: VehicleOdometry) -> None:
        self.latest_odom = msg
        self.ticks_without_odom = 0
        if not self.is_gazebo:
            self.last_odom_ros_time_s = self.get_clock().now().nanoseconds / 1e9

        if not self.initial_position_locked:
            self.init_x_m_ned = float(msg.position[0])
            self.init_y_m_ned = float(msg.position[1])
            self.initial_position_locked = True

    def odom_watchdog_callback(self) -> None:
        self.ticks_without_odom += 1

        if not self.initial_position_locked:
            if self.ticks_without_odom >= (self.odom_timeout_s * self.odom_watchdog_freq_hz):
                self.publish_offboard_heartbeat = False
                raise OdomTimeoutError("No odometry received at boot.")
        else:
            if self.is_gazebo:
                if self.ticks_without_odom >= (self.odom_timeout_s * self.odom_watchdog_freq_hz):
                    self.publish_offboard_heartbeat = False
                    raise OdomTimeoutError("Simulation running behind schedule.")
            else:
                # Use original wall clock logic for real vehicle (sim-to-real)
                current_time_s: float = self.get_clock().now().nanoseconds / 1e9
                if (current_time_s - self.last_odom_ros_time_s) > self.odom_timeout_s:
                    self.publish_offboard_heartbeat = False
                    raise OdomTimeoutError("Odometry feed lost during flight.")

    def reset_integral(self) -> None:
        self.current_integral_control_term = np.zeros(shape=self.d_out, dtype=np.float64)
        self.last_control_integrand = np.zeros(shape=self.d_out, dtype=np.float64)
        self.st_integral = np.zeros(shape=self.d_out, dtype=np.float64)

    def publish_vehicle_command(self, command: int, param1: float, param2: float) -> None:
        self.get_logger().debug(f"Publishing command {command}.")
        msg: VehicleCommand = VehicleCommand()
        msg.timestamp = int(self.latest_odom.timestamp) if self.latest_odom is not None else int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.vehicle_system_id
        msg.target_component = self.vehicle_component_id
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_publisher.publish(msg)

    def offboard_heartbeat_callback(self) -> None:
        if not self.publish_offboard_heartbeat:
            return

        msg: OffboardControlMode = OffboardControlMode()
        msg.timestamp = int(self.latest_odom.timestamp) if self.latest_odom is not None else int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = True
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint_acceleration(self, ax: float, ay: float, az: float) -> None:
        if self.latest_odom is None:
            self.get_logger().warning(f"Ignoring setpoint since there has been no odometry yet.")
            return

        msg: TrajectorySetpoint = TrajectorySetpoint()
        msg.timestamp = self.latest_odom.timestamp
        msg.acceleration = [ax, ay, az]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.yaw = 0.0  # Command a heading of 0.0 always
        self.trajectory_setpoint_publisher.publish(msg)

    def land_vehicle(self) -> None:
        if self.landing_command_sent:
            return
        self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_NAV_LAND, param1=0.0, param2=0.0)
        self.landing_command_sent = True

    def write_csv(self) -> None:
        traj_name: str = ""
        match self.desired_trajectory:
            case 1:
                traj_name = "figure_eight"
            case 2:
                traj_name = "rose"

        base_dir: str = f"/home/root/plot_data/{self.controller_type}/{traj_name}"
        os.makedirs(name=base_dir, exist_ok=True)

        if self.trial_number is not None:
            # Deterministic name tied to the Optuna trial number so a retried attempt
            # overwrites the discarded attempt's file instead of leaving an orphaned CSV
            # that can't be matched back to a trial (see item 7 in the migration writeup).
            csv_filename: str = os.path.join(base_dir, f"run_trial{self.trial_number}.csv")
        else:
            existing_files: List[str] = [f for f in os.listdir(path=base_dir) if f.endswith('.csv') and f.startswith('run_')]
            max_idx: int = 0
            for f in existing_files:
                try:
                    idx = int(f.replace('run_', '').replace('.csv', ''))
                    max_idx = max(max_idx, idx)
                except ValueError:
                    pass
            iterable: int = max_idx + 1
            csv_filename: str = os.path.join(base_dir, f"run_{iterable}.csv")
        try:
            with open(file=csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                headers: List[str] = [
                    "Time_s", "Error_Norm_m", "Control_Output_Norm_mps2",
                    "ux_mps2", "uy_mps2", "uz_mps2",
                    "udotx_mps3", "udoty_mps3", "udotz_mps3",
                    "x_m", "y_m", "z_m", "xd_m", "yd_m", "zd_m"
                ]
                if self.controller_type in ["resnet", "integrated_resnet"] and self.theta_hat_norm_history:
                    headers += ["ThetaHat_Norm", "ThetaHatDot_Norm", "ThetaBar_Projected", "ThetaDotBar_Saturated"]
                if self.controller_type in ["resnet", "integrated_resnet"] and self.phi_history:
                    num_phi: int = len(self.phi_history[0])
                    headers += [f"Phi{i}_mps2" for i in range(num_phi)]
                if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                    num_weights: int = len(self.weight_history[0])
                    headers += [f"W{i}" for i in range(num_weights)]
                writer.writerow(headers)
                for i in range(len(self.time_history)):
                    row: List[float] = [
                        self.time_history[i], self.error_norm_history[i], self.control_output_norm_history[i],
                        self.control_output_history[i][0], self.control_output_history[i][1], self.control_output_history[i][2],
                        self.u_dot_history[i][0], self.u_dot_history[i][1], self.u_dot_history[i][2],
                        self.q_history[i][0], self.q_history[i][1], self.q_history[i][2],
                        self.qd_history[i][0], self.qd_history[i][1], self.qd_history[i][2]
                    ]
                    if self.controller_type in ["resnet", "integrated_resnet"] and self.theta_hat_norm_history:
                        row += [
                            self.theta_hat_norm_history[i], self.theta_hat_dot_norm_history[i],
                            self.ball_projected_history[i], self.rate_limited_history[i]
                        ]
                    if self.controller_type in ["resnet", "integrated_resnet"] and self.phi_history:
                        row += self.phi_history[i]
                    if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                        row += self.weight_history[i]
                    writer.writerow(row)
            self.get_logger().info(f"Telemetry saved to {csv_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV: {e}")

    def check_safety_boundary(self, q: np.ndarray) -> Optional[str]:
        if not (self.safe_x_min_m_ned <= q[0] <= self.safe_x_max_m_ned):
            return f"X position {q[0]:.2f} breached bounds [{self.safe_x_min_m_ned}, {self.safe_x_max_m_ned}]."
        if not (self.safe_y_min_m_ned <= q[1] <= self.safe_y_max_m_ned):
            return f"Y position {q[1]:.2f} breached bounds [{self.safe_y_min_m_ned}, {self.safe_y_max_m_ned}]."
        if not (self.safe_z_min_m_ned <= q[2] <= self.safe_z_max_m_ned):
            return f"Z position {q[2]:.2f} breached bounds [{self.safe_z_min_m_ned}, {self.safe_z_max_m_ned}]."
        return None

    def get_desired_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.experiment_state == ExperimentState.STATE_TAKEOFF:
            # During takeoff, hold exactly above where it initialized
            return (np.array(object=[self.init_x_m_ned, self.init_y_m_ned, self.init_z_m_ned], dtype=np.float64),
                    np.zeros(shape=3, dtype=np.float64), np.zeros(shape=3, dtype=np.float64))

        return self.traj_gen.get_desired_state(t=t)

    def compute_control_output(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        qd: np.ndarray,
        qd_dot: np.ndarray,
        qd_ddot: np.ndarray,
        e: np.ndarray,
        e_dot: np.ndarray,
        r1: Optional[np.ndarray],
        dt: float,
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], bool, bool]:
        # Saturation clamping is deliberately NOT done here: it happens in the
        # caller, after u is recorded into history/cost tracking, so logged/cost-tracked
        # control effort stays pre-saturation while only the published setpoint is clamped
        # (matches the original combined-case ordering exactly). phi_val (the NN
        # feedforward term) is returned alongside u purely for history/CSV logging --
        # it's zero and unused for every controller_type except resnet/integrated_resnet,
        # where it's already folded into u below. theta_hat_dot/ball_projected/rate_limited
        # are likewise CSV-logging-only and stay at their None/False defaults for every
        # controller_type except resnet/integrated_resnet.
        u: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
        phi_val: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
        theta_hat_dot_val: Optional[np.ndarray] = None
        ball_projected: bool = False
        rate_limited: bool = False

        match self.controller_type:
            case "baseline":
                current_integrand: np.ndarray = (self.K_I * e) + (self.K_RISE * np.sign(r1))
                delta_int: np.ndarray = (dt / 2.0) * (current_integrand + self.last_control_integrand)
                if not self.freeze_int_xy:
                    self.current_integral_control_term[0:2] += delta_int[0:2]
                if not self.freeze_int_z:
                    self.current_integral_control_term[2] += delta_int[2]
                self.last_control_integrand = current_integrand
                u = (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term

            case "pid":
                current_integrand: np.ndarray = (self.K_I * e)
                delta_int: np.ndarray = (dt / 2.0) * (current_integrand + self.last_control_integrand)
                if not self.freeze_int_xy:
                    self.current_integral_control_term[0:2] += delta_int[0:2]
                if not self.freeze_int_z:
                    self.current_integral_control_term[2] += delta_int[2]
                self.last_control_integrand = current_integrand
                u = (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term

            case "resnet":
                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                    x_vec: jax.Array = jnp.array(object=np.concatenate((q, q_dot, qd, qd_dot)))

                    t_start_jax: float = time.perf_counter()
                    self.theta_hat, phi_out, ball_projected, rate_limited, theta_hat_dot = self.compiled_update_step(
                        theta_hat=self.theta_hat,
                        x_vec=x_vec,
                        r1_vec=jnp.array(object=r1),
                        dt=dt,
                        theta_bar=self.theta_bar,
                        theta_dot_bar=self.theta_dot_bar,
                        gamma_diag=self.gamma_diag,
                        s_mod=self.sigma_mod,
                        control_saturated=False #self.is_control_saturated # I'm temporarily turning this off on purpose.
                    )
                    self.theta_hat.block_until_ready()
                    t_end_jax: float = time.perf_counter()
                    jax_dt: float = t_end_jax - t_start_jax
                    if jax_dt > self.control_period_s:
                        self.get_logger().warning(f"Running behind! JAX took {jax_dt*1000:.2f}ms at t={t:.2f}s.")
                    else:
                        self.get_logger().debug(f"JAX took {jax_dt*1000:.2f}ms.")

                    self._log_theta_saturation(t=t, ball_projected=bool(ball_projected), rate_limited=bool(rate_limited))

                    phi_val = np.array(object=phi_out, dtype=np.float64)
                    theta_hat_dot_val = np.array(object=theta_hat_dot, dtype=np.float64)
                    ball_projected = bool(ball_projected)
                    rate_limited = bool(rate_limited)

                current_integrand_res: np.ndarray = (self.K_I * e) + (self.K_RISE * np.sign(r1))
                delta_int_res: np.ndarray = (dt / 2.0) * (current_integrand_res + self.last_control_integrand)
                if not self.freeze_int_xy:
                    self.current_integral_control_term[0:2] += delta_int_res[0:2]
                if not self.freeze_int_z:
                    self.current_integral_control_term[2] += delta_int_res[2]
                self.last_control_integrand = current_integrand_res
                u = phi_val + (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term

            case "integrated_resnet":
                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                    u_last: np.ndarray =  (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term
                    kappa_vec: jax.Array = jnp.array(object=np.concatenate((q, q_dot, qd, qd_dot, u_last)))

                    t_start_jax = time.perf_counter()
                    self.theta_hat, phi_out, ball_projected, rate_limited, theta_hat_dot = self.compiled_update_step(
                        theta_hat=self.theta_hat,
                        x_vec=kappa_vec,
                        r1_vec=jnp.array(object=r1),
                        dt=dt,
                        theta_bar=self.theta_bar,
                        theta_dot_bar=self.theta_dot_bar,
                        gamma_diag=self.gamma_diag,
                        s_mod=self.sigma_mod,
                        control_saturated=self.is_control_saturated
                    )
                    self.theta_hat.block_until_ready()
                    t_end_jax = time.perf_counter()
                    jax_dt = t_end_jax - t_start_jax
                    if jax_dt > self.control_period_s:
                        self.get_logger().warning(f"JAX execution took {jax_dt*1000:.2f}ms at t={t:.2f}s.")

                    self._log_theta_saturation(t=t, ball_projected=bool(ball_projected), rate_limited=bool(rate_limited))

                    phi_val = np.array(object=phi_out, dtype=np.float64)
                    theta_hat_dot_val = np.array(object=theta_hat_dot, dtype=np.float64)
                    ball_projected = bool(ball_projected)
                    rate_limited = bool(rate_limited)

                current_integrand_int: np.ndarray = (self.K_I * e) + (self.K_RISE * np.sign(r1)) + phi_val
                delta_int_int: np.ndarray = (dt / 2.0) * (current_integrand_int + self.last_control_integrand)
                if not self.freeze_int_xy:
                    self.current_integral_control_term[0:2] += delta_int_int[0:2]
                if not self.freeze_int_z:
                    self.current_integral_control_term[2] += delta_int_int[2]
                self.last_control_integrand = current_integrand_int
                u = (self.K_P * e) + (self.K_D * e_dot) + self.current_integral_control_term

            case "supertwisting":
                norm_r1: float = float(np.linalg.norm(r1))
                sgn_r1: np.ndarray = np.sign(r1)
                self.st_integral += sgn_r1 * dt
                u = qd_ddot + self.k_2 * np.sqrt(norm_r1) * sgn_r1 + self.k_3 * self.st_integral + self.k_1 * e_dot

        return u, phi_val, theta_hat_dot_val, ball_projected, rate_limited

    def control_timer_callback(self) -> None:
        # Wraps the real tick so *every* code path through it (INIT/TAKEOFF/FOLLOW_TRAJ/
        # PAUSED, including the JAX call and the publish itself) is covered by one
        # end-to-end deadline check -- not just the ResNet forward/backward pass. A tick
        # that runs long enough to eat into PX4's OFFBOARD signal-loss window is a
        # real-time violation regardless of which line inside the tick was slow.
        tick_start_s: float = time.perf_counter()
        self._control_timer_tick()
        elapsed_s: float = time.perf_counter() - tick_start_s
        budget_s: float = CONTROL_TICK_BUDGET_FRACTION * self.control_period_s
        if elapsed_s > budget_s:
            self.publish_offboard_heartbeat = False
            raise ControlLoopOverrunError(
                f"Control tick took {elapsed_s * 1000.0:.2f}ms, exceeding the "
                f"{CONTROL_TICK_BUDGET_FRACTION:.0%} budget of "
                f"{budget_s * 1000.0:.2f}ms (control_period_s={self.control_period_s * 1000.0:.2f}ms)."
            )

    def _control_timer_tick(self) -> None:
        if self.latest_odom is None: return
        current_timestamp_s: float = self.latest_odom.timestamp / 1e6

        match self.experiment_state:
            case ExperimentState.STATE_INIT:
                self.cost_started = False

                # Always stream heartbeats and 0-setpoints in INIT so PX4 accepts Offboard mode and doesn't timeout
                self.publish_offboard_heartbeat = True
                self.publish_trajectory_setpoint_acceleration(ax=0.0, ay=0.0, az=0.0)

                if not self._mode_cmd_seeded:
                    # Defer the first mode-switch/arm attempt by one retry period so PX4
                    # has already seen a handful of streamed setpoints -- an immediate
                    # attempt races the very first setpoint and PX4 will reject the switch.
                    self.last_mode_cmd_time_s = current_timestamp_s
                    self._mode_cmd_seeded = True

                if self.is_gazebo and not self.position_mode_requested:
                    # Recommended PX4 practice: enter OFFBOARD from Position mode, so
                    # that if the vehicle ever drops out of OFFBOARD it falls back to a
                    # stable hover instead of whatever mode it happened to boot into.
                    # param1=1.0 -> MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, param2=3.0 -> PX4 custom main mode POSCTL
                    self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=3.0)
                    self.position_mode_requested = True

                if not self.in_offboard_mode:
                    self.get_logger().info("Waiting for OFFBOARD mode switch...", throttle_duration_sec=2.0)

                    if self.is_gazebo and (current_timestamp_s - self.last_mode_cmd_time_s > self.mode_cmd_retry_period_s):
                        self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                        if not self.is_armed:
                            self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)
                        self.last_mode_cmd_time_s = current_timestamp_s
                else:
                    if self.is_armed:
                        self.get_logger().info(f"ARMED & OFFBOARD validated. Initializing takeoff to z={self.init_z_m_ned:.2f}m (NED).")
                        self.reset_integral()
                        self.experiment_state = ExperimentState.STATE_TAKEOFF
                        self.takeoff_entry_time_s = current_timestamp_s
                    else:
                        # Still waiting for arming to complete!
                        self.get_logger().info("OFFBOARD engaged, waiting for vehicle to arm...", throttle_duration_sec=2.0)
                        if self.is_gazebo and (current_timestamp_s - self.last_mode_cmd_time_s > self.mode_cmd_retry_period_s):
                            self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)
                            self.last_mode_cmd_time_s = current_timestamp_s

            case ExperimentState.STATE_PAUSED:
                if self.in_offboard_mode and self.is_armed:
                    # Pilot re-engaged offboard mode.
                    # We shift t_0 forward by the elapsed paused time so the trajectory completely froze during the dropout
                    time_paused_s: float = current_timestamp_s - self.pause_start_time_s
                    self.t_0 += time_paused_s
                    self.experiment_state = self.pre_pause_state
                    self.get_logger().info("OFFBOARD mode re-engaged. Resuming trajectory seamlessly.")
                else:
                    self.get_logger().info("Trajectory paused. Waiting for pilot to re-engage OFFBOARD...", throttle_duration_sec=2.0)
                return

            case ExperimentState.STATE_TAKEOFF:
                self.publish_offboard_heartbeat = True

                if not self.in_offboard_mode:
                    if self.is_gazebo:
                        self.publish_offboard_heartbeat = False
                        raise FailsafeTriggeredError("PX4 left OFFBOARD mode during SITL simulation (takeoff).")
                    else:
                        self.get_logger().warning("RC pilot intervention detected. Pausing trajectory.", throttle_duration_sec=1.0)
                        self.pre_pause_state = self.experiment_state
                        self.experiment_state = ExperimentState.STATE_PAUSED
                        self.pause_start_time_s = current_timestamp_s
                        self.reset_integral()
                        return

                # Check the takeoff-settled transition before anything else -- the fixed
                # hold target get_desired_state() uses during STATE_TAKEOFF doesn't depend
                # on the trajectory clock, so there's nothing else to update first.
                q: np.ndarray = np.array(object=self.latest_odom.position, dtype=np.float64)
                q_dot: np.ndarray = np.array(object=self.latest_odom.velocity, dtype=np.float64)

                if self.is_gazebo:
                    if (current_timestamp_s - self.takeoff_entry_time_s) > self.takeoff_timeout_s:
                        self.cost_J += self.w_fail * (self.run_length_s ** 2)
                        self.get_logger().info(f"[RESULT] Final cost = {self.cost_J:.4f} (takeoff timeout).")
                        self.publish_offboard_heartbeat = False
                        raise FailsafeTriggeredError("Failed to reach takeoff position within timeout.")

                    e_takeoff: np.ndarray = np.array(object=[self.init_x_m_ned, self.init_y_m_ned, self.init_z_m_ned], dtype=np.float64) - q
                    if np.linalg.norm(e_takeoff) <= self.init_tol_m:
                        self.experiment_state = ExperimentState.STATE_FOLLOW_TRAJ
                        # Reset t_0 so the trajectory clock starts at exactly 0.0 now
                        self.t_0 = current_timestamp_s
                        self.last_t_s = 0.0
                        self.get_logger().info(f"Takeoff settled. Step response triggered: starting trajectory {self.desired_trajectory}.")

                t: float = 0.0
                dt: float = self.control_period_s

                boundary_err: Optional[str] = self.check_safety_boundary(q=q)
                if boundary_err is not None:
                    self.cost_J += self.w_fail * ((self.run_length_s - t) ** 2)
                    self.get_logger().info(f"[RESULT] Final cost = {self.cost_J:.4f} (boundary failure).")
                    self.publish_offboard_heartbeat = False
                    raise BoundaryBreachError(boundary_err)

                qd, qd_dot, qd_ddot = self.get_desired_state(t=t)
                e: np.ndarray = qd - q
                e_dot: np.ndarray = qd_dot - q_dot
                r1: Optional[np.ndarray] = (e_dot + (self.k_1 * e)) if self.controller_type in ['resnet', 'integrated_resnet', 'baseline', 'supertwisting'] else None

                u, phi_val, _, _, _ = self.compute_control_output(
                    q=q, q_dot=q_dot, qd=qd, qd_dot=qd_dot, qd_ddot=qd_ddot, e=e, e_dot=e_dot, r1=r1, dt=dt, t=t
                )

                self.is_control_saturated = False
                self.freeze_int_xy = False
                self.freeze_int_z = False

                u_xy: np.ndarray = u[0:2]
                norm_uxy: float = float(np.linalg.norm(u_xy))
                if norm_uxy > self.acc_hor_max_mps2:
                    u[0:2] = u_xy * (self.acc_hor_max_mps2 / norm_uxy)
                    self.is_control_saturated = True
                    if np.dot(a=e[0:2], b=u[0:2]) > 0.0:
                        self.freeze_int_xy = True
                    self.get_logger().debug(f"XY saturation at t={t:.2f}s.")

                if abs(u[2]) > self.acc_vert_max_mps2:
                    u[2] = self.acc_vert_max_mps2 * np.sign(u[2])
                    self.is_control_saturated = True
                    if np.sign(e[2]) == np.sign(u[2]):
                        self.freeze_int_z = True
                    self.get_logger().debug(f"Z saturation at t={t:.2f}s.")

                self.publish_trajectory_setpoint_acceleration(ax=u[0], ay=u[1], az=u[2])

            case ExperimentState.STATE_FOLLOW_TRAJ:
                self.publish_offboard_heartbeat = True

                if not self.in_offboard_mode:
                    if self.is_gazebo:
                        self.publish_offboard_heartbeat = False
                        raise FailsafeTriggeredError("PX4 left OFFBOARD mode during SITL simulation (following trajectory).")
                    else:
                        self.get_logger().warning("RC pilot intervention detected. Pausing trajectory.", throttle_duration_sec=1.0)
                        self.pre_pause_state = self.experiment_state
                        self.experiment_state = ExperimentState.STATE_PAUSED
                        self.pause_start_time_s = current_timestamp_s
                        self.reset_integral()
                        return

                q: np.ndarray = np.array(object=self.latest_odom.position, dtype=np.float64)
                t: float = current_timestamp_s - self.t_0
                dt: float = t - self.last_t_s

                q_dot: np.ndarray = np.array(object=self.latest_odom.velocity, dtype=np.float64)

                boundary_err: Optional[str] = self.check_safety_boundary(q=q)
                if boundary_err is not None:
                    self.cost_J += self.w_fail * ((self.run_length_s - t) ** 2)
                    self.get_logger().info(f"[RESULT] Final cost = {self.cost_J:.4f} (boundary failure).")
                    self.publish_offboard_heartbeat = False
                    raise BoundaryBreachError(boundary_err)

                qd, qd_dot, qd_ddot = self.get_desired_state(t=t)
                e: np.ndarray = qd - q
                e_dot: np.ndarray = qd_dot - q_dot
                r1: Optional[np.ndarray] = (e_dot + (self.k_1 * e)) if self.controller_type in ['resnet', 'integrated_resnet', 'baseline', 'supertwisting'] else None

                # Snapshot theta_hat as it stood when this tick's control was computed, before
                # compute_control_output's internal compiled_update_step call reassigns it --
                # matches the real-hardware CSV convention (weight_history[i] is the weight
                # actually driving tick i's phi_val/control output, not the post-update value
                # that only takes effect starting next tick). JAX arrays are immutable, so this
                # plain reference is already an independent snapshot.
                theta_hat_at_tick = self.theta_hat if self.controller_type in ["resnet", "integrated_resnet"] else None

                u, phi_val, theta_hat_dot_val, ball_projected, rate_limited = self.compute_control_output(
                    q=q, q_dot=q_dot, qd=qd, qd_dot=qd_dot, qd_ddot=qd_ddot, e=e, e_dot=e_dot, r1=r1, dt=dt, t=t
                )

                norm_e: float = float(np.linalg.norm(e))
                norm_u: float = float(np.linalg.norm(u))

                self.time_history.append(t)
                self.error_norm_history.append(norm_e)
                self.control_output_norm_history.append(norm_u)
                self.control_output_history.append(u.tolist())
                self.q_history.append(q.tolist())
                self.qd_history.append(qd.tolist())

                if self.controller_type in ["resnet", "integrated_resnet"]:
                    self.weight_history.append(np.array(object=theta_hat_at_tick).flatten().tolist())
                    self.phi_history.append(phi_val.tolist())
                    self.theta_hat_norm_history.append(float(np.linalg.norm(np.array(object=theta_hat_at_tick))))
                    self.theta_hat_dot_norm_history.append(float(np.linalg.norm(theta_hat_dot_val)) if theta_hat_dot_val is not None else 0.0)
                    self.ball_projected_history.append(bool(ball_projected))
                    self.rate_limited_history.append(bool(rate_limited))

                current_error_sq: float = float(norm_e ** 2)
                current_u_sq: float = float(norm_u ** 2)

                if not self.cost_started or dt <= 0:
                    # No previous sample to difference against yet (first tick of the
                    # trajectory) - contribute zero jerk rather than a spurious spike.
                    u_dot: np.ndarray = np.zeros(shape=3, dtype=np.float64)
                    current_u_dot_sq: float = 0.0
                else:
                    u_dot = (u - self.last_u) / dt
                    current_u_dot_sq = float(np.dot(u_dot, u_dot))

                self.u_dot_history.append(u_dot.tolist())

                current_cost_integrand: float = (
                    (self.q_e * current_error_sq) + (self.r_u * current_u_sq) + (self.r_udot * current_u_dot_sq)
                )

                if not self.cost_started:
                    # Seed the history at exact start to prevent trapezoidal integration jump
                    self.last_error_sq = current_error_sq
                    self.last_u_sq = current_u_sq
                    self.last_u_dot_sq = current_u_dot_sq
                    self.last_cost_integrand = current_cost_integrand
                    self.cost_started = True

                self.error_sq_integral += (dt / 2.0) * (current_error_sq + self.last_error_sq)
                self.last_error_sq = current_error_sq

                self.u_sq_integral += (dt / 2.0) * (current_u_sq + self.last_u_sq)
                self.last_u_sq = current_u_sq

                self.u_dot_sq_integral += (dt / 2.0) * (current_u_dot_sq + self.last_u_dot_sq)
                self.last_u_dot_sq = current_u_dot_sq

                self.cost_J += (dt / 2.0) * (current_cost_integrand + self.last_cost_integrand)
                self.last_cost_integrand = current_cost_integrand

                self.last_u = u.copy()
                self.last_t_s = t

                self.is_control_saturated = False
                self.freeze_int_xy = False
                self.freeze_int_z = False

                u_xy: np.ndarray = u[0:2]
                norm_uxy: float = float(np.linalg.norm(u_xy))
                if norm_uxy > self.acc_hor_max_mps2:
                    u[0:2] = u_xy * (self.acc_hor_max_mps2 / norm_uxy)
                    self.is_control_saturated = True
                    if np.dot(a=e[0:2], b=u[0:2]) > 0.0:
                        self.freeze_int_xy = True
                    self.get_logger().debug(f"XY saturation at t={t:.2f}s.")

                if abs(u[2]) > self.acc_vert_max_mps2:
                    u[2] = self.acc_vert_max_mps2 * np.sign(u[2])
                    self.is_control_saturated = True
                    if np.sign(e[2]) == np.sign(u[2]):
                        self.freeze_int_z = True
                    self.get_logger().debug(f"Z saturation at t={t:.2f}s.")

                self.publish_trajectory_setpoint_acceleration(ax=u[0], ay=u[1], az=u[2])

                if t >= self.run_length_s:
                    rms_error: float = math.sqrt(self.error_sq_integral / self.run_length_s) if self.run_length_s > 0 else 0.0
                    rms_u: float = math.sqrt(self.u_sq_integral / self.run_length_s) if self.run_length_s > 0 else 0.0
                    rms_u_dot: float = math.sqrt(self.u_dot_sq_integral / self.run_length_s) if self.run_length_s > 0 else 0.0
                    self.get_logger().info(f"[RESULT] Final cost = {self.cost_J:.2f}.")
                    self.get_logger().info(f"[RESULT] RMS error = {rms_error:.4f}.")
                    self.get_logger().info(f"[RESULT] RMS control effort = {rms_u:.3f}.")
                    self.get_logger().info(f"[RESULT] RMS control jerk = {rms_u_dot:.3f}.")
                    raise ExperimentFinished("Trajectory completed successfully.")

def main(args: Optional[List[str]] = None) -> None:
    # The cyclic GC is a latency-jitter source we don't need: this process runs one
    # bounded experiment and exits, so there's no long-run leak risk to guard against,
    # and refcounting alone still reclaims everything that isn't part of a reference
    # cycle. Disabling it removes an unpredictable stop-the-world pause from the hot
    # control loop, where a single missed control_period_s can bleed into PX4's
    # COM_OF_LOSS_T offboard-signal-loss window.
    gc.disable()

    rclpy.init(args=args)
    node: AviaryRiseNode = AviaryRiseNode()
    try:
        rclpy.spin(node=node)
    except ExperimentFinished as e:
        node.get_logger().info(f"Experiment terminated: {e}")
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received.")
    except ValueError as e:
        node.get_logger().fatal(f"Value error: {e}")
    except JaxLatencyError as e:
        node.get_logger().fatal(f"Hardware error: {e}")
    except OdomTimeoutError as e:
        node.get_logger().fatal(f"Odometry timeout: {e}")
    except FailsafeTriggeredError as e:
        node.get_logger().fatal(f"Failsafe triggered: {e}")
    except BoundaryBreachError as e:
        node.get_logger().fatal(f"Boundary breach: {e}")
    except ControlLoopOverrunError as e:
        node.get_logger().fatal(f"Control loop overrun: {e}")
    finally:
        node.get_logger().info("Commanding vehicle to land.")
        node.land_vehicle()
        if rclpy.ok():
            if node.save_data:
                node.get_logger().info("Saving telemetry data to CSV...")
                node.write_csv()

            print("[INFO] Node cleanly destroyed.")
        else:
            print("[FATAL] Node not cleanly destroyed.")

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
