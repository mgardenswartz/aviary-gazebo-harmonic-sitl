import os
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
from aviary_rise_controller.proj import discrete_projection
from aviary_rise_controller.desired_trajectory import TrajectoryGenerator

class ExperimentState:
    STATE_INIT: int = 0
    STATE_TAKEOFF: int = 1
    STATE_FOLLOW_TRAJ: int = 2
    STATE_PAUSED: int = 3

class CriticalHardwareError(Exception):
    pass

class OdomTimeoutError(Exception):
    pass

class FailsafeTriggeredError(Exception):
    pass

class BoundaryBreachError(Exception):
    pass

class AviaryRiseNode(Node):
    def __init__(self) -> None:
        super().__init__(
            node_name='aviary_rise_node',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )

        # Basic Parameters of the Experiment
        self.is_gazebo: bool = self.get_parameter(name='is_gazebo').value
        self.desired_trajectory: int = self.get_parameter(name='desired_trajectory').value
        self.vehicle_name: str = self.get_parameter(name='vehicle_name').value
        self.controller_type: str = self.get_parameter(name='controller_type').value
        control_frequency_hz: float = self.get_parameter(name='control_frequency_hz').value
        self.control_period_s: float = 1.0 / control_frequency_hz
        self.save_data: bool = self.get_parameter(name='save_data').value
        self.trial_number: Optional[int] = self.get_parameter(name='trial_number').value if self.has_parameter('trial_number') else None
        self.run_length_s: float = self.get_parameter(name='run_length_s').value
        self.init_tol_m: float = self.get_parameter(name='init_tol_m').value
        self.d_out: int = self.get_parameter(name='d_out').value

        # Desired Trajectory
        if self.desired_trajectory not in [1,2]: 
            raise ValueError("INVALID DESIRED TRAJECTORY SELECTED.")
        # Fix: Convert parameters to primitive values for config
        self.config: Dict[str, Any] = {k: v.value for k, v in self.get_parameters_by_prefix(prefix='').items()}
        self.traj_gen: TrajectoryGenerator = TrajectoryGenerator(config=self.config)
        
        # Safety
        self.acc_hor_max_mps2: float = self.get_parameter(name='mpc_acc_hor_max_mps2').value
        self.acc_vert_max_mps2: float = self.get_parameter(name='mpc_acc_vert_max_mps2').value
        self.safe_x_min_m: float = self.get_parameter(name='safe_x_min_m_ned').value
        self.safe_x_max_m: float = self.get_parameter(name='safe_x_max_m_ned').value
        self.safe_y_min_m: float = self.get_parameter(name='safe_y_min_m_ned').value
        self.safe_y_max_m: float = self.get_parameter(name='safe_y_max_m_ned').value
        self.safe_z_min_m: float = self.get_parameter(name='safe_z_min_m_ned').value
        self.safe_z_max_m: float = self.get_parameter(name='safe_z_max_m_ned').value
        self.odom_timeout_s: float = self.get_parameter(name='odom_timeout_s').value
        self.init_z_m_ned: float = self.get_parameter(name='init_z_m_ned').value
        self.odom_watchdog_freq: float = self.get_parameter(name='odom_watchdog_freq').value
        
        # Cost Function (same formula used for post-hoc gain selection in
        # unified_orchestrator.py's compute_trial_J - no t-weighting on tracking error)
        self.q_e: float = self.get_parameter(name='q_e').value
        self.r_u: float = self.get_parameter(name='r_u').value
        self.r_udot: float = self.get_parameter(name='r_udot').value
        self.w_fail: float = self.get_parameter(name='w_fail').value

        if self.controller_type == "pid":
            self.K_P: float = self.get_parameter(name='K_P').value
            self.K_I: float = self.get_parameter(name='K_I').value
            self.K_D: float = self.get_parameter(name='K_D').value
    
        elif self.controller_type in ['baseline', 'integrated_resnet', 'resnet', 'supertwisting']:
            self.k_1: float = self.get_parameter(name='k_1').value
            self.k_2: float = self.get_parameter(name='k_2').value
            self.k_3: float = self.get_parameter(name='k_3').value

            if self.controller_type in ['baseline', 'integrated_resnet', 'resnet']:
                self.K_RISE: float = self.get_parameter(name='k_rise').value
                self.K_P: float = (self.k_1 * self.k_2) + (self.k_1 * self.k_3) + (self.k_2 * self.k_3) + 1.0
                self.K_I: float = (self.k_1 * self.k_2 * self.k_3) + self.k_1
                self.K_D: float = self.k_1 + self.k_2 + self.k_3

            if self.controller_type in ["resnet", "integrated_resnet"]:
                self.d_in: int = self.get_parameter(name='d_in').value
                
                self.theta_hat: jax.Array = jnp.array(object=self.get_parameter(name='initial_weights').value)
                
                self.gamma_diag: jax.Array = jnp.ones(shape=self.theta_hat.shape[0]) * self.get_parameter(name='gamma').value
                self.sigma_mod: float = self.get_parameter(name='sigma_mod').value
                self.theta_bar: float = self.get_parameter(name='theta_bar').value

                self.bound_resnet = jax.jit(partial(
                    resnet_network,
                    d_in=self.d_in,
                    hidden_width=self.get_parameter(name='hidden_width').value,
                    d_out=self.d_out,
                    b=self.get_parameter(name='num_blocks').value,
                    k_0=self.get_parameter(name='k_0').value,
                    k_i=self.get_parameter(name='k_i').value,
                    h_act_func=self.get_parameter(name='h_act_func').value,
                    o_act_func=self.get_parameter(name='o_act_func').value,
                    shortcut_act_func=self.get_parameter(name='shortcut_act_func').value,
                ))
            
                @jax.jit
                def compiled_update_step(theta_hat: jax.Array, x_vec: jax.Array, r1_vec: jax.Array, dt: float, theta_bar: float, gamma_diag: jax.Array, s_mod: float, saturated: bool) -> Tuple[jax.Array, jax.Array]:
                    phi_val, vjp_fn = jax.vjp(lambda t: self.bound_resnet(t, x_vec), has_aux=False, *[theta_hat])
                    grad_term = vjp_fn(r1_vec)[0]
                    theta_dot_unprojected = gamma_diag * (grad_term - s_mod * theta_hat)
                    theta_next = discrete_projection(theta_hat=theta_hat, theta_dot_unprojected=theta_dot_unprojected, dt=dt, theta_bar=theta_bar, gamma_diag=gamma_diag)
                    final_theta = jax.lax.select(pred=saturated, on_true=theta_hat, on_false=theta_next)
                    return final_theta, phi_val
                    
                self.compiled_update_step = compiled_update_step
                self.precompile_jax()
        
        # For VehicleStatus callback
        self.nav_state: int = 0
        self.vehicle_system_id: int = 1
        self.vehicle_component_id: int = 1

        # --- Aggressive diagnostics for the offboard-loss investigation ---
        # Not throttled: every nav_state/arming_state/failsafe transition is logged
        # immediately, since these are rare and each one is exactly the evidence we need.
        self._diag_last_nav_state: Optional[int] = None
        self._diag_last_arming_state: Optional[int] = None
        self._diag_last_failsafe: Optional[bool] = None

        # Wall-clock (not sim-time) instrumentation of control_timer_callback: measures both
        # the scheduling gap (time since this callback last actually ran, regardless of what
        # it did) and the processing duration (time spent inside the callback body), logged
        # separately so a slow rclpy executor (scheduling gap) can be told apart from slow
        # Python code inside the callback (processing duration).
        self._diag_last_tick_wall: Optional[float] = None
        self._diag_tick_warn_threshold_s: float = 2.0 * self.control_period_s

        # odom_callback rate counter, reported once per second via the watchdog timer --
        # PX4's vehicle_odometry publish rate is uncontrolled by us, and if it's high enough
        # it can starve rclpy's (single-threaded, by default) executor of time for
        # control_timer_callback, which would look exactly like "the control loop is slow"
        # without actually being slow code.
        self._diag_odom_count: int = 0
        self._diag_odom_count_window_start: Optional[float] = None

        # Independent canary timer: does nothing but record its own scheduling interval. If
        # control_timer_callback's measured gaps track this canary's gaps, the executor itself
        # is being starved (by odom_callback volume, GIL contention, etc.) -- not the control
        # loop's own code. If the canary stays clean while control_timer_callback stalls, the
        # opposite is true.
        self._diag_canary_last: Optional[float] = None
        self._diag_canary_intervals: List[float] = []
        self._diag_canary_timer = self.create_timer(timer_period_sec=0.05, callback=self.diag_canary_callback)

        # Init
        self.is_armed: bool = False
        self.in_offboard_mode: bool = False
        self.landing_command_sent: bool = False  
        self.cost_started: bool = False
        self.is_saturated: bool = False
        self.freeze_int_xy: bool = False
        self.freeze_int_z: bool = False
        self.initial_position_locked: bool = False
        self.latest_odom: Optional[VehicleOdometry] = None

        self.last_odom_ros_time: float = 0.0
        self.start_x: float = 0.0
        self.start_y: float = 0.0
        self.experiment_state: int = ExperimentState.STATE_INIT
        self.t_0: float = 0.0

        self.last_t: float = 0.0
        self.last_auto_cmd_time: float = 0.0
        self.pause_start_time: float = 0.0
        self.takeoff_start_time: float = 0.0
        self.pre_pause_state: int = ExperimentState.STATE_INIT
        
        self.ticks_without_odom: int = 0
        self.reset_integral_terms()
        self.cost_J: float = 0.0
        self.last_cost_integrand: float = 0.0
        self.error_sq_integral: float = 0.0
        self.last_error_sq: float = 0.0
        self.u_sq_integral: float = 0.0
        self.last_u_sq: float = 0.0
        self.last_u: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
        self.time_history: List[float] = []
        self.control_output_norm_history: List[float] = []
        self.control_output_history: List[List[float]] = []
        self.error_norm_history: List[float] = []
        self.weight_history: List[List[float]] = []
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

        self.odom_watchdog_timer = self.create_timer(timer_period_sec=1.0/self.odom_watchdog_freq, callback=self.odom_watchdog_callback)
        
        self.get_logger().info(f"Node Booted. Controller: {self.controller_type.upper()} | Trajectory: {self.desired_trajectory} | Gazebo Mode: {self.is_gazebo}")

    def precompile_jax(self) -> None:
        dummy_x: jax.Array = jnp.zeros(shape=self.d_in)
        dummy_r1: jax.Array = jnp.zeros(shape=self.d_out)
        self.get_logger().info("[JAX] Compiling XLA Graph on CPU...")
        
        self.theta_hat, _ = self.compiled_update_step(
            theta_hat=self.theta_hat,
            x_vec=dummy_x,
            r1_vec=dummy_r1,
            dt=self.control_period_s,
            theta_bar=self.theta_bar,
            gamma_diag=self.gamma_diag,
            s_mod=self.sigma_mod,
            saturated=False # I've never tested True...
        )
        self.theta_hat.block_until_ready()
        
        start_time: float = time.perf_counter()
        self.theta_hat, _ = self.compiled_update_step(
            theta_hat=self.theta_hat,
            x_vec=dummy_x,
            r1_vec=dummy_r1,
            dt=self.control_period_s,
            theta_bar=self.theta_bar,
            gamma_diag=self.gamma_diag,
            s_mod=self.sigma_mod,
            saturated=False
        )
        self.theta_hat.block_until_ready()
        hot_time: float = time.perf_counter() - start_time
        
        # Reset the weights back to true initial conditions
        self.theta_hat = jnp.array(object=self.get_parameter(name='initial_weights').value)
        self.theta_hat.block_until_ready()
        self.get_logger().info(f"[JAX] Neural Network Latency: {hot_time*1000:.2f} ms")
        if hot_time > self.control_period_s:
            self.get_logger().fatal(f"[ERROR] Execution time {hot_time}s exceeds {self.control_period_s}s limit!")
            raise CriticalHardwareError("[JAX] ResNet latency too high for selected control frequency (init).")

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        self.nav_state = msg.nav_state
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_offboard_mode = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.vehicle_system_id = msg.system_id
        self.vehicle_component_id = msg.component_id

        # Diagnostics: log every nav_state/arming_state/failsafe transition immediately
        # (not throttled) -- this is PX4's own self-reported state, straight from
        # VehicleStatus, independent of anything this node infers. nav_state_user_intention
        # shows what mode PX4 still thinks we *want* (should stay OFFBOARD=14 the whole time
        # if this node keeps commanding it) vs nav_state, the mode PX4 is *actually* in --
        # a divergence between the two is exactly what a failsafe fallback looks like.
        nav_changed = msg.nav_state != self._diag_last_nav_state
        arm_changed = msg.arming_state != self._diag_last_arming_state
        fs_changed = msg.failsafe != self._diag_last_failsafe
        if nav_changed or arm_changed or fs_changed:
            wall_now = time.perf_counter()
            self.get_logger().warn(
                f"[DIAG] VehicleStatus transition: nav_state {self._diag_last_nav_state}->{msg.nav_state} "
                f"(user_intention={msg.nav_state_user_intention}), "
                f"arming_state {self._diag_last_arming_state}->{msg.arming_state}, "
                f"failsafe {self._diag_last_failsafe}->{msg.failsafe} "
                f"(failsafe_and_user_took_over={msg.failsafe_and_user_took_over}), "
                f"nav_state_timestamp={msg.nav_state_timestamp}, wall_now={wall_now:.3f}"
            )
            self._diag_last_nav_state = msg.nav_state
            self._diag_last_arming_state = msg.arming_state
            self._diag_last_failsafe = msg.failsafe

    def odom_callback(self, msg: VehicleOdometry) -> None:
        self.latest_odom = msg
        self.ticks_without_odom = 0
        self._diag_odom_count += 1
        if not self.is_gazebo:
            self.last_odom_ros_time = self.get_clock().now().nanoseconds / 1e9

        if not self.initial_position_locked:
            self.start_x = float(msg.position[0])
            self.start_y = float(msg.position[1])
            self.initial_position_locked = True

    def diag_canary_callback(self) -> None:
        # Does nothing but measure its own scheduling interval. Compare against
        # control_timer_callback's measured gaps: if both stall together, the rclpy executor
        # itself is being starved (e.g. by odom_callback volume -- see the rate logged below --
        # or GIL contention); if only the control timer stalls, the problem is in its own body.
        now = time.perf_counter()
        if self._diag_canary_last is not None:
            self._diag_canary_intervals.append(now - self._diag_canary_last)
        self._diag_canary_last = now

    def odom_watchdog_callback(self) -> None:
        self.ticks_without_odom += 1

        # Diagnostics: report actual odom_callback rate and canary-timer scheduling jitter
        # once per second (this callback already runs at odom_watchdog_freq, default 10Hz).
        now = time.perf_counter()
        if self._diag_odom_count_window_start is None:
            self._diag_odom_count_window_start = now
        elif (now - self._diag_odom_count_window_start) >= 1.0:
            window_s = now - self._diag_odom_count_window_start
            rate_hz = self._diag_odom_count / window_s
            canary_msg = ""
            if self._diag_canary_intervals:
                ci = self._diag_canary_intervals
                canary_msg = (
                    f" | canary(20Hz) interval min/mean/max(ms)="
                    f"{min(ci)*1000:.1f}/{(sum(ci)/len(ci))*1000:.1f}/{max(ci)*1000:.1f} n={len(ci)}"
                )
            self.get_logger().warn(
                f"[DIAG] odom_callback rate: {rate_hz:.1f} Hz over {window_s:.2f}s ({self._diag_odom_count} msgs){canary_msg}"
            )
            self._diag_odom_count = 0
            self._diag_odom_count_window_start = now
            self._diag_canary_intervals = []

        if not self.initial_position_locked:
            if self.ticks_without_odom >= (self.odom_timeout_s * self.odom_watchdog_freq):
                raise OdomTimeoutError("No odometry received at boot.")
        else:
            if self.is_gazebo:
                if self.ticks_without_odom >= (self.odom_timeout_s * self.odom_watchdog_freq):
                    raise OdomTimeoutError("Simulation running behind schedule.")
            else:
                # Use original wall clock logic for real vehicle (sim-to-real)
                current_time: float = self.get_clock().now().nanoseconds / 1e9
                if (current_time - self.last_odom_ros_time) > self.odom_timeout_s:
                    raise OdomTimeoutError("Odometry feed lost during flight.")

    def reset_integral_terms(self) -> None:
        self.current_integral_control_term = np.zeros(shape=self.d_out, dtype=np.float64)
        self.last_control_integrand = np.zeros(shape=self.d_out, dtype=np.float64)
        self.st_integral = np.zeros(shape=self.d_out, dtype=np.float64)

    def publish_vehicle_command(self, command: int, param1: float, param2: float) -> None:
        self.get_logger().info(f"[DEBUG] Publishing command {command}")
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

    def publish_offboard_heartbeat(self) -> None:
        msg: OffboardControlMode = OffboardControlMode()
        msg.timestamp = int(self.latest_odom.timestamp) if self.latest_odom is not None else int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = True
        msg.attitude = False
        msg.body_rate = False
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint_acceleration(self, ax: float, ay: float, az: float) -> None:
        msg: TrajectorySetpoint = TrajectorySetpoint()
        msg.acceleration = [ax, ay, az]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.yaw = 0.0  # Command a heading of 0.0 always
        if self.latest_odom is not None:
            msg.timestamp = self.latest_odom.timestamp
        self.trajectory_setpoint_publisher.publish(msg)

    def log_csv(self) -> None:
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
                    "x", "y", "z", "xd", "yd", "zd"
                ]
                if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                    num_weights: int = len(self.weight_history[0])
                    headers += [f"W{i}" for i in range(num_weights)]
                writer.writerow(headers)
                for i in range(len(self.time_history)):
                    row: List[float] = [
                        self.time_history[i], self.error_norm_history[i], self.control_output_norm_history[i],
                        self.control_output_history[i][0], self.control_output_history[i][1], self.control_output_history[i][2],
                        self.q_history[i][0], self.q_history[i][1], self.q_history[i][2],
                        self.qd_history[i][0], self.qd_history[i][1], self.qd_history[i][2]
                    ]
                    if self.controller_type in ["resnet", "integrated_resnet"] and self.weight_history:
                        row += self.weight_history[i]
                    writer.writerow(row)
            self.get_logger().info(f"Telemetry saved to {csv_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV: {e}")

    def check_safety_boundary(self, q: np.ndarray) -> Optional[str]:
        if not (self.safe_x_min_m <= q[0] <= self.safe_x_max_m):
            return f"X position {q[0]:.2f} breached bounds [{self.safe_x_min_m}, {self.safe_x_max_m}]."
        if not (self.safe_y_min_m <= q[1] <= self.safe_y_max_m):
            return f"Y position {q[1]:.2f} breached bounds [{self.safe_y_min_m}, {self.safe_y_max_m}]."
        if not (self.safe_z_min_m <= q[2] <= self.safe_z_max_m):
            return f"Z position {q[2]:.2f} breached bounds [{self.safe_z_min_m}, {self.safe_z_max_m}]."
        return None

    def get_desired_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.experiment_state == ExperimentState.STATE_TAKEOFF:
            # During takeoff, hold exactly above where it initialized
            return (np.array(object=[self.start_x, self.start_y, self.init_z_m_ned], dtype=np.float64), 
                    np.zeros(shape=3, dtype=np.float64), np.zeros(shape=3, dtype=np.float64))
            
        return self.traj_gen.get_desired_state(t=t)

    def control_timer_callback(self) -> None:
        # Diagnostics wrapper: measures the wall-clock SCHEDULING gap since this timer last
        # actually fired (regardless of what it did last time) and the PROCESSING duration of
        # this invocation, logged separately. A scheduling gap means rclpy's executor didn't
        # get back to this timer on time (starved by odom_callback volume, GIL contention,
        # etc.) even though the callback body itself may be fast; a processing-duration warning
        # means the callback body itself is slow. Real logic is in _control_timer_tick below.
        wall_now: float = time.perf_counter()
        if self._diag_last_tick_wall is not None:
            gap: float = wall_now - self._diag_last_tick_wall
            if gap > self._diag_tick_warn_threshold_s:
                self.get_logger().warn(
                    f"[DIAG] control_timer_callback SCHEDULING GAP: {gap*1000:.1f}ms since last tick "
                    f"(expected ~{self.control_period_s*1000:.1f}ms, warn threshold "
                    f"{self._diag_tick_warn_threshold_s*1000:.1f}ms). state={self.experiment_state}"
                )
        self._diag_last_tick_wall = wall_now

        try:
            self._control_timer_tick()
        finally:
            duration: float = time.perf_counter() - wall_now
            if duration > self.control_period_s:
                self.get_logger().warn(
                    f"[DIAG] control_timer_callback PROCESSING TOOK {duration*1000:.1f}ms "
                    f"(control_period_s={self.control_period_s*1000:.1f}ms). state={self.experiment_state}"
                )

    def _control_timer_tick(self) -> None:
        if self.latest_odom is None: return
        current_timestamp_s: float = self.latest_odom.timestamp / 1e6

        match self.experiment_state:
            case ExperimentState.STATE_INIT:
                self.landing_command_sent = False
                self.cost_started = False

                # Always stream heartbeats and 0-setpoints in INIT so PX4 accepts Offboard mode and doesn't timeout
                self.publish_offboard_heartbeat()
                self.publish_trajectory_setpoint_acceleration(ax=0.0, ay=0.0, az=0.0)

                match self.in_offboard_mode:
                    case False:
                        self.get_logger().info("Waiting for PX4 Offboard Mode switch engagement...", throttle_duration_sec=2.0)

                        if self.is_gazebo and (current_timestamp_s - self.last_auto_cmd_time > 1.0):
                            self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                            if not self.is_armed:
                                self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)
                            self.last_auto_cmd_time = current_timestamp_s
                    
                    case True:
                        if self.is_armed:
                            self.get_logger().info(f"ARMED & OFFBOARD validated. Initializing Takeoff to Z={self.init_z_m_ned}.")
                            self.reset_integral_terms()
                            self.experiment_state = ExperimentState.STATE_TAKEOFF
                            self.takeoff_start_time = current_timestamp_s
                        else:
                            # Still waiting for arming to complete!
                            self.get_logger().info("Offboard engaged, waiting for vehicle to arm...", throttle_duration_sec=2.0)
                            if self.is_gazebo and (current_timestamp_s - self.last_auto_cmd_time > 1.0):
                                self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)
                                self.last_auto_cmd_time = current_timestamp_s


            case ExperimentState.STATE_PAUSED:
                if self.in_offboard_mode and self.is_armed:
                    # Pilot re-engaged offboard mode. 
                    # We shift t_0 forward by the elapsed paused time so the trajectory completely froze during the dropout
                    time_paused: float = current_timestamp_s - self.pause_start_time
                    self.t_0 += time_paused  
                    self.experiment_state = self.pre_pause_state
                    self.get_logger().info("Offboard Mode re-engaged! Resuming trajectory seamlessly.")
                else:
                    self.get_logger().info("Trajectory Paused. Waiting for Pilot to re-engage Offboard...", throttle_duration_sec=2.0)
                return

            case ExperimentState.STATE_FOLLOW_TRAJ | ExperimentState.STATE_TAKEOFF:
                self.publish_offboard_heartbeat()

                if not self.in_offboard_mode:
                    if self.is_gazebo:
                        self.get_logger().warn(
                            f"[DIAG] Detected offboard loss at raise site: nav_state={self.nav_state}, "
                            f"is_armed={self.is_armed}, wall={time.perf_counter():.3f}, "
                            f"latest_odom.timestamp={self.latest_odom.timestamp if self.latest_odom is not None else None}, "
                            f"experiment_state={self.experiment_state}"
                        )
                        raise FailsafeTriggeredError("PX4 left offboard mode during SITL simulation.")
                    else:
                        self.get_logger().warn("RC pilot intervention detected. Pausing trajectory.", throttle_duration_sec=1.0)
                        self.pre_pause_state = self.experiment_state
                        self.experiment_state = ExperimentState.STATE_PAUSED
                        self.pause_start_time = current_timestamp_s
                        
                        self.reset_integral_terms()
                        return
                
                # Check transitions before updating the clock if in TAKEOFF
                q: np.ndarray = np.array(object=self.latest_odom.position, dtype=np.float64)
                if self.is_gazebo:
                    if self.experiment_state == ExperimentState.STATE_TAKEOFF:
                        if (current_timestamp_s - self.takeoff_start_time) > 20.0:
                            self.cost_J += self.w_fail * (self.run_length_s ** 2)
                            self.get_logger().info(f"[RESULT] Final Cost = {self.cost_J:.4f} (Takeoff Timeout)")
                            raise FailsafeTriggeredError("Failed to reach takeoff position within timeout.")
                            
                        e_takeoff: np.ndarray = np.array(object=[self.start_x, self.start_y, self.init_z_m_ned], dtype=np.float64) - q
                        if np.linalg.norm(e_takeoff) <= self.init_tol_m:
                            self.experiment_state = ExperimentState.STATE_FOLLOW_TRAJ
                            # Reset t_0 so the trajectory clock starts at exactly 0.0 now
                            self.t_0 = current_timestamp_s
                            self.last_t = 0.0
                            self.get_logger().info(f"TAKEOFF SETTLED. Step Response Triggered: Starting Trajectory {self.desired_trajectory}.")

                # If in TAKEOFF, t_0 hasn't been set to the trajectory clock yet, so t evaluates to arbitrary.
                # However get_desired_state(t) strictly ignores t during TAKEOFF.
                t: float = current_timestamp_s - self.t_0 if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ else 0.0
                dt: float = t - self.last_t if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ else self.control_period_s
                
                q_dot: np.ndarray = np.array(object=self.latest_odom.velocity, dtype=np.float64)
                
                boundary_err: Optional[str] = self.check_safety_boundary(q=q)
                if boundary_err is not None:
                    # t is already 0.0 here if still in TAKEOFF (see above), so this correctly
                    # applies the same zero-credit penalty as a takeoff timeout when the
                    # breach happens before the trajectory ever starts, instead of silently
                    # skipping the cost/print and getting mislabeled as a SITL/boot failure.
                    self.cost_J += self.w_fail * ((self.run_length_s - t) ** 2)
                    self.get_logger().info(f"[RESULT] Final Cost = {self.cost_J:.4f} (Boundary Failure)")
                    raise BoundaryBreachError(boundary_err)

                qd: np.ndarray
                qd_dot: np.ndarray
                qd_ddot: np.ndarray
                qd, qd_dot, qd_ddot = self.get_desired_state(t=t)
                e: np.ndarray = qd - q
                e_dot: np.ndarray = qd_dot - q_dot
                if self.controller_type in ['resnet', 'integrated_resnet', 'baseline', 'supertwisting']:
                    r1: np.ndarray = e_dot + (self.k_1 * e)

                u: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
                phi_val: np.ndarray = np.zeros(shape=self.d_out, dtype=np.float64)
                
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
                            self.theta_hat, phi_out = self.compiled_update_step(
                                theta_hat=self.theta_hat,
                                x_vec=x_vec, 
                                r1_vec=jnp.array(object=r1),
                                dt=dt,
                                theta_bar=self.theta_bar,
                                gamma_diag=self.gamma_diag,
                                s_mod=self.sigma_mod,
                                saturated=self.is_saturated
                            )
                            self.theta_hat.block_until_ready()
                            t_end_jax: float = time.perf_counter()
                            jax_dt: float = t_end_jax - t_start_jax
                            if jax_dt > self.control_period_s:
                                self.get_logger().warn(f"[DEBUG] JAX Execution took {jax_dt*1000:.2f} ms at t={t:.2f}s!")
                                
                            phi_val = np.array(object=phi_out, dtype=np.float64)
                        
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
                            self.theta_hat, phi_out = self.compiled_update_step(
                                theta_hat=self.theta_hat,
                                x_vec=kappa_vec,
                                r1_vec=jnp.array(object=r1),
                                dt=dt,
                                theta_bar=self.theta_bar,
                                gamma_diag=self.gamma_diag,
                                s_mod=self.sigma_mod,
                                saturated=self.is_saturated
                            )
                            self.theta_hat.block_until_ready()
                            t_end_jax = time.perf_counter()
                            jax_dt = t_end_jax - t_start_jax
                            if jax_dt > self.control_period_s:
                                self.get_logger().warn(f"[CRITICAL] JAX Execution took {jax_dt*1000:.2f} ms at t={t:.2f}s!")
                                
                            phi_val = np.array(object=phi_out, dtype=np.float64)
                        
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
        
                norm_e: float = float(np.linalg.norm(e))
                norm_u: float = float(np.linalg.norm(u))
                
                self.time_history.append(t)
                self.error_norm_history.append(norm_e)
                self.control_output_norm_history.append(norm_u)
                self.control_output_history.append(u.tolist())
                self.q_history.append(q.tolist())
                self.qd_history.append(qd.tolist())

                if self.controller_type in ["resnet", "integrated_resnet"]:
                    self.weight_history.append(np.array(object=self.theta_hat).flatten().tolist())

                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ:
                    current_error_sq: float = float(norm_e ** 2)
                    current_u_sq: float = float(norm_u ** 2)

                    if not self.cost_started or dt <= 0:
                        # No previous sample to difference against yet (first tick of the
                        # trajectory) - contribute zero jerk rather than a spurious spike.
                        current_u_dot_sq: float = 0.0
                    else:
                        u_dot: np.ndarray = (u - self.last_u) / dt
                        current_u_dot_sq = float(np.dot(u_dot, u_dot))

                    current_cost_integrand: float = (
                        (self.q_e * current_error_sq) + (self.r_u * current_u_sq) + (self.r_udot * current_u_dot_sq)
                    )

                    if not self.cost_started:
                        # Seed the history at exact start to prevent trapezoidal integration jump
                        self.last_error_sq = current_error_sq
                        self.last_u_sq = current_u_sq
                        self.last_cost_integrand = current_cost_integrand
                        self.cost_started = True

                    self.error_sq_integral += (dt / 2.0) * (current_error_sq + self.last_error_sq)
                    self.last_error_sq = current_error_sq

                    self.u_sq_integral += (dt / 2.0) * (current_u_sq + self.last_u_sq)
                    self.last_u_sq = current_u_sq

                    self.cost_J += (dt / 2.0) * (current_cost_integrand + self.last_cost_integrand)
                    self.last_cost_integrand = current_cost_integrand

                    self.last_u = u.copy()
                    self.last_t = t
                
                self.is_saturated = False
                self.freeze_int_xy = False
                self.freeze_int_z = False
                
                u_xy: np.ndarray = u[0:2]
                norm_uxy: float = float(np.linalg.norm(u_xy))
                if norm_uxy > self.acc_hor_max_mps2:
                    u[0:2] = u_xy * (self.acc_hor_max_mps2 / norm_uxy)
                    self.is_saturated = True
                    if np.dot(a=e[0:2], b=u[0:2]) > 0.0:
                        self.freeze_int_xy = True
                    self.get_logger().debug(f"[DEBUG] XY SATURATION at t={t:.2f}s!")
                    
                if abs(u[2]) > self.acc_vert_max_mps2:
                    u[2] = self.acc_vert_max_mps2 * np.sign(u[2])
                    self.is_saturated = True
                    if np.sign(e[2]) == np.sign(u[2]):
                        self.freeze_int_z = True
                    self.get_logger().debug(f"[DEBUG] Z SATURATION at t={t:.2f}s!")

                self.publish_trajectory_setpoint_acceleration(ax=u[0], ay=u[1], az=u[2])
                
                if self.experiment_state == ExperimentState.STATE_FOLLOW_TRAJ and t >= self.run_length_s:
                    rms_error: float = math.sqrt(self.error_sq_integral / self.run_length_s) if self.run_length_s > 0 else 0.0
                    rms_u: float = math.sqrt(self.u_sq_integral / self.run_length_s) if self.run_length_s > 0 else 0.0
                    self.get_logger().info(f"[RESULT] Final Cost = {self.cost_J:.2f}")
                    self.get_logger().info(f"[RESULT] RMS Error = {rms_error:.4f}")
                    self.get_logger().info(f"[RESULT] RMS Control Effort = {rms_u:.3f}")
                    raise SystemExit("Trajectory completed successfully.")

def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: AviaryRiseNode = AviaryRiseNode()
    try:
        rclpy.spin(node=node)
    except SystemExit as e:
        node.get_logger().info(f"Experiment terminated: {e}")
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received.")
    except ValueError as e:
        node.get_logger().fatal(f"Value error: {e}")
    except CriticalHardwareError as e:
        node.get_logger().fatal(f"Hardware error: {e}")
    except OdomTimeoutError as e:
        node.get_logger().fatal(f"Odometry timeout: {e}")
    except FailsafeTriggeredError as e:
        node.get_logger().fatal(f"Failsafe triggered: {e}")
    except BoundaryBreachError as e:
        node.get_logger().fatal(f"Boundary breach: {e}")
    finally:
        if node.save_data:
            node.get_logger().info("Saving telemetry data to CSV...")
            node.log_csv()

        if rclpy.ok(): 
            node.get_logger().info("Commanding vehicle to land.")
            node.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_NAV_LAND, param1=0.0, param2=0.0)
            
        node.destroy_node()
        if rclpy.ok(): 
            rclpy.shutdown()
            print("[INFO] Node cleanly destroyed.")
        else:
            print("[FATAL] Node not cleanly destroyed.")

if __name__ == '__main__':
    main()
