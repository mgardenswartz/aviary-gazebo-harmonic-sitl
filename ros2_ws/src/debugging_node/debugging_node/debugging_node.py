import os
import math
import csv
import numpy as np
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleOdometry

NAV_STATE_NAMES = {
    VehicleStatus.NAVIGATION_STATE_MANUAL: "MANUAL",
    VehicleStatus.NAVIGATION_STATE_ALTCTL: "ALTCTL",
    VehicleStatus.NAVIGATION_STATE_POSCTL: "POSCTL",
    VehicleStatus.NAVIGATION_STATE_AUTO_MISSION: "AUTO_MISSION",
    VehicleStatus.NAVIGATION_STATE_AUTO_LOITER: "AUTO_LOITER",
    VehicleStatus.NAVIGATION_STATE_AUTO_RTL: "AUTO_RTL",
    VehicleStatus.NAVIGATION_STATE_ACRO: "ACRO",
    VehicleStatus.NAVIGATION_STATE_DESCEND: "DESCEND",
    VehicleStatus.NAVIGATION_STATE_TERMINATION: "TERMINATION",
    VehicleStatus.NAVIGATION_STATE_OFFBOARD: "OFFBOARD",
    VehicleStatus.NAVIGATION_STATE_STAB: "STAB",
    VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF: "AUTO_TAKEOFF",
    VehicleStatus.NAVIGATION_STATE_AUTO_LAND: "AUTO_LAND",
    VehicleStatus.NAVIGATION_STATE_AUTO_FOLLOW_TARGET: "AUTO_FOLLOW_TARGET",
    VehicleStatus.NAVIGATION_STATE_AUTO_PRECLAND: "AUTO_PRECLAND",
    VehicleStatus.NAVIGATION_STATE_ORBIT: "ORBIT",
    VehicleStatus.NAVIGATION_STATE_AUTO_VTOL_TAKEOFF: "AUTO_VTOL_TAKEOFF",
}

class ExperimentState:
    STATE_INIT: int = 0
    STATE_TAKEOFF: int = 1
    STATE_STEP_INPUT: int = 2
    STATE_FINISH_UP: int = 3
    STATE_DONE: int = 4

class OdomTimeoutError(Exception):
    pass

class FailsafeTriggeredError(Exception):
    pass

class BoundaryBreachError(Exception):
    pass

class ExperimentFinished(Exception):
    pass

class DebuggingNode(Node):
    def __init__(self) -> None:
        super().__init__(
            node_name='debugging_node',
            automatically_declare_parameters_from_overrides=True
        )
        # Basic Simulation Parameters
        self.vehicle_name: str = self.get_parameter(name='vehicle_name').value
        control_freq_hz: float = self.get_parameter(name='control_freq_hz').value
        self.control_period_s: float = 1.0 / control_freq_hz
        self.save_data: bool = self.get_parameter(name='save_data').value
        self.run_length_s: float = self.get_parameter(name='run_length_s').value
        self.init_tol_m: float = self.get_parameter(name='init_tol_m').value
        self.init_z_m_ned: float = self.get_parameter(name='init_z_m_ned').value
        self.n_axes: int = 3

        # Step input (the single acceleration command sent once takeoff has settled)
        self.step_input_delay_s: float = self.get_parameter(name='step_input_delay_s').value
        step_input_accel_x_mps2: float = self.get_parameter(name='step_input_accel_x_mps2').value
        step_input_accel_y_mps2: float = self.get_parameter(name='step_input_accel_y_mps2').value
        step_input_accel_z_mps2: float = self.get_parameter(name='step_input_accel_z_mps2').value
        self.step_input_accel_mps2: np.ndarray = np.array(
            object=[step_input_accel_x_mps2, step_input_accel_y_mps2, step_input_accel_z_mps2], dtype=np.float64
        )
        self.heartbeat_cutoff_delay_s: float = self.get_parameter(name='heartbeat_cutoff_delay_s').value

        # Safety
        self.acc_hor_max_mps2: float = self.get_parameter(name='mpc_acc_hor_max_mps2').value
        self.acc_vert_max_mps2: float = self.get_parameter(name='mpc_acc_vert_max_mps2').value
        self.safe_x_min_m_ned: float = self.get_parameter(name='safe_x_min_m_ned').value
        self.safe_x_max_m_ned: float = self.get_parameter(name='safe_x_max_m_ned').value
        self.safe_y_min_m_ned: float = self.get_parameter(name='safe_y_min_m_ned').value
        self.safe_y_max_m_ned: float = self.get_parameter(name='safe_y_max_m_ned').value
        self.safe_z_min_m_ned: float = self.get_parameter(name='safe_z_min_m_ned').value
        self.safe_z_max_m_ned: float = self.get_parameter(name='safe_z_max_m_ned').value
        self.offboard_mode_heartbeat_freq_hz: float = self.get_parameter(name='offboard_mode_heartbeat_freq_hz').value
        offboard_mode_heartbeat_period_s = 1.0 / self.offboard_mode_heartbeat_freq_hz
        self.odom_timeout_s: float = self.get_parameter(name='odom_timeout_s').value
        odom_watchdog_freq_hz: float = self.get_parameter(name='odom_watchdog_freq_hz').value
        odom_watchdog_period_s = 1.0 / odom_watchdog_freq_hz
        mode_publisher_freq_hz: float = self.get_parameter(name='mode_publisher_freq_hz').value
        self.mode_publisher_period_s = 1.0 / mode_publisher_freq_hz

        # Control
        self.k_p: float = self.get_parameter(name='k_p').value
        self.k_i: float = self.get_parameter(name='k_i').value
        self.k_d: float = self.get_parameter(name='k_d').value

        # For VehicleStatus callback
        self.nav_state: int = 0
        self.vehicle_system_id: int = 1
        self.vehicle_component_id: int = 1

        # Internal Flags
        self.is_armed: bool = False
        self.in_offboard_mode: bool = False
        self.landing_command_sent: bool = False
        self.freeze_int_xy: bool = False
        self.freeze_int_z: bool = False
        self.initial_position_locked: bool = False
        self.publish_offboard_heartbeat: bool = False
        self.position_mode_requested: bool = False
        self.step_command_sent: bool = False
        self.latest_odom: Optional[VehicleOdometry] = None

        self.last_odom_ros_time_s: float = 0.0
        self.init_x_m_ned: float = 0.0
        self.init_y_m_ned: float = 0.0
        self.experiment_state: int = ExperimentState.STATE_INIT

        # Per-state timestamps, set as the state machine transitions
        # (last_mode_cmd_time_s is initialized below, once heartbeat_raised_time_s is known)
        self.takeoff_entry_time_s: float = 0.0
        self.step_input_entry_time_s: float = 0.0
        self.finish_up_entry_time_s: float = 0.0
        self.heartbeat_stopped_time_s: Optional[float] = None

        self.reset_integral()

        self.time_history: List[float] = []
        self.control_output_norm_history: List[float] = []
        self.control_output_history: List[List[float]] = []
        self.error_norm_history: List[float] = []
        self.position_history: List[List[float]] = []
        self.qd_history: List[List[float]] = []

        qos_profile: QoSProfile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
            history=HistoryPolicy.KEEP_LAST
        )

        self.offboard_control_mode_pub = self.create_publisher(
            msg_type=OffboardControlMode,
            topic=f'/{self.vehicle_name}/fmu/in/offboard_control_mode',
            qos_profile=qos_profile
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            msg_type=TrajectorySetpoint,
            topic=f'/{self.vehicle_name}/fmu/in/trajectory_setpoint',
            qos_profile=qos_profile
        )
        self.vehicle_command_pub = self.create_publisher(
            msg_type=VehicleCommand,
            topic=f'/{self.vehicle_name}/fmu/in/vehicle_command',
            qos_profile=qos_profile
        )

        self.status_sub = self.create_subscription(
            msg_type=VehicleStatus,
            topic=f'/{self.vehicle_name}/fmu/out/vehicle_status',
            callback=self.vehicle_status_callback,
            qos_profile=qos_profile
        )
        self.odom_sub = self.create_subscription(
            msg_type=VehicleOdometry,
            topic=f'/{self.vehicle_name}/fmu/out/vehicle_odometry',
            callback=self.odom_callback,
            qos_profile=qos_profile
        )

        self.control_timer = self.create_timer(
            timer_period_sec=self.control_period_s,
            callback=self.control_timer_callback
        )
        self.odom_watchdog_timer = self.create_timer(
            timer_period_sec=odom_watchdog_period_s,
            callback=self.odom_watchdog_callback
        )
        self.offboard_heartbeat_timer = self.create_timer(
            timer_period_sec=offboard_mode_heartbeat_period_s,
            callback=self.offboard_heartbeat_callback
        )

        self.publish_offboard_heartbeat = False
        self.heartbeat_raised_time_s = self.get_clock().now().nanoseconds / 1e9
        # Defer the first mode-switch/arm attempt by one mode_publisher_period_s so
        # PX4 has already seen a handful of streamed setpoints -- an immediate
        # attempt (last_mode_cmd_time_s starting at 0.0) races the very first
        # setpoint and PX4 will reject the switch (see the "send a few setpoints
        # before starting" note in the MAVROS offboard tutorial).
        self.last_mode_cmd_time_s = self.heartbeat_raised_time_s

        self.get_logger().info(f"Node Initialized Successfully. Offboard heartbeat raised; waiting for OFFBOARD mode confirmation.")

    def land_vehicle(self) -> None:
        if self.landing_command_sent:
            return
        self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_NAV_LAND, param1=0.0, param2=0.0)
        self.landing_command_sent = True

    def vehicle_status_callback(self, msg: VehicleStatus) -> None:
        was_in_offboard_mode: bool = self.in_offboard_mode
        was_armed: bool = self.is_armed
        prev_nav_state: int = self.nav_state

        self.nav_state = msg.nav_state
        if self.nav_state != prev_nav_state:
            # Which failsafe mode PX4 actually falls back into on OFFBOARD loss is
            # governed by PX4 params (COM_OBL_ACT / COM_OBL_RC_ACT), not by this
            # node -- log it plainly so a run tells us which mode was picked, and
            # keep watching (don't exit here) in case it cascades to another mode.
            self.get_logger().info(
                f"nav_state: {NAV_STATE_NAMES.get(prev_nav_state, prev_nav_state)} -> "
                f"{NAV_STATE_NAMES.get(self.nav_state, self.nav_state)}"
            )
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.in_offboard_mode = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.vehicle_system_id = msg.system_id
        self.vehicle_component_id = msg.component_id

        if was_in_offboard_mode and not self.in_offboard_mode:
            now_s: float = self.get_clock().now().nanoseconds / 1e9
            if self.heartbeat_stopped_time_s is not None:
                self.get_logger().info(
                    f"PX4 exited OFFBOARD mode {now_s - self.heartbeat_stopped_time_s:.3f}s after the heartbeat was stopped."
                )
            else:
                self.get_logger().info(f"PX4 exited OFFBOARD mode at t={now_s:.3f}s (heartbeat was still active).")

        # The flight isn't actually over just because OFFBOARD was left -- PX4's
        # failsafe fallback (Hold/Land/RTL/...) still has to run its course. Wait
        # for the real end-of-flight signal (auto-disarm after landing) so the log
        # captures whatever that fallback actually does, instead of going dark
        # right as it starts.
        if was_armed and not self.is_armed and self.experiment_state == ExperimentState.STATE_DONE:
            raise ExperimentFinished("PX4 disarmed after the deliberate heartbeat cutoff.")

    def odom_callback(self, msg: VehicleOdometry) -> None:
        self.latest_odom = msg
        self.last_odom_ros_time_s = self.get_clock().now().nanoseconds / 1e9

        if not self.initial_position_locked:
            self.init_x_m_ned = float(msg.position[0])
            self.init_y_m_ned = float(msg.position[1])
            self.initial_position_locked = True

    def odom_watchdog_callback(self) -> None:
        if self.latest_odom is None: return

        elapsed_s = self.get_clock().now().nanoseconds / 1e9 - self.last_odom_ros_time_s

        if elapsed_s >= self.odom_timeout_s:
            self.publish_offboard_heartbeat = False
            raise OdomTimeoutError(f"No odometry received for {elapsed_s:.1f}s.")


    def reset_integral(self) -> None:
        self.current_control_integrand = np.zeros(shape=self.n_axes, dtype=np.float64)
        self.last_control_integrand = np.zeros(shape=self.n_axes, dtype=np.float64)

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
        self.vehicle_command_pub.publish(msg)

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
        self.offboard_control_mode_pub.publish(msg)

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

        self.trajectory_setpoint_pub.publish(msg)

    def write_csv(self) -> None:
        # You can either dump the CSV all at once at the end or periodically (at a specified rate to not slow down the control). TODO: Update to periodically.

        base_dir: str = f"/home/root/gazebo_debugging_data"
        os.makedirs(name=base_dir, exist_ok=True)
        csv_filename: str = os.path.join(base_dir, f"data.csv")

        try:
            with open(file=csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                headers: List[str] = [
                    "time_s", "x_m_ned", "y_m_ned", "z_m_ned",
                    "xd_m_ned", "yd_m_ned", "zd_m_ned",
                    "error_norm_m", "ax_mps2", "ay_mps2", "az_mps2", "control_output_norm_mps2",
                ]
                writer.writerow(headers)

                for row_num in range(len(self.time_history)):
                    row_data: List[float] = [
                        self.time_history[row_num],
                        *self.position_history[row_num],
                        *self.qd_history[row_num],
                        self.error_norm_history[row_num],
                        *self.control_output_history[row_num],
                        self.control_output_norm_history[row_num],
                    ]
                    writer.writerow(row_data)

            self.get_logger().info(f"Telemetry saved to {csv_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to write CSV: {e}")

    def check_safety_boundary(self, position: np.ndarray) -> Optional[str]:
        if not (self.safe_x_min_m_ned <= position[0] <= self.safe_x_max_m_ned):
            return f"X position {position[0]:.2f} breached bounds [{self.safe_x_min_m_ned}, {self.safe_x_max_m_ned}]."
        if not (self.safe_y_min_m_ned <= position[1] <= self.safe_y_max_m_ned):
            return f"Y position {position[1]:.2f} breached bounds [{self.safe_y_min_m_ned}, {self.safe_y_max_m_ned}]."
        if not (self.safe_z_min_m_ned <= position[2] <= self.safe_z_max_m_ned):
            return f"Z position {position[2]:.2f} breached bounds [{self.safe_z_min_m_ned}, {self.safe_z_max_m_ned}]."
        return None

    def get_desired_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.experiment_state != ExperimentState.STATE_TAKEOFF:
            raise ValueError(f"get_desired_state() is only defined during STATE_TAKEOFF (state={self.experiment_state}).")

        # During takeoff, hold exactly above where it initialized
        return (np.array(object=[self.init_x_m_ned, self.init_y_m_ned, self.init_z_m_ned], dtype=np.float64),
                np.zeros(shape=3, dtype=np.float64), np.zeros(shape=3, dtype=np.float64))

    def run_pid(self, qd: np.ndarray, qd_dot: np.ndarray, dt: float) -> np.ndarray:
        q: np.ndarray = np.array(object=self.latest_odom.position, dtype=np.float64)
        q_dot: np.ndarray = np.array(object=self.latest_odom.velocity, dtype=np.float64)

        e: np.ndarray = qd - q
        e_dot: np.ndarray = qd_dot - q_dot

        # PID Controller (trapezoidal integration)
        current_integrand: np.ndarray = (self.k_i * e)
        delta_int: np.ndarray = (dt / 2.0) * (current_integrand + self.last_control_integrand)
        if not self.freeze_int_xy:
            self.current_control_integrand[0:2] += delta_int[0:2]
        if not self.freeze_int_z:
            self.current_control_integrand[2] += delta_int[2]
        self.last_control_integrand = current_integrand

        u: np.ndarray = (self.k_p * e) + (self.k_d * e_dot) + self.current_control_integrand

        # Check saturation
        self.freeze_int_xy = False
        self.freeze_int_z = False
        u_xy: np.ndarray = u[0:2]
        norm_uxy: float = float(np.linalg.norm(u_xy))
        if norm_uxy > self.acc_hor_max_mps2:
            u[0:2] = u_xy * (self.acc_hor_max_mps2 / norm_uxy)
            if np.dot(a=e[0:2], b=u[0:2]) > 0.0:
                self.freeze_int_xy = True
            self.get_logger().debug(f"XY SATURATION!")

        if abs(u[2]) > self.acc_vert_max_mps2:
            u[2] = self.acc_vert_max_mps2 * np.sign(u[2])
            if np.sign(e[2]) == np.sign(u[2]):
                self.freeze_int_z = True
            self.get_logger().debug(f"Z SATURATION!")

        if self.save_data:
            now_s: float = self.get_clock().now().nanoseconds / 1e9
            self.time_history.append(now_s)
            self.position_history.append(q.tolist())
            self.qd_history.append(qd.tolist())
            self.error_norm_history.append(float(np.linalg.norm(e)))
            self.control_output_history.append(u.tolist())
            self.control_output_norm_history.append(float(np.linalg.norm(u)))

        return u

    def control_timer_callback(self) -> None:
        if self.latest_odom is None: return

        now_s: float = self.get_clock().now().nanoseconds / 1e9

        # Runs every tick regardless of state, including STATE_DONE, so a boundary
        # breach still cuts the experiment short even during the deliberate
        # hands-off post-heartbeat-cutoff phase.
        q: np.ndarray = np.array(object=self.latest_odom.position, dtype=np.float64)
        boundary_err: Optional[str] = self.check_safety_boundary(position=q)
        if boundary_err is not None:
            self.publish_offboard_heartbeat = False
            raise BoundaryBreachError(boundary_err)

        match self.experiment_state:
            case ExperimentState.STATE_INIT:
                # Must publish setpoints with a TrajectorySetpoint otherwise transition to Offboard will be declined
                self.publish_offboard_heartbeat = True
                self.publish_trajectory_setpoint_acceleration(ax=0.0, ay=0.0, az=0.0)

                if not self.position_mode_requested:
                    # Recommended PX4 practice: enter OFFBOARD from Position mode, so that if
                    # the vehicle ever drops out of OFFBOARD it falls back to a stable hover
                    # instead of whatever mode it happened to boot into.
                    # param1=1.0 -> MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, param2=3.0 -> PX4 custom main mode POSCTL
                    self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=3.0)
                    self.position_mode_requested = True

                if now_s - self.heartbeat_raised_time_s > self.run_length_s:
                    raise FailsafeTriggeredError(
                        f"Vehicle did not reach ARMED + OFFBOARD within run_length_s={self.run_length_s:.1f}s of the heartbeat being raised."
                    )

                # SITL-only: there's no RC pilot / QGC operator to arm and flip the mode
                # switch, so this node has to do both itself. On real hardware this whole
                # block is unnecessary -- the node would just wait for offboard.
                #
                # NOTE: PX4 will not accept an arm command until it is already switching
                # into OFFBOARD (it rejects COMPONENT_ARM_DISARM while sitting in
                # the default AUTO mode). So both commands must be retried together, not
                # arm-then-switch -- gating the mode-switch behind is_armed deadlocks, since
                # arming itself depends on the switch being in flight.
                if not (self.is_armed and self.in_offboard_mode):
                    self.get_logger().info("Waiting for ARM + OFFBOARD mode switch...", throttle_duration_sec=2.0)
                    if now_s - self.last_mode_cmd_time_s > self.mode_publisher_period_s:
                        # param1=1.0 -> MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, param2=6.0 -> PX4 custom main mode OFFBOARD
                        self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                        if not self.is_armed:
                            self.publish_vehicle_command(command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0, param2=0.0)
                        self.last_mode_cmd_time_s = now_s
                    return

                # Step 2: PX4 has confirmed OFFBOARD. Time how long that took and log it.
                elapsed_s: float = now_s - self.heartbeat_raised_time_s
                self.get_logger().info(f"PX4 confirmed OFFBOARD mode after {elapsed_s:.3f}s.")

                self.get_logger().info(f"ARMED & OFFBOARD validated. Initializing takeoff to z={self.init_z_m_ned:.2f}m (NED).")
                self.reset_integral()
                self.experiment_state = ExperimentState.STATE_TAKEOFF
                self.takeoff_entry_time_s = now_s

            case ExperimentState.STATE_TAKEOFF:
                if not self.in_offboard_mode:
                    self.publish_offboard_heartbeat = False
                    raise FailsafeTriggeredError("PX4 left OFFBOARD mode during takeoff.")

                if now_s - self.heartbeat_raised_time_s > self.run_length_s:
                    self.publish_offboard_heartbeat = False
                    raise FailsafeTriggeredError(
                        f"Takeoff did not settle within run_length_s={self.run_length_s:.1f}s of the heartbeat being raised."
                    )

                qd, qd_dot, qd_ddot = self.get_desired_state()
                u: np.ndarray = self.run_pid(qd=qd, qd_dot=qd_dot, dt=self.control_period_s) + qd_ddot
                self.publish_trajectory_setpoint_acceleration(ax=u[0], ay=u[1], az=u[2])

                # Transition once within init_tol_m of the full desired state
                # (position vector norm, not just the z-component)
                error_norm: float = float(np.linalg.norm(qd - q))
                if error_norm <= self.init_tol_m:
                    self.get_logger().info(
                        f"TAKEOFF SETTLED after {now_s - self.takeoff_entry_time_s:.3f}s "
                        f"(error={error_norm:.2f}m <= tol={self.init_tol_m:.2f}m)."
                    )
                    self.experiment_state = ExperimentState.STATE_STEP_INPUT
                    self.step_input_entry_time_s = now_s

            case ExperimentState.STATE_STEP_INPUT:
                if not self.in_offboard_mode:
                    self.publish_offboard_heartbeat = False
                    raise FailsafeTriggeredError("PX4 left OFFBOARD mode before the step input was sent.")

                # Step 5: wait step_input_delay_s, then send exactly one acceleration
                # command and stop publishing TrajectorySetpoint entirely.
                if now_s - self.step_input_entry_time_s >= self.step_input_delay_s:
                    ax, ay, az = self.step_input_accel_mps2
                    self.publish_trajectory_setpoint_acceleration(ax=ax, ay=ay, az=az)
                    self.get_logger().info(
                        f"Step input sent {now_s - self.step_input_entry_time_s:.3f}s after settling: "
                        f"accel=[{ax:.2f}, {ay:.2f}, {az:.2f}] m/s^2. No further setpoints will be sent."
                    )
                    self.experiment_state = ExperimentState.STATE_FINISH_UP
                    self.finish_up_entry_time_s = now_s

            case ExperimentState.STATE_FINISH_UP:
                # No TrajectorySetpoint messages are sent here; only the heartbeat continues.
                if now_s - self.finish_up_entry_time_s >= self.heartbeat_cutoff_delay_s:
                    self.publish_offboard_heartbeat = False
                    self.heartbeat_stopped_time_s = now_s
                    self.get_logger().info(
                        f"Heartbeat stopped {now_s - self.finish_up_entry_time_s:.3f}s after the step input. "
                        f"Waiting for PX4 to exit OFFBOARD."
                    )
                    self.experiment_state = ExperimentState.STATE_DONE

            case ExperimentState.STATE_DONE:
                pass

def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: DebuggingNode = DebuggingNode()
    try:
        rclpy.spin(node=node)
    except ExperimentFinished as e:
        node.get_logger().info(f"Experiment terminated: {e}")
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received.")
    except ValueError as e:
        node.get_logger().fatal(f"Value error: {e}")
    except OdomTimeoutError as e:
        node.get_logger().fatal(f"Odometry timeout: {e}")
    except FailsafeTriggeredError as e:
        node.get_logger().fatal(f"Failsafe triggered: {e}")
    except BoundaryBreachError as e:
        node.get_logger().fatal(f"Boundary breach: {e}")
    finally:
        node.get_logger().info("Commanding vehicle to land.")
        node.land_vehicle()
        if rclpy.ok():
            if node.save_data:
                node.get_logger().info("Saving run data to CSV...")
                node.write_csv()

            print("[INFO] Node cleanly destroyed.")
        else:
            print("[FATAL] Node not cleanly destroyed.")

        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
