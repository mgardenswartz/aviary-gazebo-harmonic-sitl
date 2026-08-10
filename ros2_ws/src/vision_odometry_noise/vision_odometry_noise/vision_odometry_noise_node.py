import os
import csv
import time
from typing import Optional, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry

# Static rotation from FLU (ROS body) to FRD (PX4 body), and the symmetric ENU<->NED
# rotation, exactly matching GZBridge::rotateQuaternion in
# src/modules/simulation/gz_bridge/GZBridge.cpp -- this node replaces that native path
# (see the disabled _visual_odometry_pub.publish() call there) and must apply the
# identical frame conversion or attitude will be silently wrong.
_Q_FLU_TO_FRD_INV = np.array([0.0, -1.0, 0.0, 0.0])  # conjugate of (0,1,0,0), a unit quaternion
_Q_ENU_TO_NED = np.array([0.0, 0.70711, 0.70711, 0.0])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Hamilton product, both operands and result as [w, x, y, z].
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def flu_enu_quat_to_frd_ned(q_flu_to_enu: np.ndarray) -> np.ndarray:
    # q_FRD_to_NED = q_ENU_to_NED * q_FLU_to_ENU * q_FLU_to_FRD.Inverse()
    return quat_mul(quat_mul(_Q_ENU_TO_NED, q_flu_to_enu), _Q_FLU_TO_FRD_INV)


class VisionOdometryNoiseNode(Node):
    def __init__(self) -> None:
        super().__init__(
            node_name='vision_odometry_noise_node',
            automatically_declare_parameters_from_overrides=True
        )

        self.vehicle_name: str = self.get_parameter(name='vehicle_name').value

        # Ablation switch. True (default): realistic sensor emulation -- clipped Gaussian
        # noise on position/velocity/orientation, plus irregular bootstrap-sampled publish
        # timing (~107Hz mean). False: pass Gazebo ground truth straight through with zero
        # added noise, on a fixed-rate timer (clean_publish_rate_hz below) instead of
        # synchronously on every ground-truth sample -- see the note below for why.
        self.noise_and_delays_enabled: bool = self.get_parameter(name='noise_and_delays_enabled').value

        ground_truth_topic: str = self.get_parameter(name='ground_truth_topic').value
        if not ground_truth_topic:
            # Ground truth from the sentinel_vision model's gz-sim-odometry-publisher-system
            # plugin, bridged to ROS 2 by gz_odom.yaml.
            ground_truth_topic = f'/model/{self.vehicle_name}/odometry'

        self.position_noise_std_m: float = self.get_parameter(name='position_noise_std_m').value
        self.velocity_noise_std_mps: float = self.get_parameter(name='velocity_noise_std_mps').value
        self.orientation_noise_std_rad: float = self.get_parameter(name='orientation_noise_std_rad').value

        self.position_noise_clip_sigma: float = self.get_parameter(name='position_noise_clip_sigma').value
        self.velocity_noise_clip_sigma: float = self.get_parameter(name='velocity_noise_clip_sigma').value

        rng_seed: int = self.get_parameter(name='bootstrap_pool_rng_seed').value
        self.rng: np.random.Generator = np.random.default_rng(seed=None if rng_seed < 0 else rng_seed)

        # Fixed publish rate used ONLY when noise_and_delays_enabled=false. Previously this
        # mode published synchronously on every ground-truth tick (~250Hz, matching Gazebo's
        # physics step) -- confounding the ablation, since noise_and_delays_enabled=true's
        # bootstrap-jittered timer runs at a completely different rate (~107Hz mean) on top of
        # adding noise. Publishing clean data at a comparable, deliberately-chosen rate instead
        # isolates noise as the only variable between the two arms.
        self.clean_publish_rate_hz: float = self.get_parameter(name='clean_publish_rate_hz').value

        self.bootstrap_pool_s: Optional[np.ndarray] = None
        if self.noise_and_delays_enabled:
            bootstrap_pool_csv: str = self.get_parameter(name='bootstrap_pool_csv').value
            if not bootstrap_pool_csv:
                bootstrap_pool_csv = os.path.join(
                    get_package_share_directory('vision_odometry_noise'), 'data', 'dt_bootstrap_pool_us.csv'
                )
            self.bootstrap_pool_s = self._load_bootstrap_pool(csv_path=bootstrap_pool_csv)

        self.latest_ground_truth: Optional[Odometry] = None
        self.latest_capture_time_us: int = 0

        fast_qos_profile: QoSProfile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
            history=HistoryPolicy.KEEP_LAST
        )

        self.vehicle_visual_odometry_pub = self.create_publisher(
            msg_type=VehicleOdometry,
            topic=f'/{self.vehicle_name}/fmu/in/vehicle_visual_odometry',
            qos_profile=fast_qos_profile
        )

        self.ground_truth_sub = self.create_subscription(
            msg_type=Odometry,
            topic=ground_truth_topic,
            callback=self.ground_truth_callback,
            qos_profile=10
        )

        # Publish timing is always timer-driven, never directly from ground_truth_callback --
        # that's what keeps the two ablation arms comparable on rate. noise_and_delays_enabled
        # =true: each inter-publish interval is drawn with replacement from the bootstrap pool
        # (mean ~9.3ms / ~107Hz), so the node reschedules a fresh one-shot timer after every
        # publish. =false: a single fixed-period timer at clean_publish_rate_hz.
        self._publish_timer = None
        if self.noise_and_delays_enabled:
            self._schedule_next_publish()
            self.get_logger().info(
                f"Node Initialized Successfully. Ground truth from '{ground_truth_topic}', "
                f"publishing noisy vehicle_visual_odometry with {len(self.bootstrap_pool_s)}-sample bootstrap timing."
            )
        else:
            self._publish_timer = self.create_timer(
                timer_period_sec=1.0 / self.clean_publish_rate_hz, callback=self._publish_odometry
            )
            self.get_logger().info(
                f"Node Initialized Successfully. Ground truth from '{ground_truth_topic}', "
                f"publishing CLEAN (noise_and_delays_enabled=false) vehicle_visual_odometry "
                f"at a fixed {self.clean_publish_rate_hz:.1f}Hz."
            )

        # --- Pipeline latency diagnostics ---
        # This node's use_sim_time is a built-in rclpy Node parameter (always present, no
        # need to declare it) -- log it explicitly since it changes how the latency numbers
        # below must be read. If false (the repo-wide default -- grepped, nothing sets it),
        # self.get_clock().now() returns real wall-clock time while msg.header.stamp on the
        # bridged ground truth is Gazebo SIM time, so "receive wall time minus sim capture
        # time" mixes two clock domains. That's still informative -- a STABLE (non-growing)
        # offset over the run is a genuine, usable latency number (real time and sim time
        # advance at the same rate whenever sim runs at real-time-factor 1.0); a GROWING
        # offset would instead mean RTF != 1 and these numbers need a different treatment.
        # The interval measurements (arrival jitter) are wall-clock-only and unambiguous
        # either way.
        self.get_logger().warn(
            f"[DIAG] use_sim_time={self.get_parameter('use_sim_time').value} -- see latency "
            f"diagnostic comment in source for how to interpret the numbers below."
        )
        self._diag_gt_arrival_last_wall: Optional[float] = None
        self._diag_gt_arrival_intervals: List[float] = []
        # Apparent latency at ground-truth RECEIPT: how stale the sample already was the
        # instant this node got it (bridge + gz-transport + DDS latency only).
        self._diag_recv_latency_samples: List[float] = []
        # Apparent latency at PUBLISH to PX4: how stale the sample is by the time PX4 sees it
        # (receipt latency above + however long it sat cached before this node's own publish
        # schedule got to it). This second number is the one that should inform EKF2_EV_DELAY
        # -- it's what PX4 actually needs to compensate for.
        self._diag_pub_latency_samples: List[float] = []
        self._diag_report_timer = self.create_timer(timer_period_sec=2.0, callback=self.diag_report_callback)

    def diag_report_callback(self) -> None:
        def summarize(samples: List[float]) -> str:
            if not samples:
                return "n=0"
            s = sorted(samples)
            n = len(s)
            p95 = s[int(0.95 * (n - 1))]
            return f"n={n} min={s[0]*1e3:.1f}ms mean={(sum(s)/n)*1e3:.1f}ms p95={p95*1e3:.1f}ms max={s[-1]*1e3:.1f}ms"

        self.get_logger().warn(
            f"[DIAG] gt_arrival_interval: {summarize(self._diag_gt_arrival_intervals)} | "
            f"recv_latency: {summarize(self._diag_recv_latency_samples)} | "
            f"pub_latency (EV_DELAY-relevant): {summarize(self._diag_pub_latency_samples)}"
        )
        self._diag_gt_arrival_intervals = []
        self._diag_recv_latency_samples = []
        self._diag_pub_latency_samples = []

    def _clipped_normal(self, scale: float, clip_sigma: float, size: int) -> np.ndarray:
        # Truncated Gaussian: draw i.i.d. N(0, scale^2) per axis, then clamp each component to
        # +/- clip_sigma*scale. Keeps the noise Gaussian in the bulk while putting a hard bound
        # on how far a single sample can land. clip_sigma must be kept well below the relevant
        # PX4 EKF2 innovation gate (EKF2_EVV_GATE/EKF2_EVP_GATE), not just "a few sigma" -- see
        # the derivation in params/vision_odometry_noise.yaml. At clip_sigma == gate the clip
        # bound sits exactly on the EKF2 rejection boundary (test_ratio ~= 1.0), which is not
        # actually safe.
        sample = self.rng.normal(loc=0.0, scale=scale, size=size)
        bound = clip_sigma * scale
        return np.clip(sample, -bound, bound)

    def _load_bootstrap_pool(self, csv_path: str) -> np.ndarray:
        with open(file=csv_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # header: dt_us
            dt_us: List[int] = [int(row[0]) for row in reader if row]

        if not dt_us:
            raise ValueError(f"Bootstrap pool CSV '{csv_path}' contained no dt values.")

        return np.array(dt_us, dtype=np.float64) / 1e6  # microseconds -> seconds

    def _schedule_next_publish(self) -> None:
        if self._publish_timer is not None:
            self.destroy_timer(self._publish_timer)

        dt_s: float = float(self.rng.choice(a=self.bootstrap_pool_s))
        self._publish_timer = self.create_timer(timer_period_sec=dt_s, callback=self._publish_timer_callback)

    def ground_truth_callback(self, msg: Odometry) -> None:
        self.latest_ground_truth = msg
        # True capture time of the sample, per its own header stamp -- NOT the time we get
        # around to publishing it. Transport/pipeline latency is modeled separately via the
        # PX4-side EKF2_EV_DELAY parameter, not simulated here.
        self.latest_capture_time_us = int(msg.header.stamp.sec) * 1_000_000 + int(msg.header.stamp.nanosec) // 1000

        # Diagnostics: real (wall-clock, unambiguous) interval between ground-truth arrivals --
        # directly measures the ros_gz_bridge's actual delivery jitter, no inference needed.
        wall_now: float = time.perf_counter()
        if self._diag_gt_arrival_last_wall is not None:
            self._diag_gt_arrival_intervals.append(wall_now - self._diag_gt_arrival_last_wall)
        self._diag_gt_arrival_last_wall = wall_now

        # Diagnostics: apparent latency at receipt (see the clock-domain caveat in __init__).
        ros_now_s: float = self.get_clock().now().nanoseconds / 1e9
        capture_s: float = self.latest_capture_time_us / 1e6
        self._diag_recv_latency_samples.append(ros_now_s - capture_s)
        # Publishing itself is always timer-driven now (see __init__) -- this callback only
        # updates the cached sample and diagnostics.

    def _publish_timer_callback(self) -> None:
        self._schedule_next_publish()
        self._publish_odometry()

    def _publish_odometry(self) -> None:
        if self.latest_ground_truth is None:
            self.get_logger().warning(
                "Ignoring publish tick: no ground truth odometry received yet.", throttle_duration_sec=2.0
            )
            return

        gt: Odometry = self.latest_ground_truth

        # Ground truth is in Gazebo's world ENU / body FLU convention (same OdometryPublisher
        # instance GZBridge previously consumed) and must be converted to NED / body FRD to
        # match what PX4 expects on vehicle_visual_odometry.
        p_enu = gt.pose.pose.position
        position_ned = np.array([p_enu.y, p_enu.x, -p_enu.z])

        q_flu_to_enu = np.array([
            gt.pose.pose.orientation.w,
            gt.pose.pose.orientation.x,
            gt.pose.pose.orientation.y,
            gt.pose.pose.orientation.z,
        ])
        q_frd_to_ned = flu_enu_quat_to_frd_ned(q_flu_to_enu=q_flu_to_enu)

        v_flu = gt.twist.twist.linear
        velocity_frd = np.array([v_flu.x, -v_flu.y, -v_flu.z])

        q_noisy = q_frd_to_ned

        if self.noise_and_delays_enabled:
            # Independent per-axis Gaussian noise, clipped to +/- clip_sigma std devs
            # (position/velocity noise is isotropic, and the NED/FRD conversions above are
            # pure axis permutations/sign flips, so applying it before or after conversion is
            # statistically identical).
            position_ned += self._clipped_normal(
                scale=self.position_noise_std_m, clip_sigma=self.position_noise_clip_sigma, size=3
            )
            velocity_frd += self._clipped_normal(
                scale=self.velocity_noise_std_mps, clip_sigma=self.velocity_noise_clip_sigma, size=3
            )

            # Orientation noise as a small-angle SO(3) perturbation applied in the body frame
            # (right-multiplied), not per-quaternion-component noise (which wouldn't preserve
            # unit norm and isn't a meaningful attitude uncertainty model).
            phi = self.rng.normal(loc=0.0, scale=self.orientation_noise_std_rad, size=3)
            angle = float(np.linalg.norm(phi))
            if angle > 1e-9:
                axis = phi / angle
            else:
                axis = np.zeros(3)
            q_delta = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])
            q_noisy = quat_mul(q_frd_to_ned, q_delta)
            q_noisy /= np.linalg.norm(q_noisy)

        msg = VehicleOdometry()
        msg.timestamp = self.latest_capture_time_us
        msg.timestamp_sample = self.latest_capture_time_us

        msg.pose_frame = VehicleOdometry.POSE_FRAME_NED
        msg.position = position_ned.tolist()
        msg.q = q_noisy.tolist()

        msg.velocity_frame = VehicleOdometry.VELOCITY_FRAME_BODY_FRD
        msg.velocity = velocity_frd.tolist()

        # EKF2_EV_NOISE_MD is set to 1 (use EKF2_EVx_NOISE directly) in this experiment's
        # airframe config, so PX4 ignores these regardless of noise_and_delays_enabled --
        # populated anyway so the message is self-consistent and correct if that mode is ever
        # flipped back to 0.
        if self.noise_and_delays_enabled:
            msg.position_variance = [self.position_noise_std_m ** 2] * 3
            msg.orientation_variance = [self.orientation_noise_std_rad ** 2] * 3
            msg.velocity_variance = [self.velocity_noise_std_mps ** 2] * 3
        else:
            msg.position_variance = [0.0] * 3
            msg.orientation_variance = [0.0] * 3
            msg.velocity_variance = [0.0] * 3

        # No angular velocity field is populated -- that channel isn't fused (EKF2_EV_CTRL
        # bit 2 controls velocity, not angular rate; angular velocity comes from the IMU).

        # Diagnostics: apparent latency at the moment PX4 actually receives this sample --
        # receipt latency (measured in ground_truth_callback) plus however long the sample
        # sat cached before this publish tick got to it. This is the number that should
        # inform EKF2_EV_DELAY, not the receipt-latency one above.
        ros_now_s: float = self.get_clock().now().nanoseconds / 1e9
        capture_s: float = self.latest_capture_time_us / 1e6
        self._diag_pub_latency_samples.append(ros_now_s - capture_s)

        self.vehicle_visual_odometry_pub.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: VisionOdometryNoiseNode = VisionOdometryNoiseNode()
    try:
        rclpy.spin(node=node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
