import os
import csv
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

        bootstrap_pool_csv: str = self.get_parameter(name='bootstrap_pool_csv').value
        if not bootstrap_pool_csv:
            bootstrap_pool_csv = os.path.join(
                get_package_share_directory('vision_odometry_noise'), 'data', 'dt_bootstrap_pool_us.csv'
            )
        self.bootstrap_pool_s: np.ndarray = self._load_bootstrap_pool(csv_path=bootstrap_pool_csv)

        rng_seed: int = self.get_parameter(name='bootstrap_pool_rng_seed').value
        self.rng: np.random.Generator = np.random.default_rng(seed=None if rng_seed < 0 else rng_seed)

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

        # Publish timing is not a fixed rate: each inter-publish interval is drawn with
        # replacement from the bootstrap pool, so the node reschedules a fresh one-shot
        # timer after every publish rather than using a single fixed-period timer.
        self._publish_timer = None
        self._schedule_next_publish()

        self.get_logger().info(
            f"Node Initialized Successfully. Ground truth from '{ground_truth_topic}', "
            f"publishing noisy vehicle_visual_odometry with {len(self.bootstrap_pool_s)}-sample bootstrap timing."
        )

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

    def _publish_timer_callback(self) -> None:
        self._schedule_next_publish()

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

        # Independent per-axis Gaussian noise, clipped to +/- clip_sigma std devs (position/
        # velocity noise is isotropic, and the NED/FRD conversions above are pure axis
        # permutations/sign flips, so applying it before or after conversion is statistically
        # identical).
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
        # airframe config, so PX4 ignores these -- populated anyway so the message is
        # self-consistent and correct if that mode is ever flipped back to 0.
        msg.position_variance = [self.position_noise_std_m ** 2] * 3
        msg.orientation_variance = [self.orientation_noise_std_rad ** 2] * 3
        msg.velocity_variance = [self.velocity_noise_std_mps ** 2] * 3

        # No angular velocity field is populated -- that channel isn't fused (EKF2_EV_CTRL
        # bit 2 controls velocity, not angular rate; angular velocity comes from the IMU).

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
