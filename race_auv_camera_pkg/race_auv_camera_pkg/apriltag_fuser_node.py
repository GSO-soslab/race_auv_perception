"""Multi-camera AprilTag fuser.

Subscribes to one or more ``vision_msgs/Detection3DArray`` topics published
by the per-camera ``apriltag_detector_node`` instances, loads the same URDF
the detectors used (so it knows ``T_base_to_tag`` for every tag id), and on
each tick runs a single joint Umeyama / SVD SE(3) solve over **all** tags
seen by **all** cameras. The result is published as a single TF:

    <reference_frame>  --(T_ref_to_output)-->  <output_frame>

In addition, when ``publish_tag_tfs`` is enabled the fuser broadcasts one
TF per freshly-detected tag:

    <reference_frame>  --(T_ref_to_tag)-->  apriltag_<family>_<id>

e.g. ``base_link -> apriltag_tag36h11_5``. These are raw per-observation
poses (noisier than the fused output) and they do not depend on the
joint solve succeeding. When two cameras see the same tag in one tick,
the estimate from the camera whose observation is closest to the tag
wins. Tag frames disappear from TF as soon as the detection goes stale
(same lifetime semantics as the fused TF).

The frame that ends up in the published TF is the URDF link named
``object.dock_link_name`` (default: the object's base link itself). With
the default config the dock frame is the station's ``base_link``; set
``object.dock_link_name: "dock_point"`` to publish the dock-point frame
instead. The dock frame and ``object_base/pose`` topics are the same
shape and are computed by composing the solved ``T_ref_to_base`` with the
URDF's ``T_base_to_dock`` (a fixed post-multiply).

This replaces the per-camera ``object_base_camN`` frame, which only sees the
subset of tags in that camera's field of view and therefore drifts (or
disappears) when one camera loses its tags.

The fuser works with one input topic just as well as with many: a single
camera becomes the degenerate "one source" case and the math is identical.

Detection freshness is enforced by ``detection_max_age``; a detection older
than the threshold is dropped so the solve cannot be skewed by a stale
detection from a camera that has since moved.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation as R
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from vision_msgs.msg import Detection3DArray

from .apriltag_geom import (
    is_bad_rotation,
    load_yaml_config,
    matrix_to_pose_msg,
    matrix_to_transform_stamped,
    resolve_urdf_path,
    sanitize_rotation,
    solve_cam_to_base,
    solve_cam_to_base_ransac,
)
from .urdf_tag_parser import TagTransform, extract_tag_transforms, link_transform_from_base


def _pose_to_matrix(pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4 homogeneous transform."""
    q = pose.orientation
    p = pose.position
    Rm = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = [p.x, p.y, p.z]
    return T


def _parse_tag_family_id(det_id: str) -> Optional[Tuple[str, int]]:
    """Parse ``Detection3D.id`` into ``(family, tag_id)``.

    ``Detection3D.id`` is ``f"{family}:{tag_id}"`` (e.g. ``"tag36h11:5"``).
    Returns ``None`` for malformed ids. The numeric ``tag_id`` is
    per-family: two tags in different families may share the same id.
    """
    if not det_id:
        return None
    if ":" not in det_id:
        return None
    family_part, _, id_part = det_id.rpartition(":")
    family = family_part.strip()
    if not family:
        return None
    try:
        return family, int(id_part)
    except (TypeError, ValueError):
        return None


class AprilTagFuserNode(Node):
    """Fuse per-camera tag detections into one docked-object TF."""

    def __init__(self) -> None:
        super().__init__("apriltag_fuser")

        # --- Parameters ---
        self.declare_parameter("config_yaml", "")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("urdf_package", "")
        self.declare_parameter("urdf_filename", "")
        self.declare_parameter(
            "reference_frame", "base_link",
            ParameterDescriptor(description="TF frame in which the fused dock pose is published."),
        )
        self.declare_parameter(
            "dock_link_name", "",
            ParameterDescriptor(
                description=(
                    "URDF link whose pose (in the object's base frame) is "
                    "post-multiplied onto the solved T_ref_to_base. Default "
                    "empty -> identity (publishes the base frame itself). "
                    "Configurable via 'object.dock_link_name' in the YAML; "
                    "this ROS param overrides it."
                )
            ),
        )
        self.declare_parameter("output_frame", "object_base")
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter(
            "detections_topics",
            [
                "/cam1/apriltag_detection/detections3d",
                "/cam2/apriltag_detection/detections3d",
            ],
        )
        self.declare_parameter(
            "min_pairs", 3,
            ParameterDescriptor(
                description=(
                    "Minimum (cam, tag) pairs required for the joint Umeyama "
                    "solve. With fewer pairs the fuser falls back to a "
                    "single-tag solution."
                )
            ),
        )
        self.declare_parameter(
            "detection_max_age", 0.5,
            ParameterDescriptor(description="Drop cached detections older than this many seconds."),
        )
        self.declare_parameter(
            "tf_timeout", 0.1,
            ParameterDescriptor(
                description=(
                    "How long to wait (s) for a TF lookup before giving up on "
                    "a detection this tick."
                )
            ),
        )
        self.declare_parameter(
            "output_pose_topic", "object_base/pose",
            ParameterDescriptor(
                description=(
                    "Topic on which to publish a geometry_msgs/PoseStamped "
                    "mirror of the fused dock pose. Empty string disables "
                    "the topic."
                )
            ),
        )
        self.declare_parameter(
            "publish_tag_tfs", True,
            ParameterDescriptor(
                description=(
                    "Broadcast one TF per freshly-detected tag: "
                    "reference_frame -> apriltag_<family>_<id> (e.g. "
                    "base_link -> apriltag_tag36h11_5). Raw per-camera "
                    "observation, noisier than the fused output TF; "
                    "closest camera wins when two cameras see the "
                    "same tag in one tick."
                )
            ),
        )

        # --- Load URDF tag transforms ---
        # Keyed by (family, tag_id) since tag ids are per-family.
        self._tag_transforms: Dict[Tuple[str, int], TagTransform] = {}
        self._urdf_path: str = ""
        self._obj_cfg: dict = {}
        self._T_base_to_dock: Optional[np.ndarray] = None
        self._load_tag_transforms()

        # --- TF ---
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        # --- PoseStamped mirror of the fused object_base pose ---
        pose_topic = str(self.get_parameter("output_pose_topic").value or "")
        self._pose_pub = None
        if pose_topic:
            self._pose_pub = self.create_publisher(
                PoseStamped, pose_topic,
                QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    depth=1,
                ),
            )
            self.get_logger().info(f"Publishing PoseStamped on: {pose_topic}")
        else:
            self.get_logger().info("PoseStamped mirror disabled (output_pose_topic empty).")

        # --- Detection cache: (source_topic, family, tag_id) -> (stamp_ns, frame, T_cam_to_tag) ---
        # tag_id is per-family; the (family, tag_id) tuple uniquely
        # identifies a tag across all configured families and sizes.
        # ``stamp_ns`` is the message header time in nanoseconds (ROS
        # time, not wall-clock) so the freshness check in ``_tick``
        # stays correct when ``use_sim_time`` is on or when wall-clock
        # drifts from ROS time (rosbag replay, paused time).
        self._lock = threading.Lock()
        self._latest: Dict[Tuple[str, str, int], Tuple[int, str, np.ndarray]] = {}

        # --- Subscriptions (one per detection topic) ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=5,
        )
        topics = self._get_param_list("detections_topics")
        if not topics:
            self.get_logger().warn(
                "No detections_topics configured; fuser will never receive data."
            )
        for topic in topics:
            self.create_subscription(
                Detection3DArray, topic,
                self._make_detection_cb(topic), qos,
            )
            self.get_logger().info(f"Subscribed to detections: {topic}")

        # --- Timer ---
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            raise ValueError("publish_rate must be > 0")
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"apriltag_fuser ready: ref={self.get_parameter('reference_frame').value} "
            f"output={self.get_parameter('output_frame').value} rate={rate} Hz "
            f"min_pairs={self.get_parameter('min_pairs').value} "
            f"detection_max_age={self.get_parameter('detection_max_age').value} s "
            f"tag_tfs={self.get_parameter('publish_tag_tfs').value}"
        )

    # ------------------------------------------------------------------ helpers
    def _get_param_list(self, name: str) -> List[str]:
        """Return a parameter as a list of strings, accepting string-or-list."""
        v = self.get_parameter(name).value
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if str(x).strip()]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            # Accept YAML-style list passed as a single string.
            if s.startswith("["):
                import yaml
                parsed = yaml.safe_load(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if str(x).strip()]
            return [s]
        return [str(v)]

    def _load_tag_transforms(self) -> None:
        cfg_path = self.get_parameter("config_yaml").value
        if not cfg_path:
            raise RuntimeError(
                "config_yaml is required by apriltag_fuser (it must point at "
                "the same YAML the detector nodes use, so the fuser knows "
                "T_base_to_tag for every tag id)."
            )
        inner = load_yaml_config(str(cfg_path))
        obj_cfg = inner.get("object", {}) or {}
        urdf_override = self.get_parameter("urdf_path").value
        if urdf_override:
            urdf_path = str(urdf_override)
        else:
            pkg_override = self.get_parameter("urdf_package").value
            file_override = self.get_parameter("urdf_filename").value
            if pkg_override:
                obj_cfg = dict(obj_cfg)
                obj_cfg["urdf_package"] = str(pkg_override)
            if file_override:
                obj_cfg = dict(obj_cfg)
                obj_cfg["urdf_filename"] = str(file_override)
            urdf_path = resolve_urdf_path(obj_cfg)
        if not urdf_path:
            raise RuntimeError(
                "config_yaml needs 'object.urdf_package' + 'object.urdf_filename' "
                "(or override via the 'urdf_path' ROS param)"
            )
        base_link = obj_cfg.get("base_link_name") or None
        prefix = obj_cfg.get("tag_link_prefix", "apriltag")
        tag_transforms = extract_tag_transforms(
            urdf_path, prefix=prefix, base_link_name=base_link,
        )
        if not tag_transforms:
            raise RuntimeError(
                f"No '{prefix}<N>' links found in URDF {urdf_path} under base '{base_link}'"
            )
        self.get_logger().info(
            f"URDF resolved: {urdf_path} ({len(tag_transforms)} tag links)"
        )
        self._tag_transforms = tag_transforms
        self._urdf_path = urdf_path
        self._obj_cfg = obj_cfg

        # Resolve the dock link: ROS param overrides YAML 'object.dock_link_name';
        # empty -> identity (publish the base frame itself).
        dock_override = self.get_parameter("dock_link_name").value
        dock_link = str(dock_override) if dock_override else ""
        if not dock_link:
            dock_link = str(obj_cfg.get("dock_link_name", "") or "")
        dock_link = dock_link.strip()
        if not dock_link:
            self._T_base_to_dock = np.eye(4)
            self.get_logger().info(
                "No dock_link_name set; fuser will publish the base frame "
                "itself (T_base_to_dock = I)."
            )
        else:
            base_for_walk = base_link  # same base as tag extraction
            self._T_base_to_dock = link_transform_from_base(
                urdf_path, dock_link, base_link_name=base_for_walk,
            )
            t = self._T_base_to_dock[:3, 3]
            self.get_logger().info(
                f"Dock link '{dock_link}' resolved; T_base_to_dock translation="
                f"[{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]"
            )

    def _make_detection_cb(self, topic: str):
        def cb(msg: Detection3DArray) -> None:
            stamp = msg.header.stamp
            # Store the message header time as integer nanoseconds so
            # the freshness check in ``_tick`` compares like-with-like
            # (ROS clock vs ROS clock). Previously this was float
            # seconds against ``time.time()`` -- a wall-clock vs ROS
            # bug that silently dropped valid detections when
            # ``use_sim_time`` was on or when wall-clock and ROS clock
            # diverged (rosbag replay, paused time).
            stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            cam_frame = msg.header.frame_id or ""
            for det in msg.detections:
                parsed = _parse_tag_family_id(det.id)
                if parsed is None:
                    continue
                family, tag_id = parsed
                if not det.results or det.results[0].pose is None:
                    continue
                try:
                    T = _pose_to_matrix(det.results[0].pose.pose)
                except Exception as e:
                    self.get_logger().warn(
                        f"Bad pose in {topic} for id={det.id}: {e}"
                    )
                    continue
                # Sanitize immediately; keep the cached T clean.
                if is_bad_rotation(T[:3, :3]):
                    continue
                T[:3, :3] = sanitize_rotation(T[:3, :3])
                with self._lock:
                    self._latest[(topic, family, tag_id)] = (stamp_ns, cam_frame, T)
        return cb

    @staticmethod
    def _detection_weights(
        pairs: List[Tuple[np.ndarray, np.ndarray, Tuple[str, int]]],
    ) -> np.ndarray:
        """Per-pair weight ``w_i ∝ 1 / d_i^2``, normalized to sum to 1.

        Closer tags have a tighter pose solve (range uncertainty
        dominates for a fixed physical size), so this proxy captures
        most of the signal without any protocol change -- the fuser
        only has the pose, not the pixel area, on the wire.

        A small floor (``d_min = 1e-3``) prevents division-by-zero when
        a tag is reported at the camera centre, and the resulting
        weights are normalized so the absolute scale doesn't matter
        downstream.
        """
        d = np.array(
            [max(1e-3, float(np.linalg.norm(p[0][:3, 3]))) for p in pairs],
            dtype=np.float64,
        )
        w = 1.0 / (d * d)
        total = float(w.sum())
        if total <= 0.0 or not np.isfinite(total):
            # Degenerate input: fall back to uniform weights.
            return np.ones(len(pairs), dtype=np.float64) / max(1, len(pairs))
        return w / total

    @staticmethod
    def _fallback_solve(
        pairs: List[Tuple[np.ndarray, np.ndarray, Tuple[str, int]]],
        weights: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Average per-tag T_ref_to_base estimates when joint solve unavailable.

        This keeps the fuser producing a pose even when fewer than three tags
        are visible, by inverting each (T_ref_to_tag, T_base_to_tag) pair and
        averaging the resulting transforms.

        Parameters
        ----------
        pairs
            List of ``(T_ref_to_tag, T_base_to_tag, tag_key)`` tuples.
        weights
            Optional per-pair weight, shape ``(len(pairs),)``. When
            provided, both the translation and the (hemisphere-aligned)
            quaternion averages are weighted. ``None`` means uniform
            weight -- the original behaviour.
        """
        if not pairs:
            return None
        n = len(pairs)
        if weights is None:
            w = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            total = float(w.sum())
            if total <= 0.0:
                w = np.full(n, 1.0 / n, dtype=np.float64)
            else:
                w = w / total

        Ts = [
            T_ref_to_tag @ np.linalg.inv(T_base_to_tag)
            for T_ref_to_tag, T_base_to_tag, _ in pairs
        ]
        # Weighted translation average.
        t_avg = np.sum(
            [wi * T[:3, 3] for wi, T in zip(w, Ts)], axis=0,
        )
        qs = [R.from_matrix(sanitize_rotation(T[:3, :3])).as_quat() for T in Ts]
        # Keep quaternions in the same hemisphere before averaging.
        q0 = qs[0]
        for i in range(1, len(qs)):
            if np.dot(q0, qs[i]) < 0.0:
                qs[i] = -qs[i]
        q_avg = np.sum([wi * q for wi, q in zip(w, qs)], axis=0)
        norm = np.linalg.norm(q_avg)
        if norm < 1e-6:
            return None
        q_avg /= norm
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q_avg).as_matrix()
        T[:3, 3] = t_avg
        return T

    # ------------------------------------------------------------------- Tick
    def _tick(self) -> None:
        ref_frame = str(self.get_parameter("reference_frame").value)
        out_frame = str(self.get_parameter("output_frame").value)
        max_age = float(self.get_parameter("detection_max_age").value)
        min_pairs = int(self.get_parameter("min_pairs").value)
        tf_timeout = float(self.get_parameter("tf_timeout").value)
        publish_tag_tfs = bool(self.get_parameter("publish_tag_tfs").value)

        # Use the ROS clock for the freshness check. Comparing message
        # header time (ROS clock) against ``time.time()`` (wall clock)
        # is wrong when ``use_sim_time`` is on or when wall-clock and
        # ROS clock diverge (rosbag replay, paused time).
        now_ns = self.get_clock().now().nanoseconds
        max_age_ns = int(max_age * 1e9)
        # Snapshot the cache so the solve doesn't race the subscribers.
        with self._lock:
            snapshot = list(self._latest.items())

        # Build (T_ref_to_tag, T_base_to_tag, (family, tag_id)) for each fresh detection.
        pairs: List[Tuple[np.ndarray, np.ndarray, Tuple[str, int]]] = []
        # Per-tag raw estimates for the TF broadcast, keyed by
        # (family, tag_id): (distance_to_tag, T_ref_to_tag). When two
        # cameras see the same tag in one tick the closest observation
        # wins.
        tag_estimates: Dict[Tuple[str, int], Tuple[float, np.ndarray]] = {}
        stale = 0
        for key, (stamp_ns, cam_frame, T_cam_to_tag) in snapshot:
            if (now_ns - stamp_ns) > max_age_ns:
                stale += 1
                continue
            if not cam_frame:
                continue
            # Lookup T_ref -> cam_frame.
            try:
                tf_stamped = self._tf_buffer.lookup_transform(
                    ref_frame, cam_frame,
                    rclpy.time.Time(),  # latest
                    timeout=rclpy.duration.Duration(seconds=tf_timeout),
                )
            except Exception as e:
                # Only log occasionally; per-tick spam is unhelpful.
                if not hasattr(self, "_last_tf_warn") or (
                    now_ns - self._last_tf_warn > 5_000_000_000
                ):
                    self.get_logger().warn(
                        f"TF {ref_frame} -> {cam_frame} lookup failed: {e} "
                        f"(will retry; subsequent failures are throttled)"
                    )
                    self._last_tf_warn = now_ns
                continue
            T_ref_to_cam = _tf_to_matrix(tf_stamped)
            T_ref_to_tag = T_ref_to_cam @ T_cam_to_tag
            _topic, family, tag_id = key
            tag_key = (family, tag_id)
            if tag_key not in self._tag_transforms:
                # No URDF link for this tag; skip rather than crash.
                continue
            T_base_to_tag = self._tag_transforms[tag_key].T_base_to_tag
            pairs.append((T_ref_to_tag, T_base_to_tag, tag_key))
            dist = float(np.linalg.norm(T_ref_to_tag[:3, 3]))
            prev = tag_estimates.get(tag_key)
            if prev is None or dist < prev[0]:
                tag_estimates[tag_key] = (dist, T_ref_to_tag)

        # Drop stale entries from the cache so they don't accumulate forever.
        if stale:
            with self._lock:
                self._latest = {
                    k: v for k, v in self._latest.items()
                    if (now_ns - v[0]) <= max_age_ns
                }

        if not pairs:
            return

        now_msg = self.get_clock().now().to_msg()

        # Per-tag TFs are raw per-observation poses and do not depend
        # on the joint solve below succeeding.
        if publish_tag_tfs:
            tag_msgs = [
                matrix_to_transform_stamped(
                    T_tag, ref_frame, f"apriltag_{family}_{tag_id}", now_msg,
                )
                for (family, tag_id), (_dist, T_tag) in sorted(tag_estimates.items())
            ]
            if tag_msgs:
                self._tf_broadcaster.sendTransform(tag_msgs)

        # Per-pair weights (closer tags dominate; see _detection_weights).
        weights = self._detection_weights(pairs)

        # Solve. (T_cam_to_tag_observed, T_base_to_tag_known) -> T_cam_to_base
        # Here the "cam" frame is the reference frame, so we solve for
        # T_ref_to_base.
        #
        # * With >= 4 pairs, run the RANSAC wrapper -- it draws random
        #   3-pair subsets, scores by inlier count, and refits on the
        #   best consensus set. This is robust against single-tag
        #   outliers (e.g. a mis-classified id, corner flip, sensor
        #   glitch) that would otherwise pull the entire solve.
        # * With exactly 3 pairs, the joint Umeyama solve is already
        #   exact -- no point spending RANSAC iterations.
        # * With fewer than 3 pairs the joint solve is under-determined;
        #   fall back to the per-tag inverse averaging so the pose
        #   never goes silent as long as at least one tag is visible.
        T_ref_to_base = None
        raw_pairs = [(T_r_t, T_b_t) for T_r_t, T_b_t, _ in pairs]
        if len(pairs) >= max(min_pairs, 4):
            T_ref_to_base = solve_cam_to_base_ransac(raw_pairs, weights=weights)
        elif len(pairs) >= min_pairs:
            T_ref_to_base = solve_cam_to_base(raw_pairs, weights=weights)

        if T_ref_to_base is None and pairs:
            T_ref_to_base = self._fallback_solve(pairs, weights=weights)
            if T_ref_to_base is not None:
                self.get_logger().debug(
                    f"Used per-tag fallback with {len(pairs)} pair(s) "
                    f"(min_pairs={min_pairs})"
                )

        if T_ref_to_base is None:
            return

        # T_base_to_dock is a fixed rigid offset from the URDF; composing it
        # onto the solved T_ref_to_base gives T_ref_to_dock (or to the base
        # frame itself when no dock_link_name is configured).
        T_base_to_dock = self._T_base_to_dock
        if T_base_to_dock is None:
            T_base_to_dock = np.eye(4)
        T_ref_to_out = T_ref_to_base @ T_base_to_dock

        self._tf_broadcaster.sendTransform(
            matrix_to_transform_stamped(
                T_ref_to_out, ref_frame, out_frame, now_msg,
            )
        )
        if self._pose_pub is not None:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = now_msg
            pose_msg.header.frame_id = ref_frame
            pose_msg.pose = matrix_to_pose_msg(T_ref_to_out)
            self._pose_pub.publish(pose_msg)


def _tf_to_matrix(tf_stamped: TransformStamped) -> np.ndarray:
    """geometry_msgs/TransformStamped -> 4x4 homogeneous transform."""
    t = tf_stamped.transform.translation
    q = tf_stamped.transform.rotation
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def main() -> None:
    rclpy.init()
    node = AprilTagFuserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()