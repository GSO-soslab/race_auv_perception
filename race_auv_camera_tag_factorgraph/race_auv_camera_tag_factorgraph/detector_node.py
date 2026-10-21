"""ROS 2 node ``tag_detector_node``: camera image -> tag36h11 corners + poses.

One node per camera. Each timer tick (``publish_rate``, 5 Hz on the vehicle)
takes the newest image, and if it is new:

1. decodes it (straight to grayscale unless the debug image is on),
2. rectifies it (:mod:`rectify`),
3. detects, checks and poses the configured tags (:mod:`tags`),
4. publishes, all stamped with the **image** stamp and the camera frame:

   ``apriltag_detection/tags`` (``apriltag_msgs/AprilTagDetectionArray``)
       The 4 corner pixels of each accepted tag in the rectified image,
       canonical order ``lb, rb, rt, lt``. This is dock_fuser's input. It
       is published for *every* processed image, also with no tags: each
       message becomes one dock_fuser keyframe.
   ``apriltag_detection/detections3d`` (``vision_msgs/Detection3DArray``)
       The IPPE pose ``T_cam_tag`` of each accepted tag (``id`` =
       ``"tag36h11:<id>"``), for other consumers and visualization.
   ``apriltag_detection/camera_info`` (``sensor_msgs/CameraInfo``, latched)
       The rectified camera matrix the corners live in. dock_fuser's
       ``cameras.<name>.fx/fy/cx/cy/width/height`` must equal it.
   ``apriltag_detection/image/compressed`` (``sensor_msgs/CompressedImage``)
       Only with ``publish_image: true``: the rectified image with the tag
       outlines, IDs and axes drawn (costs a JPEG encode per frame). Named
       the image_transport way (base topic ``apriltag_detection/image``,
       transport ``compressed``).

Timing: the camera publishes faster (15 Hz) than the detector runs. The
image callback only stores the newest message; decoding happens in the
tick, so skipped frames cost nothing. A tick with no new image since the
previous one is skipped (no re-published duplicate stamps).

Synced mode (``sync_to``, e.g. cam_down synced to cam_front): no timer.
Every message on ``sync_to`` (another detector's ``apriltag_detection/tags``)
makes this node process its own buffered image nearest that message's
stamp (:mod:`sync`), within ``sync_max_dt``. The two cameras' detections
are then at most half a camera frame apart (33 ms at 15 Hz) instead of up
to a whole detector period (0.2 s), which is what dock_fuser bridges with
the EKF when it attaches this camera to the other's keyframes.

Parameters: see ``race_auv_bringup/config/tag_factorgraph_detector.yaml``.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import rclpy
from apriltag_msgs.msg import AprilTagDetection, AprilTagDetectionArray
from apriltag_msgs.msg import Point as TagPoint
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .rectify import Rectifier
from .sync import ImageSync
from .tags import FAMILY, FrameResult, TagDetector, select_tags

# Camera input: only the newest image matters; anything queued is stale.
_SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
# Detections: reliable with a small queue. Every message is a dock_fuser
# keyframe, so none should be lost while the fuser is briefly busy.
_OUTPUT_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
# Rectified CameraInfo: published once, delivered to late subscribers too.
_LATCHED_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL,
                          history=HistoryPolicy.KEEP_LAST, depth=1)


def _pose_msg(T: np.ndarray) -> Pose:
    """4x4 homogeneous transform -> ``geometry_msgs/Pose`` (quaternion via Rodrigues)."""
    rvec = cv2.Rodrigues(T[:3, :3])[0].ravel()
    angle = float(np.linalg.norm(rvec))
    axis = rvec / angle if angle > 1e-12 else np.array([1.0, 0.0, 0.0])
    p = Pose()
    p.position.x, p.position.y, p.position.z = (float(v) for v in T[:3, 3])
    s = np.sin(0.5 * angle)
    p.orientation.x, p.orientation.y, p.orientation.z = (float(v) for v in axis * s)
    p.orientation.w = float(np.cos(0.5 * angle))
    return p


class TagDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__("tag_detector")
        # ------------------------------------------------------------ parameters
        p = {
            "image_topic": "",
            "image_transport": "compressed",     # "compressed" (JPEG) or "raw"
            "camera_frame": "",                  # header.frame_id of every output
            "publish_rate": 5.0,                 # Hz; = dock_fuser keyframe rate
            # Raw (distorted) intrinsics of the camera.
            "intrinsics.fx": 0.0, "intrinsics.fy": 0.0,
            "intrinsics.cx": 0.0, "intrinsics.cy": 0.0,
            "intrinsics.width": 0, "intrinsics.height": 0,
            "intrinsics.distortion": [0.0, 0.0, 0.0, 0.0],
            "intrinsics.fisheye": True,
            # tag36h11 IDs and black-square sides [m], same length: the
            # station's tag list (station_tags.yaml).
            "tag_ids": [0],
            "tag_sizes": [0.0],
            # The subset of tag_ids this camera reports ([-1] = all of them).
            "camera_tag_ids": [-1],
            # Synced mode: process the image nearest each message of this
            # topic (another detector's tags) instead of running a timer.
            "sync_to": "",
            "sync_max_dt": 0.05,       # s; half a 15 Hz frame plus clock slack
            "sync_wait": 0.1,          # s of reference time to wait for a nearer image
            # apriltag3.
            "threads": 4,
            "quad_decimate": 2.0,
            "refine_edges": True,
            # Image checks (tags.py).
            "min_edge_px": 10.0,
            "min_side_px": 15.0,
            "max_fit_rms_px": 1.5,
            # Debug image.
            "publish_image": False,
            "jpeg_quality": 80,
        }
        for name, default in p.items():
            self.declare_parameter(name, default)
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        # ------------------------------------------------------------ models
        fx, fy, cx, cy = (float(g(f"intrinsics.{k}")) for k in ("fx", "fy", "cx", "cy"))
        size = (int(g("intrinsics.width")), int(g("intrinsics.height")))
        if min(fx, fy, *size) <= 0:
            raise ValueError("intrinsics.fx/fy/width/height must be set (raw camera calibration)")
        tag_sizes = select_tags(list(g("tag_ids")), list(g("tag_sizes")), list(g("camera_tag_ids")))
        ids, sizes = list(tag_sizes), list(tag_sizes.values())
        self.rectifier = Rectifier(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                                   g("intrinsics.distortion"), size, bool(g("intrinsics.fisheye")))
        r = self.rectifier
        self.detector = TagDetector(
            tag_sizes, r.K, r.width, r.height,
            threads=int(g("threads")), decimate=float(g("quad_decimate")),
            refine_edges=bool(g("refine_edges")), min_edge_px=float(g("min_edge_px")),
            min_side_px=float(g("min_side_px")), max_fit_rms_px=float(g("max_fit_rms_px")))
        self.frame_id = str(g("camera_frame"))
        self.publish_image = bool(g("publish_image"))
        self.jpeg_quality = int(g("jpeg_quality"))

        # ------------------------------------------------------------ state
        # The image callback (writer) and the timer (reader) may run on
        # different threads under a multi-threaded executor.
        self._lock = threading.Lock()
        self._latest = None              # newest image message, not yet decoded
        self._last_stamp = None          # (sec, nanosec) of the image processed last
        self._n_rejected = 0

        # ------------------------------------------------------------ ROS I/O
        compressed = str(g("image_transport")) == "compressed"
        self.create_subscription(CompressedImage if compressed else Image, str(g("image_topic")),
                                 self._image_cb, _SENSOR_QOS)
        self.tags_pub = self.create_publisher(AprilTagDetectionArray, "apriltag_detection/tags",
                                              _OUTPUT_QOS)
        self.poses_pub = self.create_publisher(Detection3DArray, "apriltag_detection/detections3d",
                                               _OUTPUT_QOS)
        self.info_pub = self.create_publisher(CameraInfo, "apriltag_detection/camera_info",
                                              _LATCHED_QOS)
        # image_transport naming (<base>/compressed for a CompressedImage), so
        # RViz / rqt decode it through the compressed plugin; Foxglove takes it as is.
        self.image_pub = (self.create_publisher(CompressedImage,
                                                "apriltag_detection/image/compressed",
                                                _OUTPUT_QOS) if self.publish_image else None)
        if not compressed:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        self._publish_camera_info()
        self.sync_to = str(g("sync_to") or "")
        if self.sync_to:
            self._sync = ImageSync(float(g("sync_max_dt")), float(g("sync_wait")))
            self.create_subscription(AprilTagDetectionArray, self.sync_to, self._sync_cb,
                                     _OUTPUT_QOS)
        else:
            self.create_timer(1.0 / float(g("publish_rate")), self._tick)

        self.get_logger().info(
            f"{FAMILY} ids {ids} sizes {sizes} on {g('image_topic')} at {g('publish_rate')} Hz; "
            f"{r.describe()} (latched on apriltag_detection/camera_info for dock_fuser)"
            + (f"; synced to {self.sync_to} (nearest image within {g('sync_max_dt')} s)"
               if self.sync_to else ""))

    # ==================================================================== input
    def _image_cb(self, msg) -> None:
        """Keep only the newest image; decoding is deferred to the tick (or the sync match)."""
        with self._lock:
            self._latest = msg
        if self.sync_to:
            self._sync.add_image(_sec(msg.header.stamp), msg)
            self._process_synced()

    def _sync_cb(self, ref: AprilTagDetectionArray) -> None:
        """A reference detection: process this camera's image nearest its stamp (synced mode)."""
        self._sync.add_reference(_sec(ref.header.stamp))
        self._process_synced()

    def _process_synced(self) -> None:
        for _, _, msg in self._sync.pop_ready():
            self._process(msg)

    def _decode(self, msg) -> np.ndarray:
        """Image message -> numpy array (grayscale, or BGR when drawing the debug image)."""
        if isinstance(msg, CompressedImage):
            flag = cv2.IMREAD_COLOR if self.publish_image else cv2.IMREAD_GRAYSCALE
            return cv2.imdecode(np.frombuffer(msg.data, np.uint8), flag)
        enc = "bgr8" if self.publish_image else "mono8"
        return self._bridge.imgmsg_to_cv2(msg, desired_encoding=enc)

    # ==================================================================== tick
    def _tick(self) -> None:
        with self._lock:
            msg = self._latest
        if msg is None:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self._last_stamp:
            return   # no new image since the last tick: nothing to publish
        self._last_stamp = stamp
        self._process(msg)

    def _process(self, msg) -> None:
        """Decode, rectify, detect and publish one image (tick or sync match)."""
        t0 = time.perf_counter()
        img = self._decode(msg)
        if img is None:
            self.get_logger().warn("could not decode image", throttle_duration_sec=5.0)
            return
        rect = self.rectifier.rectify(img)
        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) if rect.ndim == 3 else rect
        res = self.detector.detect(gray)
        self._publish(res, msg.header.stamp)
        if self.image_pub is not None:
            self._publish_image(rect, res, msg.header.stamp)

        self._n_rejected += len(res.rejected)
        for tag_id, reason in res.rejected:
            self.get_logger().debug(f"{FAMILY}:{tag_id} rejected: {reason}")
        self.get_logger().info(
            f"{len(res.tags)} tags, {len(res.rejected)} rejected (total {self._n_rejected}), "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms", throttle_duration_sec=10.0)

    # ==================================================================== output
    def _publish(self, res: FrameResult, stamp) -> None:
        tags = AprilTagDetectionArray()
        poses = Detection3DArray()
        tags.header.stamp = poses.header.stamp = stamp
        tags.header.frame_id = poses.header.frame_id = self.frame_id
        for t in res.tags:
            d = AprilTagDetection()
            d.family = FAMILY
            d.id = t.tag_id
            d.hamming = t.hamming
            d.decision_margin = float(t.margin)
            d.centre = TagPoint(x=float(t.corners[:, 0].mean()), y=float(t.corners[:, 1].mean()))
            d.corners = [TagPoint(x=float(u), y=float(v)) for u, v in t.corners]
            tags.detections.append(d)

            d3 = Detection3D()
            d3.header = poses.header
            d3.id = f"{FAMILY}:{t.tag_id}"
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(t.tag_id)
            hyp.hypothesis.score = 1.0
            hyp.pose.pose = _pose_msg(t.T_cam_tag)
            d3.results = [hyp]
            poses.detections.append(d3)
        self.tags_pub.publish(tags)
        self.poses_pub.publish(poses)

    def _publish_camera_info(self) -> None:
        """Latch the rectified intrinsics once (distortion-free pinhole).

        Transient-local QoS delivers this one message to every subscriber,
        also to those that start later.
        """
        r = self.rectifier
        info = CameraInfo()
        info.header.stamp = self.get_clock().now().to_msg()
        info.header.frame_id = self.frame_id
        info.width, info.height = r.width, r.height
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [float(v) for v in r.K.ravel()]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(r.K[0, 0]), 0.0, float(r.K[0, 2]), 0.0,
                  0.0, float(r.K[1, 1]), float(r.K[1, 2]), 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)

    def _publish_image(self, rect: np.ndarray, res: FrameResult, stamp) -> None:
        """Rectified image with tag outlines (green), IDs and pose axes, as JPEG."""
        img = rect if rect.ndim == 3 else cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)
        K = self.detector.K
        for t in res.tags:
            c = t.corners.astype(np.int32)
            cv2.polylines(img, [c], True, (0, 255, 0), 2)
            cv2.putText(img, str(t.tag_id), tuple(c[3] + (0, -6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)
            cv2.drawFrameAxes(img, K, None, cv2.Rodrigues(t.T_cam_tag[:3, :3])[0],
                              t.T_cam_tag[:3, 3], 0.5 * t.size)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if ok:
            out = CompressedImage(format="jpeg", data=buf.tobytes())
            out.header.stamp = stamp
            out.header.frame_id = self.frame_id
            self.image_pub.publish(out)


def _sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TagDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
