"""ROS 2 node ``image_rotate_node``: turn a compressed camera image by 180 deg, for display.

The down camera (stellar1, ``cam_down``) is mounted upside down relative to
how it should be watched: its picture needs flipping horizontally *and*
vertically, which is a 180 deg rotation. RViz's Image display cannot rotate,
so this node does it on the viewing computer (topside) and republishes:

    in   ``input``   (``sensor_msgs/CompressedImage``), e.g. /race/stellar1/image/compressed
    out  ``output``  (``sensor_msgs/CompressedImage``), e.g. /race/stellar1/image_rot180/compressed

Display only: the vehicle, the tag detector and the factor graph keep using
the camera's own image (and its calibration, which is for that orientation).
Started by race_auv_bringup/launch/bringup_rviz_factorgraph.launch.py.

Cost: one JPEG decode + encode per frame on the computer running it. Only
the newest frame matters for a live view, so the subscription keeps one
message (best effort) and frames are never queued.
"""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

# Newest frame only, never a backlog; best effort matches any camera publisher.
_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                  history=HistoryPolicy.KEEP_LAST, depth=1)


class ImageRotateNode(Node):
    def __init__(self) -> None:
        super().__init__("image_rotate")
        self.declare_parameter("input", "/race/stellar1/image/compressed")
        self.declare_parameter("output", "/race/stellar1/image_rot180/compressed")
        self.declare_parameter("jpeg_quality", 80)
        self.quality = int(self.get_parameter("jpeg_quality").value)
        out = self.get_parameter("output").value
        self.pub = self.create_publisher(CompressedImage, out, _QOS)
        self.create_subscription(CompressedImage, self.get_parameter("input").value,
                                 self._cb, _QOS)
        self.get_logger().info(f"{self.get_parameter('input').value} -> {out} (rotated 180 deg)")

    def _cb(self, msg: CompressedImage) -> None:
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn("could not decode image", throttle_duration_sec=5.0)
            return
        ok, buf = cv2.imencode(".jpg", cv2.rotate(img, cv2.ROTATE_180),
                               [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        out = CompressedImage()
        out.header = msg.header          # same stamp and frame: only the picture is turned
        out.format = "jpeg"
        out.data = buf.tobytes()
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    node = ImageRotateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
