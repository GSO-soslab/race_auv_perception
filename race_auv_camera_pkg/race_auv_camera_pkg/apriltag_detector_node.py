"""Per-camera AprilTag detector node (no TF, CPU-only).

A single instance of this node handles one camera. It subscribes to
either a raw ``sensor_msgs/Image`` or a ``sensor_msgs/CompressedImage``
topic, plus an optional ``sensor_msgs/CameraInfo`` topic, runs the
:class:`race_auv_camera_pkg.apriltag_processor.AprilTagDetector` on
each frame, and publishes:

* ``vision_msgs/Detection3DArray`` on ``output_detections_topic`` with
  one entry per detected tag. ``Detection3D.id`` is
  ``f"{family}:{tag_id}"`` and ``results[0].pose.pose`` is the tag's
  pose in the camera frame.
* An annotated ``sensor_msgs/CompressedImage`` on
  ``output_image_topic``: tag bounding boxes + id labels + pose
  overlays, plus a red crosshair at the image center. This is the
  image shown by Foxglove / topside.

This node does NOT publish any TF. The original pipeline in
``race_auv_sim_pkg/apriltag_detector_node.py`` broadcast per-tag frames
(``apriltag<family>_<id>``) for the multi-camera fuser; that
responsibility moved to ``apriltag_fuser_node`` (which now broadcasts
``<reference_frame> -> apriltag_<family>_<id>`` per freshly-detected
tag, toggleable via its ``publish_tag_tfs`` param). Here, annotated
images + ``Detection3DArray`` topics are the only outputs.

Multi-family / multi-size tags
------------------------------
The detector instantiates one ``apriltag.apriltag(family=...)`` per
configured family inside a single ``AprilTagDetector``. Per-tag size
is looked up *after* decode from the YAML ``tags:`` list, so tags of
different sizes within the same family share one quad-scan instead
of being re-decoded for each size bucket.

Two sources for the tag list:

* ``tags_override`` (JSON-encoded list of ``{id, family, size}``
  dicts) -- set by the per-camera launch file from ``apriltag.yaml``.
  When provided and non-empty, this list drives the detector and the
  YAML ``tags:`` list is ignored.
* Otherwise the YAML ``tags:`` list (the global fallback) is used.

Hardware acceleration
---------------------
Image-pipeline acceleration via ``cv2.cuda`` and NVIDIA ``nvjpeg``
remains removed from this package: neither worked reliably on the
target platforms (Jetson Orin + generic Linux), because the PyPI
OpenCV wheels do not include CUDA and the JetPack system package in
the version we target either ships without NVCOMPRESS or without the
required CUDA runtime. The default image path is therefore CPU-only:
``cv_bridge`` decode, the CPU ``ImageRectifier`` and ``cv2.imencode``.

Two optional GPU paths are available:

* ``detector_backend: "cuda"`` runs tag *detection* on the GPU with an
  in-process cuAprilTags backend (``race_auv_apriltag_cuda``). It is
  imported lazily and decodes ``tag36h11`` only; ``"python"``
  (apriltag3) is the default and supports every family.
* ``image_pipeline: "cuda"`` (requires ``image_transport:
  "compressed"`` and ``detector_backend: "cuda"``) replaces the whole
  image path with in-process NVIDIA stages: nvjpeg decode, VPI CUDA
  fisheye rectify and nvjpeg encode. The detection-resolution frame
  stays in device memory and is fed straight to cuAprilTags; only the
  full-resolution rectified frame is copied back for annotation.

The historical ``use_cuda`` / ``jpeg_backend`` keys are still read but
only logged -- they no longer change behaviour.

Per-camera performance knobs (CPU pipeline):

* ``process_scale`` (per-camera, default 1.0) -- detect on a
  downscaled image; annotation stays at full resolution. ``0.5``
  gives ~3-5x faster detection with negligible accuracy loss for
  dock-sized tags.
* ``jpeg_quality`` (per-camera, default 80) -- quality for the
  annotated ``CompressedImage`` (Foxglove bandwidth knob).

When ``info_topic`` is empty, the node falls back to intrinsics
supplied via ROS parameters (typically populated by the launch file
from ``apriltag.yaml``). This is the path used for the DWE camera
driver, which does not publish ``CameraInfo`` by default.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseWithCovariance, Vector3
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from vision_msgs.msg import (
    BoundingBox3D, Detection3D, Detection3DArray, ObjectHypothesis,
    ObjectHypothesisWithPose,
)

from .apriltag_geom import (
    is_bad_rotation, load_yaml_config, matrix_to_pose_msg, sanitize_rotation,
)
from .apriltag_processor import AprilTagDetector
from .image_processing import build_rectifier


# OpenCV's JPEG-quality param-tag. Used as the first element of the
# ``params=[...]`` list passed to :func:`cv2.imencode`. Kept as a
# module-level constant so the value is looked up once at import time
# instead of on every encode.
_IMWRITE_JPEG_QUALITY = int(cv2.IMWRITE_JPEG_QUALITY)


# =============================================================================
# Module-level QoS profiles
# =============================================================================
# Subscriptions: best-effort, low depth. The detector always works on the
# latest frame; queued frames are wasted CPU. ``depth=1`` makes the
# subscriber drop any in-flight message as soon as a newer one arrives,
# which matches "I only care about the most recent frame" semantics.
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
# Publications: reliable, low depth. ``foxglove_bridge`` relays them
# as-is so we want the same semantics the user expects from a
# CompressedImage stream.
_OUTPUT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


# =============================================================================
# Crosshair drawing constants
# =============================================================================
# The crosshair is drawn at the image center after rectification so the
# operator can sanity-check that the principal point of the rectified
# intrinsics lines up with the optical axis.
_CROSSHAIR_COLOR = (0, 0, 255)        # BGR red
_CROSSHAIR_THICKNESS = 2
_CROSSHAIR_ARM_DIVISOR = 15           # arm = min(H, W) // 15 (~3% of min dim)


# =============================================================================
# Small parameter coercers
# =============================================================================
def _as_float(v: Any, default: float = 0.0) -> float:
    """Coerce ``v`` to ``float``; return ``default`` on ``TypeError``/``ValueError``.

    Used for the YAML-intrinsics fallback path where the parameter
    server may hand us ``None`` or a non-numeric string when a
    default has not been set yet.
    """
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _as_int(v: Any, default: int = 0) -> int:
    """Coerce ``v`` to ``int``; return ``default`` on ``TypeError``/``ValueError``."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Crosshair helper
# =============================================================================
def _draw_crosshair(image: np.ndarray) -> None:
    """Draw a small red ``+`` at the centre of ``image`` in-place.

    The arm length is ``max(1, min(H, W) // _CROSSHAIR_ARM_DIVISOR)`` so
    it scales with the image: ~3% of the shorter side.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    arm = max(1, min(h, w) // _CROSSHAIR_ARM_DIVISOR)
    cv2.line(image, (cx - arm, cy), (cx + arm, cy),
             _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS, cv2.LINE_AA)
    cv2.line(image, (cx, cy - arm), (cx, cy + arm),
             _CROSSHAIR_COLOR, _CROSSHAIR_THICKNESS, cv2.LINE_AA)


class AprilTagDetectorNode(Node):
    """Per-camera detector: image (+ optional CameraInfo) -> detections + image.

    Lifecycle
    ---------
    1. ``__init__`` declares all parameters and loads the YAML config.
    2. We wait for either a ``CameraInfo`` message (``info_topic`` set)
       or until the YAML intrinsics are populated (``info_topic``
       empty). In either case we call :meth:`_build_pipeline` once.
    3. A timer (``publish_rate`` Hz) drains the latest received frame
       and runs the rectification + detection + annotation + publish
       pipeline via :meth:`_tick`.
    """

    def __init__(self) -> None:
        super().__init__("apriltag_detector")

        # ============================================================ params
        # YAML config path (set by the launch file).
        self.declare_parameter("config_yaml", "")

        # Transport / topics.
        self.declare_parameter("image_transport", "raw")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("info_topic", "")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("output_image_topic", "apriltag_detection/image")
        self.declare_parameter("output_detections_topic", "apriltag_detection/detections3d")
        self.declare_parameter("publish_rate", 5.0)

        # Tag list: per-camera JSON override (set by launch), else YAML tags.
        self.declare_parameter("tags_override", "[]")

        # Per-camera performance knobs.
        self.declare_parameter("process_scale", 1.0)
        self.declare_parameter("jpeg_quality", 80)

        # Minimum pixel distance from a detected tag's edge to the
        # rectified image border. Detections closer than this are
        # dropped (see ``_filter_edge_clipped``).
        self.declare_parameter("min_edge_dist", 10)

        # Detector backend: "python" (apriltag3 CPU, all families) or
        # "cuda" (in-process cuAprilTags, tag36h11 only).
        self.declare_parameter("detector_backend", "python")
        # CUDA backend knobs (ignored by the python backend).
        self.declare_parameter("cuda_nominal_size", 0.125)
        self.declare_parameter("cuda_tile_size", 4)
        self.declare_parameter("cuda_max_tags", 64)

        # Image-stage backend:
        #   "cpu"  -- cv_bridge/cv2 decode + CPU rectify + cv2.imencode
        #   "cuda" -- nvjpeg decode + VPI CUDA rectify + nvjpeg encode.
        #             Requires image_transport="compressed" and
        #             detector_backend="cuda"; ignored (with a warning)
        #             otherwise.
        self.declare_parameter("image_pipeline", "cpu")
        # VPI warp grid spacing for the GPU rectifier (power of two).
        self.declare_parameter("gpu_rectify_interval", 4)

        # Intrinsics fallback (used when ``info_topic`` is empty).
        self.declare_parameter("intrinsics.fx", 0.0)
        self.declare_parameter("intrinsics.fy", 0.0)
        self.declare_parameter("intrinsics.cx", 0.0)
        self.declare_parameter("intrinsics.cy", 0.0)
        self.declare_parameter("intrinsics.width", 0)
        self.declare_parameter("intrinsics.height", 0)
        self.declare_parameter("intrinsics.distortion", [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("intrinsics.fisheye", False)

        # ============================================================= state
        # The bridge converts ROS Image/CompressedImage <-> numpy BGR.
        self._bridge = CvBridge()
        # All shared state is touched from the image callback (writer)
        # and the timer tick (reader); protect with this lock.
        self._lock = threading.Lock()
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_stamp = None
        self._info_msg: Optional[CameraInfo] = None
        # The single detector instance (python ``AprilTagDetector`` or
        # CUDA ``CuAprilTagDetector``). Both expose detect()/annotate()
        # and look up per-tag size from ``_group_tags_by_family`` after
        # decode. ``None`` until ``_build_pipeline`` runs.
        self._detector: Optional[object] = None
        # Selected backend name ("python" or "cuda"); decides whether
        # _process_frame feeds grayscale or BGR to the detector.
        self._detector_backend: str = "python"
        # GPU image pipeline (nvjpeg/VPI); None on the CPU image path.
        self._gpu_pipeline: Optional[object] = None
        # Selected image-stage backend ("cpu" or "cuda").
        self._image_pipeline: str = "cpu"
        # Transport selected in __init__ ("raw" or "compressed").
        self._image_transport: str = "raw"
        # Latest compressed frame bytes, kept undecoded on the GPU path
        # and decoded once per tick instead of once per camera frame.
        self._latest_jpeg: Optional[bytes] = None
        # The single CPU ``ImageRectifier`` instance.
        self._rectifier: Optional[object] = None
        self._rectifier_backend: str = "cpu"
        # JPEG quality cached at pipeline-build time so the encode
        # path does not need to read a ROS parameter on every frame.
        self._jpeg_quality: int = 80
        self._process_scale: float = 1.0
        # Cached at pipeline-build time; see ``_filter_edge_clipped``.
        self._min_edge_dist: int = 10
        # ``_ready`` flips true once ``_build_pipeline`` has run; the
        # tick callback no-ops until then so we don't publish empties
        # before the first CameraInfo arrives.
        self._ready = False
        # Running counter of detections dropped because of a bad pose;
        # used to throttle the per-tick warning spam.
        self._bad_pose_count = 0

        # Load tag list + detector defaults from YAML. The historical
        # ``use_cuda`` / ``jpeg_backend`` keys are read here but only
        # logged at startup -- they no longer change behaviour.
        self._tags_config: List[Dict] = []
        self._detector_params_template: Dict = {}
        self._load_yaml_config()

        # ============================================================ subscriptions
        transport = str(self.get_parameter("image_transport").value or "raw").lower()
        self._image_transport = transport
        image_topic = str(self.get_parameter("image_topic").value or "")
        info_topic = str(self.get_parameter("info_topic").value or "")
        if not image_topic:
            raise RuntimeError("image_topic parameter is required")

        # Pick the right subscription type / callback based on the
        # transport: "compressed" subscribes to ``CompressedImage``
        # directly (no cv_bridge decode cost on the wire) and
        # "raw" subscribes to the normal ``Image`` topic.
        image_type = CompressedImage if transport == "compressed" else Image
        image_cb = self._image_cb_compressed if transport == "compressed" else self._image_cb_raw
        self.image_sub = self.create_subscription(image_type, image_topic, image_cb, _SENSOR_QOS)

        if info_topic:
            self.info_sub = self.create_subscription(CameraInfo, info_topic, self._info_cb, _SENSOR_QOS)
        else:
            self.info_sub = None
            # No CameraInfo on the wire -- build the pipeline straight
            # from the YAML intrinsics.
            self._try_build_from_yaml_intrinsics()

        # ============================================================= publications
        self.image_pub = self.create_publisher(
            CompressedImage, self.get_parameter("output_image_topic").value, _OUTPUT_QOS,
        )
        self.detections_pub = self.create_publisher(
            Detection3DArray, self.get_parameter("output_detections_topic").value, _OUTPUT_QOS,
        )

        # ============================================================= tick / init
        rate = float(self.get_parameter("publish_rate").value)
        if rate <= 0.0:
            raise ValueError("publish_rate must be > 0")
        self._timer = self.create_timer(1.0 / rate, self._tick)

        self.get_logger().info(
            f"apriltag_detector ready: image={image_topic} "
            f"info={info_topic or '<yaml intrinsics>'} transport={transport} "
            f"rate={rate}Hz tags={len(self._tags_config)}"
        )

    # =====================================================================
    # YAML / configuration
    # =====================================================================
    def _load_yaml_config(self) -> None:
        """Populate ``_tags_config`` and ``_detector_params_template``.

        ``tags_override`` (a JSON-encoded list of ``{id, family, size}``)
        takes precedence when non-empty; otherwise the YAML ``tags:``
        list is used. Detector defaults always come from the YAML.

        The historical ``detector_defaults.use_cuda`` and
        ``detector_defaults.jpeg_backend`` keys are read and stored
        only so the startup banner can report them -- they no longer
        switch behaviour because the GPU paths were removed.
        """
        cfg_path = str(self.get_parameter("config_yaml").value or "")
        inner: Dict = {}
        det_cfg: Dict = {}
        if cfg_path:
            inner = load_yaml_config(cfg_path)
            det_cfg = inner.get("detector_defaults", {}) or {}

        self._detector_params_template = {
            "nthreads": int(det_cfg.get("nthreads", 4)),
            "quad_decimate": float(det_cfg.get("quad_decimate", 2.0)),
            "quad_sigma": float(det_cfg.get("quad_sigma", 0.0)),
            "refine_edges": bool(det_cfg.get("refine_edges", True)),
        }

        # Historical HW-accel knobs -- read so we can log them, but
        # they do not switch behaviour anymore.
        use_cuda_yaml = bool(det_cfg.get("use_cuda", False))
        jpeg_backend_yaml = str(det_cfg.get("jpeg_backend", "cpu")).lower()
        if use_cuda_yaml or jpeg_backend_yaml not in ("", "cpu", "auto"):
            self.get_logger().warn(
                f"YAML 'detector_defaults.use_cuda={use_cuda_yaml}' / "
                f"'detector_defaults.jpeg_backend={jpeg_backend_yaml!r}' "
                "are ignored: the cv2.cuda and NVIDIA nvjpeg paths were "
                "removed from this package. Running on the CPU."
            )

        tags_cfg: List[Dict] = []
        try:
            parsed = json.loads(str(self.get_parameter("tags_override").value or "[]"))
        except json.JSONDecodeError as e:
            self.get_logger().warn(
                f"Could not parse tags_override JSON ({e!r}); falling back to YAML tags."
            )
            parsed = []
        if isinstance(parsed, list) and parsed:
            tags_cfg = parsed
            self.get_logger().info(f"Using tags_override ({len(tags_cfg)} entries).")
        else:
            tags_cfg = inner.get("tags", []) or []
            self.get_logger().info(f"Using YAML tags ({len(tags_cfg)} entries).")

        for entry in tags_cfg:
            try:
                tag_id = int(entry["id"])
                family = str(entry.get("family", "tag36h11"))
                size = float(entry["size"])
            except (KeyError, TypeError, ValueError) as e:
                self.get_logger().warn(f"Skipping malformed tag entry {entry}: {e}")
                continue
            self._tags_config.append({"id": tag_id, "family": family, "size": size})
        if not self._tags_config:
            raise RuntimeError("No valid tag entries (tags_override or YAML tags).")

    def _group_tags_by_family(self) -> Tuple[Dict[str, Dict[int, float]], Dict[Tuple[str, int], float], List[str]]:
        """Bucket ``_tags_config`` by family and produce the per-tag size table.

        Returns
        -------
        per_family
            ``{family: {tag_id: size}}`` -- the per-family id->size map.
            Currently informational only (the upstream wrapper doesn't
            need this), but useful for logging.
        id_to_size
            ``{(family, tag_id): size}`` -- the lookup the detector
            uses after decode to pick the right size for ``estimate_tag_pose``.
        families
            Sorted list of unique family names -- the order the
            detector instantiates its ``apriltag`` wrappers.
        """
        per_family: Dict[str, Dict[int, float]] = {}
        for tag in self._tags_config:
            per_family.setdefault(tag["family"], {})[int(tag["id"])] = float(tag["size"])
        id_to_size = {
            (fam, tag_id): size
            for fam, d in per_family.items()
            for tag_id, size in d.items()
        }
        return per_family, id_to_size, sorted(per_family)

    # =====================================================================
    # Intrinsics -> rectifier + detector pool
    # =====================================================================
    def _yaml_intrinsics(self) -> Optional[Tuple[float, float, float, float, int, int, list, bool]]:
        """Read the ``intrinsics.*`` ROS params into an 8-tuple, or ``None``.

        Returns ``(fx, fy, cx, cy, width, height, distortion_list, fisheye)``
        when the YAML has all the required fields; ``None`` when any
        focal length or dimension is missing/zero.
        """
        fx = _as_float(self.get_parameter("intrinsics.fx").value)
        fy = _as_float(self.get_parameter("intrinsics.fy").value)
        cx = _as_float(self.get_parameter("intrinsics.cx").value)
        cy = _as_float(self.get_parameter("intrinsics.cy").value)
        width = _as_int(self.get_parameter("intrinsics.width").value)
        height = _as_int(self.get_parameter("intrinsics.height").value)
        distortion = self.get_parameter("intrinsics.distortion").value or [0.0] * 5
        fisheye = bool(self.get_parameter("intrinsics.fisheye").value)
        if fx <= 0.0 or fy <= 0.0 or width <= 0 or height <= 0:
            return None
        if not isinstance(distortion, (list, tuple)):
            return None
        d = [float(x) for x in distortion][:5]
        while len(d) < 5:
            d.append(0.0)
        return fx, fy, cx, cy, width, height, d, fisheye

    def _try_build_from_yaml_intrinsics(self) -> None:
        """Build the pipeline from YAML intrinsics when no CameraInfo is on the wire.

        Called from ``__init__`` when ``info_topic`` is empty. Warns
        and bails (the tick callback will no-op until a CameraInfo
        arrives) when the YAML does not have usable intrinsics yet.
        """
        cfg = self._yaml_intrinsics()
        if cfg is None:
            self.get_logger().warn(
                "No info_topic and no usable YAML intrinsics; detector will "
                "wait for a CameraInfo message to build its pipeline."
            )
            return
        fx, fy, cx, cy, width, height, d, fisheye = cfg
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array(d[:5], dtype=np.float32)
        self.get_logger().info(
            f"Building detectors from YAML intrinsics: {width}x{height} "
            f"fx={fx} fy={fy} cx={cx} cy={cy} fisheye={fisheye}"
        )
        self._build_pipeline(K, D, width, height, is_fisheye=fisheye)
        self._ready = True

    def _info_cb(self, msg: CameraInfo) -> None:
        """First-CameraInfo callback: build the pipeline then ignore later ones.

        After the pipeline is built, every subsequent ``CameraInfo`` is
        dropped -- the camera intrinsics are assumed stable for the
        life of the node. (If you ever need to re-tune intrinsics at
        runtime, this is the place to do it.)
        """
        if self._ready:
            return
        self._info_msg = msg
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        width = int(msg.width)
        height = int(msg.height)
        d = list(msg.d) if msg.d else [0.0] * 5
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        D = np.array(d[:5], dtype=np.float32)
        if any(abs(x) > 1e-9 for x in d):
            self.get_logger().warn(
                f"Non-zero distortion coefficients {d}; using plumb-bob model."
            )
        self._build_pipeline(K, D, width, height, is_fisheye=False)
        self._ready = True
        self.get_logger().info("CameraInfo received; detector pipeline ready.")

    def _build_pipeline(
        self,
        K: np.ndarray,
        D: np.ndarray,
        width: int,
        height: int,
        is_fisheye: bool,
    ) -> None:
        """Build the rectifier, JPEG encoder, and single detector.

        Called exactly once per node, either from
        :meth:`_try_build_from_yaml_intrinsics` (no ``info_topic``) or
        from :meth:`_info_cb` (first ``CameraInfo`` received).

        Steps:
        1. Clamp ``process_scale`` to ``(0, 1]``.
        2. Build the CPU ``ImageRectifier`` and read its rectified
           intrinsics + size.
        3. If ``process_scale < 1``, compute the intrinsics + size of
           the downscaled image so ``apriltag3`` can pose-solve in the
           smaller coordinate system.
        4. Cache the ``jpeg_quality`` parameter (the encode itself is
           done inline by :meth:`_publish_annotated` with
           :func:`cv2.imencode`).
        5. Build the selected detector backend (python or cuda); per-tag
           size is looked up after decode.
        6. Print the startup banner (per-stage backend / scale / quality).
        """
        # --- 0. Resolve backends ---------------------------------------------
        backend = str(
            self.get_parameter("detector_backend").value or "python"
        ).lower()
        if backend not in ("python", "cuda"):
            self.get_logger().warn(
                f"Unknown detector_backend={backend!r}; falling back to 'python'."
            )
            backend = "python"
        self._detector_backend = backend

        image_pipeline = str(
            self.get_parameter("image_pipeline").value or "cpu"
        ).lower()
        if image_pipeline not in ("cpu", "cuda"):
            self.get_logger().warn(
                f"Unknown image_pipeline={image_pipeline!r}; using 'cpu'."
            )
            image_pipeline = "cpu"
        if image_pipeline == "cuda":
            if self._image_transport != "compressed":
                self.get_logger().warn(
                    "image_pipeline='cuda' requires image_transport='compressed'; "
                    "using the CPU image pipeline."
                )
                image_pipeline = "cpu"
            elif backend != "cuda":
                self.get_logger().warn(
                    "image_pipeline='cuda' requires detector_backend='cuda'; "
                    "using the CPU image pipeline."
                )
                image_pipeline = "cpu"
        self._image_pipeline = image_pipeline

        if self._image_pipeline == "cuda":
            try:
                self._build_gpu_pipeline(K, D, width, height, is_fisheye)
                return
            except Exception as e:
                self.get_logger().warn(
                    f"CUDA image pipeline unavailable ({e}); falling back to "
                    "the CPU image pipeline + python detector."
                )
                self._image_pipeline = "cpu"
                self._detector_backend = "python"
                backend = "python"

        # --- 1. process_scale ------------------------------------------------
        scale = float(self.get_parameter("process_scale").value or 1.0)
        if scale <= 0.0 or scale > 1.0:
            self.get_logger().warn(
                f"process_scale={scale} out of range (0, 1]; clamping to 1.0."
            )
            scale = 1.0
        self._process_scale = scale

        # --- 2. CPU rectifier -------------------------------------------------
        self._rectifier, self._rectifier_backend = build_rectifier(
            logger=self.get_logger(),
            camera_matrix=K,
            dist_coeffs=D,
            image_size=(width, height),
            is_fisheye=is_fisheye,
            crop_to_valid_pixels=True,
        )
        new_K = self._rectifier.get_intrinsics()
        new_size = {
            "img_width": new_K["img_width"],
            "img_height": new_K["img_height"],
        }
        new_D = np.zeros(5, dtype=np.float32)

        # --- 3. Downscaled intrinsics --------------------------------------
        # apriltag3 pose-solves in the coordinate system of the image
        # we feed it, so when we feed it the downscaled gray frame we
        # must also pass it downscaled intrinsics.
        if scale < 1.0:
            small_K = dict(new_K)
            small_K["fx"] = float(new_K["fx"]) * scale
            small_K["fy"] = float(new_K["fy"]) * scale
            small_K["cx"] = float(new_K["cx"]) * scale
            small_K["cy"] = float(new_K["cy"]) * scale
            small_size = {
                "img_width": max(1, int(new_size["img_width"] * scale)),
                "img_height": max(1, int(new_size["img_height"] * scale)),
            }
        else:
            small_K = new_K
            small_size = new_size

        # --- 4. Cache the JPEG quality --------------------------------------
        # The encode itself is done inline in ``_publish_annotated`` via
        # ``cv2.imencode``; we only need the quality value here so the
        # encode path does not have to read a ROS parameter on every tick.
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value or 80)
        self._min_edge_dist = _as_int(self.get_parameter("min_edge_dist").value, 10)

        # --- 5. Build the selected detector backend -------------------------
        # Both backends expose the same detect()/annotate() surface and
        # look up per-tag size from ``id_to_size`` after decode. The
        # backend name was resolved in step 0.
        per_family, id_to_size, families = self._group_tags_by_family()
        child_logger = self.get_logger().get_child("det_all")

        if backend == "cuda":
            # Lazy import: the CUDA package is an optional dependency for
            # hosts that only run the python backend.
            try:
                from .apriltag_cuda import CuAprilTagDetector

                self._detector = CuAprilTagDetector(
                    id_to_size=id_to_size,
                    family="tag36h11",
                    camera_intrinsics=small_K,
                    camera_distortion=new_D.tolist(),
                    image_size=small_size,
                    logger=child_logger,
                    nominal_size=float(
                        self.get_parameter("cuda_nominal_size").value or 0.125
                    ),
                    tile_size=_as_int(self.get_parameter("cuda_tile_size").value, 4),
                    max_tags=_as_int(self.get_parameter("cuda_max_tags").value, 64),
                )
            except Exception as e:
                # The native shim is optional: it is not built on hosts
                # without CUDA and is skipped when the third-party library
                # is missing. Degrade to the CPU python detector instead
                # of failing to start.
                self.get_logger().warn(
                    f"CUDA detector backend unavailable ({e}); falling back "
                    "to the python (CPU) detector backend."
                )
                backend = "python"
                self._detector_backend = "python"

        if backend == "python":
            self._detector = AprilTagDetector(
                families=families,
                id_to_size=id_to_size,
                camera_intrinsics=small_K,
                camera_distortion=new_D.tolist(),
                image_size=small_size,
                logger=child_logger,
                detector_params=dict(self._detector_params_template),
            )

        # --- 6. Startup banner ----------------------------------------------
        # Single-line, easy-to-grep summary of what was actually built.
        self.get_logger().info(
            "=== Pipeline ===\n"
            f"  detector backend: {backend}\n"
            f"  rectify backend : {self._rectifier_backend}\n"
            f"  jpeg   backend  : cpu (cv2.imencode)\n"
            f"  process_scale   : {self._process_scale}\n"
            f"  jpeg_quality    : {self._jpeg_quality}\n"
            f"  min_edge_dist   : {self._min_edge_dist}px\n"
            f"  families        : {families} ({sum(len(v) for v in per_family.values())} tags)"
        )

    def _build_gpu_pipeline(
        self,
        K: np.ndarray,
        D: np.ndarray,
        width: int,
        height: int,
        is_fisheye: bool,
    ) -> None:
        """Build the nvjpeg/VPI image pipeline and the CUDA detector.

        Called from :meth:`_build_pipeline` when ``image_pipeline`` is
        ``"cuda"``. The rectifier geometry is taken from the same CPU
        ``ImageRectifier`` the CPU path uses, so display output is
        identical; the detection-resolution rectification is a scaled
        variant of it.
        """
        from .apriltag_cuda import CuAprilTagDetector, GpuImagePipeline

        scale = float(self.get_parameter("process_scale").value or 1.0)
        if scale <= 0.0 or scale > 1.0:
            self.get_logger().warn(
                f"process_scale={scale} out of range (0, 1]; clamping to 1.0."
            )
            scale = 1.0
        self._process_scale = scale
        self._jpeg_quality = int(self.get_parameter("jpeg_quality").value or 80)
        self._min_edge_dist = _as_int(self.get_parameter("min_edge_dist").value, 10)

        self._gpu_pipeline = GpuImagePipeline(
            camera_intrinsics={
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            },
            camera_distortion=D.tolist(),
            image_size={"img_width": width, "img_height": height},
            logger=self.get_logger().get_child("gpu_image"),
            is_fisheye=is_fisheye,
            process_scale=scale,
            rectify_interval=_as_int(
                self.get_parameter("gpu_rectify_interval").value, 4
            ),
        )

        per_family, id_to_size, families = self._group_tags_by_family()
        child_logger = self.get_logger().get_child("det_all")
        detect_K = self._gpu_pipeline.detect_intrinsics
        self._detector = CuAprilTagDetector(
            id_to_size=id_to_size,
            family="tag36h11",
            camera_intrinsics={
                "fx": float(detect_K["fx"]),
                "fy": float(detect_K["fy"]),
                "cx": float(detect_K["cx"]),
                "cy": float(detect_K["cy"]),
            },
            camera_distortion=[0.0] * 5,
            image_size={
                "img_width": int(detect_K["img_width"]),
                "img_height": int(detect_K["img_height"]),
            },
            logger=child_logger,
            nominal_size=float(
                self.get_parameter("cuda_nominal_size").value or 0.125
            ),
            tile_size=_as_int(self.get_parameter("cuda_tile_size").value, 4),
            max_tags=_as_int(self.get_parameter("cuda_max_tags").value, 64),
        )

        self.get_logger().info(
            "=== Pipeline ===\n"
            f"  detector backend: {self._detector_backend}\n"
            "  image  backend  : cuda (nvjpeg decode, VPI rectify, nvjpeg encode)\n"
            f"  display         : {self._gpu_pipeline.display_width}x"
            f"{self._gpu_pipeline.display_height}\n"
            f"  detect          : {self._gpu_pipeline.detect_width}x"
            f"{self._gpu_pipeline.detect_height} (scale {self._process_scale})\n"
            f"  jpeg_quality    : {self._jpeg_quality}\n"
            f"  min_edge_dist   : {self._min_edge_dist}px\n"
            f"  families        : {families} ({sum(len(v) for v in per_family.values())} tags)"
        )

    # =====================================================================
    # Image callbacks
    # =====================================================================
    def _image_cb_raw(self, msg: Image) -> None:
        """``raw`` transport callback: ``sensor_msgs/Image`` -> BGR numpy."""
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge failed: {e}")
            return
        with self._lock:
            self._latest_bgr = bgr
            self._latest_stamp = msg.header.stamp

    def _image_cb_compressed(self, msg: CompressedImage) -> None:
        """``compressed`` transport callback.

        On the GPU image path the JPEG bytes are stored as-is and decoded
        once per tick (avoids decoding camera frames the tick will skip);
        otherwise they are decoded immediately with ``cv_bridge``.
        """
        if self._image_pipeline == "cuda":
            with self._lock:
                self._latest_jpeg = bytes(msg.data)
                self._latest_stamp = msg.header.stamp
            return
        try:
            bgr = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge compressed failed: {e}")
            return
        with self._lock:
            self._latest_bgr = bgr
            self._latest_stamp = msg.header.stamp

    # =====================================================================
    # Tick: process_frame -> build_detection_message -> publish_annotated
    # =====================================================================
    def _tick(self) -> None:
        """Drain the latest received frame and run the full pipeline.

        Runs at ``publish_rate`` Hz. Skips silently when:

        * the pipeline is not built yet (``_ready`` is False -- waiting
          on the first ``CameraInfo``);
        * no frame has been received yet (``_latest_bgr`` is None).
        """
        if not self._ready:
            return
        if self._image_pipeline == "cuda":
            self._tick_gpu()
            return
        with self._lock:
            bgr = self._latest_bgr
            stamp = self._latest_stamp
        if bgr is None:
            return

        work, detected_tags = self._process_frame(bgr)
        stamp = stamp if stamp is not None else self.get_clock().now().to_msg()

        _draw_crosshair(work)
        self._publish_annotated(work, stamp)
        self._publish_detections(detected_tags, stamp)

    def _tick_gpu(self) -> None:
        """GPU image path: nvjpeg decode -> VPI rectify -> cuAprilTags -> nvjpeg.

        The decoded/rectified detection-resolution frame never leaves the
        device. Only the full-resolution rectified frame is copied back to
        the host, for the CPU annotation overlay and crosshair.
        """
        with self._lock:
            jpeg = self._latest_jpeg
            stamp = self._latest_stamp
        if jpeg is None:
            return
        pipeline = self._gpu_pipeline
        detector = self._detector
        if pipeline is None or detector is None:
            return
        stamp = stamp if stamp is not None else self.get_clock().now().to_msg()

        try:
            display = pipeline.decode_and_rectify(jpeg)
        except Exception as e:
            self.get_logger().warn(
                f"GPU image pipeline failed: {e}; skipping this tick."
            )
            return

        detections: List[Dict] = []
        try:
            detections = detector.detect_device(
                pipeline.detect_ptr,
                pipeline.detect_width,
                pipeline.detect_height,
                pipeline.detect_width * 3,
            )
        except Exception as e:
            self.get_logger().warn(f"Detector failed: {e}; skipping this tick.")

        # Corner coordinates are in the detection-resolution frame; scale
        # them up so the overlays land on the full-res display image.
        if detections and pipeline.detect_width != pipeline.display_width:
            inv = pipeline.display_width / float(pipeline.detect_width)
            detections = [
                {**d, "corners": d["corners"] * inv} for d in detections
            ]

        # Same edge-clip filter and pose sanitization as the CPU path.
        if detections:
            h, w = display.shape[:2]
            detections = [
                d for d in detections
                if self._within_frame(d["corners"], w, h)
            ]
        detected = self._sanitize_detections(detections)

        try:
            detector.annotate(display, detections)
        except Exception as e:
            self.get_logger().warn(f"Detector annotate failed: {e}")

        _draw_crosshair(display)
        try:
            jpeg_out = pipeline.encode(display, self._jpeg_quality)
        except Exception as e:
            self.get_logger().error(f"GPU JPEG encode failed: {e}")
            jpeg_out = None
        if jpeg_out is not None:
            self._publish_annotated_bytes(jpeg_out, stamp)
        self._publish_detections(detected, stamp)

    def _within_frame(self, corners: np.ndarray, width: int, height: int) -> bool:
        """``True`` when every corner is >= ``min_edge_dist`` px from the border.

        ``corners`` must already be in the full-resolution rectified
        image's coordinate system (i.e. after the ``process_scale``
        up-scaling in :meth:`_process_frame`).
        """
        d = self._min_edge_dist
        if d <= 0:
            return True
        x_min, y_min = corners[:, 0].min(), corners[:, 1].min()
        x_max, y_max = corners[:, 0].max(), corners[:, 1].max()
        return (
            x_min >= d and y_min >= d
            and x_max <= (width - 1 - d) and y_max <= (height - 1 - d)
        )

    def _process_frame(
        self, bgr: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[str, int, np.ndarray, bool]]]:
        """Rectify + downscale + detect + annotate a single frame.

        Parameters
        ----------
        bgr
            Latest BGR frame as decoded by ``cv_bridge``.

        Returns
        -------
        (work, detections)
            ``work`` is the full-resolution rectified BGR ready for
            the crosshair + JPEG encode pass. ``detections`` is the
            list of ``(family, tag_id, T_cam_to_tag, was_bad)`` that
            ``_publish_detections`` turns into a
            ``vision_msgs/Detection3DArray``.
        """
        # 1. Rectify on the CPU (always full resolution).
        work = self._rectifier.rectify(bgr)

        # 2. Optionally downscale for faster detection.
        if self._process_scale < 1.0:
            small = cv2.resize(
                work, None,
                fx=self._process_scale, fy=self._process_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = work

        # 3. Run the selected detector. cuAprilTags consumes the BGR
        #    frame directly (uchar3); apriltag3 needs grayscale.
        detected: List[Tuple[str, int, np.ndarray, bool]] = []
        detector = self._detector
        if detector is None:
            return work, detected
        try:
            if self._detector_backend == "cuda":
                detections = detector.detect(small)
            else:
                detections = detector.detect(
                    cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                )
        except Exception as e:
            self.get_logger().warn(f"Detector failed: {e}; skipping this tick.")
            return work, detected

        # 4. Scale corner coordinates back up to full resolution so
        #    the annotation overlays line up on ``work``.
        if self._process_scale < 1.0 and detections:
            inv = 1.0 / self._process_scale
            detections = [
                {**d, "corners": d["corners"] * inv} for d in detections
            ]

        # 4b. Drop detections whose corners fall within
        #     ``min_edge_dist`` pixels of the rectified image border --
        #     a tag partially clipped by the frame edge yields
        #     unreliable corner geometry and pose.
        if detections:
            h, w = work.shape[:2]
            detections = [
                d for d in detections
                if self._within_frame(d["corners"], w, h)
            ]

        # 5. Sanitize the pose; drop detections whose rotations
        #    are unrecoverable (very noisy / degenerate).
        detected = self._sanitize_detections(detections)

        # 6. Draw the bounding box + label block on the full-res
        #    ``work`` image. Annotation errors must not stop the
        #    pipeline -- log and continue.
        try:
            detector.annotate(work, detections)
        except Exception as e:
            self.get_logger().warn(f"Detector annotate failed: {e}")

        return work, detected

    def _sanitize_detections(
        self, detections: List[Dict],
    ) -> List[Tuple[str, int, np.ndarray, bool]]:
        """Project rotations onto SO(3); drop unrecoverable detections.

        Shared by the CPU and GPU image paths. Returns the
        ``(family, tag_id, T, was_bad)`` tuples ``_publish_detections``
        consumes and mutates ``det["T"]`` in place so annotation draws
        the sanitized pose.
        """
        detected: List[Tuple[str, int, np.ndarray, bool]] = []
        for det in detections:
            family = det["family"]
            tag_id = det["tag_id"]
            T = det["T"]
            try:
                was_bad = is_bad_rotation(T[:3, :3])
                T[:3, :3] = sanitize_rotation(T[:3, :3])
                if (
                    not np.all(np.isfinite(T[:3, :3]))
                    or np.linalg.det(T[:3, :3]) < 0.5
                ):
                    raise ValueError("rotation unrecoverable after SVD")
                detected.append((family, tag_id, T, was_bad))
            except Exception as e:
                self._bad_pose_count += 1
                if self._bad_pose_count in (1, 10, 100, 1000) or (
                    self._bad_pose_count % 1000 == 0
                ):
                    self.get_logger().warn(
                        f"Skipped tag {family}:{tag_id} with bad pose "
                        f"(total bad-rotation drops: {self._bad_pose_count}): {e}"
                    )
        return detected

    # =====================================================================
    # Publications
    # =====================================================================
    def _publish_annotated(self, work: np.ndarray, stamp) -> None:
        """Encode ``work`` to JPEG on the CPU and publish.

        Only used by the CPU image path; the GPU path encodes with
        nvjpeg in :meth:`_tick_gpu` and calls
        :meth:`_publish_annotated_bytes`.

        The frame id of the published message comes from
        :meth:`_camera_frame_id` (ROS param override -> ``CameraInfo``
        header -> hard-coded fallback).
        """
        try:
            ok, buf = cv2.imencode(
                ".jpg", work,
                [_IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if not ok:
                raise RuntimeError("cv2.imencode returned False")
            jpeg_bytes = bytes(buf.tobytes() if hasattr(buf, "tobytes") else buf)
        except Exception as e:
            self.get_logger().error(f"JPEG encode failed: {e}")
            return

        self._publish_annotated_bytes(jpeg_bytes, stamp)

    def _publish_annotated_bytes(self, jpeg_bytes: bytes, stamp) -> None:
        """Publish already-encoded JPEG bytes as ``CompressedImage``."""
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = jpeg_bytes
        msg.header.stamp = stamp
        msg.header.frame_id = self._camera_frame_id()
        self.image_pub.publish(msg)

    def _publish_detections(
        self,
        detected: List[Tuple[str, int, np.ndarray, bool]],
        stamp,
    ) -> None:
        """Publish ``detected`` as ``vision_msgs/Detection3DArray``.

        Each entry's ``id`` is ``f"{family}:{tag_id}"`` so downstream
        consumers (the fuser, Foxglove panels) can disambiguate
        tags that share an id across families.
        """
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self._camera_frame_id()

        for family, tag_id, T_cam_to_tag, _was_bad in detected:
            det3d = Detection3D()
            det3d.header = msg.header
            det3d.id = f"{family}:{tag_id}"
            hyp = ObjectHypothesis(class_id=str(tag_id), score=1.0)
            pwp = ObjectHypothesisWithPose(hypothesis=hyp)
            pwp.pose = PoseWithCovariance()
            pwp.pose.pose = matrix_to_pose_msg(T_cam_to_tag)
            det3d.results = [pwp]
            det3d.bbox = BoundingBox3D()
            det3d.bbox.center = Pose()
            det3d.bbox.size = Vector3()
            msg.detections.append(det3d)

        self.detections_pub.publish(msg)

    # =====================================================================
    # Frame id
    # =====================================================================
    def _camera_frame_id(self) -> str:
        """Pick the TF frame id for published messages.

        Order of precedence:

        1. ``camera_frame`` ROS param (set by the launch file).
        2. ``CameraInfo.header.frame_id`` (when ``info_topic`` is set).
        3. Hard-coded fallback ``"camera_optical_frame"``.
        """
        override = str(self.get_parameter("camera_frame").value or "")
        if override:
            return override
        if self._info_msg is not None and self._info_msg.header.frame_id:
            return self._info_msg.header.frame_id
        return "camera_optical_frame"


def main() -> None:
    """Entry point for the ``apriltag_detector_node`` console script."""
    rclpy.init()
    node = AprilTagDetectorNode()
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