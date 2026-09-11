"""In-process CUDA AprilTag detector backed by NVIDIA's cuAprilTags.

This module is the ``cuda`` backend for ``apriltag_detector_node``. It
mirrors the public surface of
:class:`race_auv_camera_pkg.apriltag_processor.AprilTagDetector` so the
node's ``detect`` / ``annotate`` call sites are backend-agnostic:

* ``detect(bgr)`` returns the same list-of-dicts structure.
* ``annotate(image, detections)`` delegates to the shared
  :func:`race_auv_camera_pkg.apriltag_processor.annotate_detections`.
* ``camera_matrix`` / ``dist_coeffs`` are exposed for annotation.

The heavy lifting happens through a C shim
(``race_auv_apriltag_cuda/librace_auv_apriltag_cuda.so``) that flattens
NVIDIA's ``cuAprilTagsID_t`` struct into primitive arrays so Python
never has to mirror its padding.

Only the ``tag36h11`` family is supported by cuAprilTags. Entries from
other families in ``id_to_size`` are ignored; the detector logs a
warning when the configured tag list contains no ``tag36h11`` tags.

Because the pose is solved inside the library from a fixed ``tag_dim``,
one detector is created per image size at ``nominal_size`` and each
detection's translation is rescaled to the tag's true size:

    t_true = t_reported * (size_true / nominal_size)

The rotation is unaffected (it does not depend on scale). Per the
``cuAprilTags.h`` contract, ``translation`` is expressed in the same
units as ``tag_dim``.

Coordinate convention
---------------------
Poses are ``T_camera_to_tag`` in the *rectified* camera frame, matching
the python backend.
"""

from __future__ import annotations

import ctypes
import os
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .apriltag_geom import is_bad_rotation
from .apriltag_processor import annotate_detections
from .image_processing import build_rectifier


#: The only tag family cuAprilTags can decode.
FAMILY = "tag36h11"

#: Default detector creation size (metres). Matches the 12.5 cm tags on
#: the station; smaller/larger tags are rescaled per detection.
DEFAULT_NOMINAL_SIZE = 0.125

#: Adaptive-threshold window size. Isaac ROS' own node uses 4.
DEFAULT_TILE_SIZE = 4

#: Maximum number of tags returned per frame (Isaac ROS' node default).
DEFAULT_MAX_TAGS = 64

_LIBRARY_NAME = "librace_auv_apriltag_cuda.so"
_LIBRARY_ENV = "RACE_AUV_APRILTAG_CUDA_LIB"
_IMAGE_LIBRARY_NAME = "librace_auv_image_cuda.so"
_IMAGE_LIBRARY_ENV = "RACE_AUV_IMAGE_CUDA_LIB"


def _library_path(
    explicit: Optional[str],
    library_name: str = _LIBRARY_NAME,
    env_var: str = _LIBRARY_ENV,
    feature: str = "race_auv_apriltag_cuda",
) -> str:
    """Resolve a shim ``.so`` path.

    Order: explicit argument -> env var -> the installed package share
    directory.
    """
    if explicit:
        return explicit
    env_path = os.environ.get(env_var, "").strip()
    if env_path:
        return env_path
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(
            get_package_share_directory("race_auv_apriltag_cuda"),
            "lib",
            library_name,
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not locate {feature}'s {library_name}. "
            "Source the workspace (install/setup.bash) or set "
            f"{env_var} to the full path of the shared library."
        ) from e


def _bind_signatures(lib: ctypes.CDLL) -> None:
    """Declare every shim prototype exactly once."""
    lib.race_at_create.restype = ctypes.c_void_p
    lib.race_at_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.race_at_detect.restype = ctypes.c_int
    lib.race_at_detect.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.race_at_detect_device.restype = ctypes.c_int
    lib.race_at_detect_device.argtypes = list(lib.race_at_detect.argtypes)
    lib.race_at_destroy.restype = None
    lib.race_at_destroy.argtypes = [ctypes.c_void_p]
    lib.race_at_last_error.restype = ctypes.c_char_p
    lib.race_at_last_error.argtypes = []


def _bind_image_signatures(lib: ctypes.CDLL) -> None:
    """Declare the GPU image shim prototypes."""
    lib.race_img_create.restype = ctypes.c_void_p
    lib.race_img_create.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.race_img_destroy.restype = None
    lib.race_img_destroy.argtypes = [ctypes.c_void_p]
    lib.race_img_set_fisheye_job.restype = ctypes.c_int
    lib.race_img_set_fisheye_job.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.race_img_decode.restype = ctypes.c_int
    lib.race_img_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ]
    lib.race_img_rectify.restype = ctypes.c_int
    lib.race_img_rectify.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.race_img_get_decoded.restype = ctypes.c_void_p
    lib.race_img_get_decoded.argtypes = [ctypes.c_void_p]
    lib.race_img_get_rectified.restype = ctypes.c_void_p
    lib.race_img_get_rectified.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.race_img_get_rectified_size.restype = ctypes.c_int
    lib.race_img_get_rectified_size.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.race_img_rectified_to_host.restype = ctypes.c_int
    lib.race_img_rectified_to_host.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ]
    lib.race_img_encode.restype = ctypes.c_int
    lib.race_img_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.race_img_last_error.restype = ctypes.c_char_p
    lib.race_img_last_error.argtypes = []


class CuAprilTagDetector:
    """Detect ``tag36h11`` tags on an undistorted BGR image with CUDA.

    Parameters
    ----------
    id_to_size
        Mapping ``(family, tag_id) -> size_m``. Only ``(tag36h11, ...)``
        entries are usable; the rest are ignored.
    camera_intrinsics
        Dict with ``fx``, ``fy``, ``cx``, ``cy`` (pixels) of the image
        that will be passed to :meth:`detect` (i.e. already multiplied
        by ``process_scale`` if the parent node downscales).
    image_size
        Dict with ``img_width`` and ``img_height`` of that same image.
    logger
        A ``rclpy``-compatible logger.
    family
        Tag family; must be ``"tag36h11"``.
    camera_distortion
        Optional distortion coefficients. Pose estimation ignores them;
        they are only used to project the axis overlay.
    nominal_size
        Physical side length the detector is created with. All tag
        translations are rescaled from this size to each tag's true
        size. Defaults to 0.125 m.
    tile_size
        Adaptive-threshold window size (defaults to 4).
    max_tags
        Maximum detections returned per frame (defaults to 64).
    library_path
        Optional explicit path to ``librace_auv_apriltag_cuda.so``.

    Raises
    ------
    RuntimeError
        If the shim library or the CUDA detector cannot be created.
    """

    def __init__(
        self,
        id_to_size: Dict[Tuple[str, int], float],
        camera_intrinsics: dict,
        image_size: dict,
        logger,
        family: str = FAMILY,
        camera_distortion: Optional[Sequence[float]] = None,
        nominal_size: float = DEFAULT_NOMINAL_SIZE,
        tile_size: int = DEFAULT_TILE_SIZE,
        max_tags: int = DEFAULT_MAX_TAGS,
        library_path: Optional[str] = None,
    ):
        self.logger = logger

        family = str(family)
        if family != FAMILY:
            raise ValueError(
                f"cuAprilTags supports only '{FAMILY}' (got '{family}')."
            )
        self._family = family

        self._id_to_size: Dict[int, float] = {
            int(i): float(s)
            for (f, i), s in id_to_size.items()
            if str(f) == family
        }
        if not self._id_to_size:
            self.logger.warn(
                f"CUDA backend configured with no '{family}' tags; it will "
                f"detect nothing. (cuAprilTags cannot decode other families.)"
            )

        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])
        self.img_width = int(image_size["img_width"])
        self.img_height = int(image_size["img_height"])
        self._nominal_size = float(nominal_size)
        if self._nominal_size <= 0.0:
            raise ValueError(
                f"cuda_nominal_size must be positive (got {nominal_size})."
            )
        self._tile_size = int(tile_size)
        self._max_tags = int(max_tags)
        if self._max_tags <= 0:
            raise ValueError(f"cuda_max_tags must be positive (got {max_tags}).")

        # Same attribute names / normalization as AprilTagDetector so the
        # shared annotate_detections() call site is unchanged.
        self.camera_matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        if camera_distortion is None:
            self.dist_coeffs = np.zeros(5, dtype=np.float32)
        else:
            dist = np.array(camera_distortion, dtype=np.float32)
            if dist.shape[0] < 4:
                padded = np.zeros(5, dtype=np.float32)
                padded[: dist.shape[0]] = dist
                self.dist_coeffs = padded
            else:
                self.dist_coeffs = dist[:5]

        lib_path = _library_path(library_path)
        self._lib = ctypes.CDLL(lib_path)
        _bind_signatures(self._lib)

        self._handle = self._lib.race_at_create(
            self.img_width, self.img_height, self._tile_size,
            self._nominal_size, fx, fy, cx, cy,
        )
        if not self._handle:
            raise RuntimeError(
                "cuAprilTags detector creation failed: "
                f"{self._shim_error()}"
            )

        # Output buffers, allocated once and reused every frame.
        self._ids = np.zeros(self._max_tags, dtype=np.uint16)
        self._corners = np.zeros((self._max_tags, 4, 2), dtype=np.float32)
        self._orientation = np.zeros((self._max_tags, 9), dtype=np.float32)
        self._translation = np.zeros((self._max_tags, 3), dtype=np.float32)
        self._count = ctypes.c_int(0)

        self.logger.info(
            f"cuAprilTags ready: family='{family}' "
            f"image={self.img_width}x{self.img_height} "
            f"tile_size={self._tile_size} tag_dim={self._nominal_size:g}m "
            f"max_tags={self._max_tags} "
            f"tags={len(self._id_to_size)} library={lib_path}"
        )

    # =====================================================================
    # detect
    # =====================================================================
    def detect(self, bgr: np.ndarray) -> list[dict]:
        """Run detection on an undistorted BGR image.

        Returns the same list-of-dicts structure as the python backend
        (``family``, ``tag_id``, ``size``, ``T``, ``corners``,
        ``bad_pose``).
        """
        if bgr is None:
            self.logger.warn("Received a null image for AprilTag detection.")
            return []
        if self._handle is None:
            return []

        bgr = np.ascontiguousarray(bgr)
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            self.logger.warn(
                f"CUDA backend expects an HxWx3 uint8 BGR image, got "
                f"shape={bgr.shape} dtype={bgr.dtype}; skipping this tick."
            )
            return []

        height, width = bgr.shape[:2]
        if width != self.img_width or height != self.img_height:
            self.logger.warn(
                f"CUDA detector created for {self.img_width}x{self.img_height} "
                f"but received {width}x{height}; skipping this tick."
            )
            return []

        self._count.value = 0
        status = self._lib.race_at_detect(
            self._handle,
            bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bgr.strides[0],
            width, height, self._max_tags,
            self._ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            self._corners.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._orientation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._translation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(self._count),
        )
        if status != 0:
            self.logger.warn(
                f"cuAprilTags detect failed: {self._shim_error()}; "
                f"skipping this tick."
            )
            return []
        return self._collect_detections()

    def detect_device(
        self, device_ptr: int, width: int, height: int, pitch: int,
    ) -> list[dict]:
        """Run detection on an image that already lives in device memory.

        Used by the GPU image pipeline so no host->device copy is made.
        ``device_ptr`` is the integer address of a BGR device buffer
        (e.g. :attr:`GpuImagePipeline.detect_ptr`), ``pitch`` its row
        stride in bytes.
        """
        if self._handle is None or not device_ptr:
            return []
        if width != self.img_width or height != self.img_height:
            self.logger.warn(
                f"CUDA detector created for {self.img_width}x{self.img_height} "
                f"but got device image {width}x{height}; skipping this tick."
            )
            return []
        if pitch < width * 3:
            self.logger.warn("detect_device: pitch smaller than width*3")
            return []

        self._count.value = 0
        status = self._lib.race_at_detect_device(
            self._handle,
            ctypes.cast(
                ctypes.c_void_p(device_ptr), ctypes.POINTER(ctypes.c_uint8)
            ),
            pitch,
            width, height, self._max_tags,
            self._ids.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            self._corners.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._orientation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._translation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(self._count),
        )
        if status != 0:
            self.logger.warn(
                f"cuAprilTags detect_device failed: {self._shim_error()}; "
                f"skipping this tick."
            )
            return []
        return self._collect_detections()

    def _collect_detections(self) -> list[dict]:
        """Build the public detection dicts from the shim's output buffers."""
        out: list[dict] = []
        for i in range(int(self._count.value)):
            tag_id = int(self._ids[i])
            size = self._id_to_size.get(tag_id)
            if size is None:
                # Decoded, but not a tag we are configured to use.
                continue

            # orientation is column-major per cuAprilTags.h.
            R_mat = self._orientation[i].reshape(3, 3, order="F").astype(
                np.float64
            )
            t_vec = self._translation[i].astype(np.float64) * (
                size / self._nominal_size
            )
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = t_vec

            bad_pose = (
                not np.all(np.isfinite(R_mat))
                or not np.all(np.isfinite(t_vec))
                or is_bad_rotation(R_mat)
            )
            out.append({
                "family": self._family,
                "tag_id": tag_id,
                "size": float(size),
                "T": T,
                "corners": self._corners[i].copy(),
                "bad_pose": bad_pose,
            })
        return out

    # =====================================================================
    # annotate
    # =====================================================================
    def annotate(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw the shared overlay via :func:`annotate_detections`."""
        return annotate_detections(
            image, detections, self.camera_matrix, self.dist_coeffs, self.logger,
        )

    # =====================================================================
    # life cycle
    # =====================================================================
    def _shim_error(self) -> str:
        try:
            return self._lib.race_at_last_error().decode("utf-8", "replace")
        except Exception:
            return "<no error string>"

    def close(self) -> None:
        """Destroy the underlying CUDA detector. Safe to call twice."""
        if getattr(self, "_handle", None):
            self._lib.race_at_destroy(self._handle)
            self._handle = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class GpuImagePipeline:
    """GPU image stages for one camera: nvjpeg decode + VPI rectify + nvjpeg encode.

    One instance owns a fixed decoded-image size and up to two rectified
    outputs ("jobs"):

    * job 0 -- display: same geometry as the CPU
      :class:`~race_auv_camera_pkg.image_processing.ImageRectifier`
      (same optimal camera matrix and ROI crop), full resolution.
    * job 1 -- detect: the same geometry scaled by ``process_scale``.
      Only created when ``process_scale < 1``.

    The expensive copies are avoided on the detection path: the rectified
    image stays in device memory and
    :meth:`CuAprilTagDetector.detect_device` consumes it directly.

    Typical tick::

        display_bgr = pipeline.decode_and_rectify(jpeg_bytes)
        detections = detector.detect_device(
            pipeline.detect_ptr, pipeline.detect_width,
            pipeline.detect_height, pipeline.detect_width * 3)
        jpeg_out = pipeline.encode(annotated_bgr, quality=90)

    Parameters
    ----------
    camera_intrinsics
        Dict with ``fx``, ``fy``, ``cx``, ``cy`` of the *distorted* image.
    camera_distortion
        Distortion coefficients (4 for fisheye, 5 for plumb-bob).
    image_size
        Dict with ``img_width`` / ``img_height`` of the decoded image.
    logger
        A ``rclpy``-compatible logger.
    is_fisheye
        ``True`` for the equidistant fisheye model.
    process_scale
        Detection downscale in ``(0, 1]``. ``1.0`` disables the detect job.
    rectify_interval
        VPI warp grid spacing (power of two; 4 is a good default).
    crop_to_valid_pixels
        Match the CPU rectifier's ROI crop. Leave ``True``.
    library_path
        Optional explicit path to ``librace_auv_image_cuda.so``.
    """

    def __init__(
        self,
        camera_intrinsics: dict,
        camera_distortion: Sequence[float],
        image_size: dict,
        logger,
        is_fisheye: bool = False,
        process_scale: float = 1.0,
        rectify_interval: int = 4,
        crop_to_valid_pixels: bool = True,
        library_path: Optional[str] = None,
    ):
        self.logger = logger
        lib_path = _library_path(
            library_path,
            _IMAGE_LIBRARY_NAME,
            _IMAGE_LIBRARY_ENV,
            feature="race_auv_image_cuda",
        )
        self._lib = ctypes.CDLL(lib_path)
        _bind_image_signatures(self._lib)

        self.img_width = int(image_size["img_width"])
        self.img_height = int(image_size["img_height"])
        self._camera_matrix = np.array(
            [
                [float(camera_intrinsics["fx"]), 0, float(camera_intrinsics["cx"])],
                [0, float(camera_intrinsics["fy"]), float(camera_intrinsics["cy"])],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self._dist_coeffs = np.asarray(
            camera_distortion if camera_distortion is not None else [],
            dtype=np.float64,
        ).ravel()
        self._is_fisheye = bool(is_fisheye)
        self._dist4 = self._dist_coeffs[:4].astype(np.float64)

        # Reuse the CPU rectifier so maps, ROI and intrinsics match the
        # CPU path exactly.
        self._rectifier, _ = build_rectifier(
            logger=logger,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            image_size=(self.img_width, self.img_height),
            is_fisheye=self._is_fisheye,
            crop_to_valid_pixels=crop_to_valid_pixels,
        )

        self._handle = self._lib.race_img_create(self.img_width, self.img_height)
        if not self._handle:
            raise RuntimeError(
                f"race_img_create({self.img_width}x{self.img_height}) failed: "
                f"{self._shim_error()}"
            )

        roi_x, roi_y, roi_w, roi_h = (
            int(v) for v in self._rectifier.roi
        )
        self._roi = (roi_x, roi_y, roi_w, roi_h)
        map_x, map_y = self._rectifier.get_maps()
        self._maps = []
        display_x, display_y = self._slice_maps(
            map_x, map_y, roi_x, roi_y, roi_w, roi_h
        )
        self._set_job(0, display_x, display_y, roi_w, roi_h, rectify_interval)
        self._jobs = [0]

        self.display_width = roi_w
        self.display_height = roi_h
        self.display_intrinsics = dict(self._rectifier.get_intrinsics())

        self._detect_job = 0
        self.detect_width = self.display_width
        self.detect_height = self.display_height
        self.detect_intrinsics = dict(self.display_intrinsics)

        scale = float(process_scale)
        if scale <= 0.0:
            scale = 1.0
        if scale < 1.0:
            detect_x, detect_y, det_w, det_h, det_K = self._scaled_maps(scale)
            self._set_job(1, detect_x, detect_y, det_w, det_h, rectify_interval)
            self._jobs.append(1)
            self._detect_job = 1
            self.detect_width = det_w
            self.detect_height = det_h
            self.detect_intrinsics = det_K

        self.logger.info(
            f"GPU image pipeline ready: decode {self.img_width}x{self.img_height} "
            f"-> display {self.display_width}x{self.display_height}"
            + (
                f", detect {self.detect_width}x{self.detect_height}"
                if self._detect_job
                else ""
            )
            + f" library={lib_path}"
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _slice_maps(map_x, map_y, x, y, w, h):
        return (
            np.ascontiguousarray(map_x[y:y + h, x:x + w], dtype=np.float32),
            np.ascontiguousarray(map_y[y:y + h, x:x + w], dtype=np.float32),
        )

    def _scaled_maps(self, scale: float):
        """Maps/intrinsics for the detection job (same ROI, scaled)."""
        new_k = self._rectifier.new_camera_matrix
        roi_x, roi_y, roi_w, roi_h = self._roi
        scaled_k = new_k.copy()
        scaled_k[0, 0] *= scale
        scaled_k[1, 1] *= scale
        scaled_k[0, 2] = (new_k[0, 2] - roi_x) * scale
        scaled_k[1, 2] = (new_k[1, 2] - roi_y) * scale
        out_w = max(1, int(round(roi_w * scale)))
        out_h = max(1, int(round(roi_h * scale)))
        if self._is_fisheye:
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                self._camera_matrix, self._dist4, np.eye(3), scaled_k,
                (out_w, out_h), cv2.CV_32FC1,
            )
        else:
            map_x, map_y = cv2.initUndistortRectifyMap(
                self._camera_matrix, self._dist_coeffs, None, scaled_k,
                (out_w, out_h), cv2.CV_32FC1,
            )
        base = self._rectifier.get_intrinsics()
        detect_intrinsics = {
            "fx": float(base["fx"]) * scale,
            "fy": float(base["fy"]) * scale,
            "cx": float(base["cx"]) * scale,
            "cy": float(base["cy"]) * scale,
            "img_width": out_w,
            "img_height": out_h,
        }
        return (
            np.ascontiguousarray(map_x, dtype=np.float32),
            np.ascontiguousarray(map_y, dtype=np.float32),
            out_w, out_h, detect_intrinsics,
        )

    def _set_job(self, job: int, map_x, map_y, width, height, interval):
        map_x = np.ascontiguousarray(map_x, dtype=np.float32)
        map_y = np.ascontiguousarray(map_y, dtype=np.float32)
        rc = self._lib.race_img_set_fisheye_job(
            self._handle, int(job),
            map_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            map_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(width), int(height), int(interval),
        )
        if rc != 0:
            raise RuntimeError(
                f"race_img_set_fisheye_job({job}) failed: {self._shim_error()}"
            )
        # Keep the host arrays alive as long as the pipeline (the shim
        # copies them, but this documents the ownership).
        self._maps.append((map_x, map_y))

    # ------------------------------------------------------------------ frame
    def decode_and_rectify(self, jpeg_bytes: bytes) -> np.ndarray:
        """Decode ``jpeg_bytes`` and rectify every configured job.

        Returns the full-resolution rectified BGR image (host) for
        annotation. The detection-resolution image stays on the device;
        use :attr:`detect_ptr`.
        """
        if self._handle is None:
            raise RuntimeError("GPU image pipeline is closed")
        buffer = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        rc = self._lib.race_img_decode(
            self._handle,
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(jpeg_bytes),
        )
        if rc != 0:
            raise RuntimeError(f"race_img_decode failed: {self._shim_error()}")
        for job in self._jobs:
            rc = self._lib.race_img_rectify(self._handle, job)
            if rc != 0:
                raise RuntimeError(
                    f"race_img_rectify({job}) failed: {self._shim_error()}"
                )
        display = np.empty(
            (self.display_height, self.display_width, 3), dtype=np.uint8
        )
        rc = self._lib.race_img_rectified_to_host(
            self._handle, 0,
            display.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.display_width * 3,
        )
        if rc != 0:
            raise RuntimeError(
                f"race_img_rectified_to_host failed: {self._shim_error()}"
            )
        return display

    @property
    def detect_ptr(self) -> int:
        """Device address of the detection-resolution rectified BGR image."""
        if self._handle is None:
            return 0
        ptr = self._lib.race_img_get_rectified(self._handle, self._detect_job)
        return int(ptr) if ptr else 0

    def encode(self, host_bgr: np.ndarray, quality: int = 90) -> bytes:
        """Encode a host BGR image (annotated frame) as JPEG."""
        if self._handle is None:
            raise RuntimeError("GPU image pipeline is closed")
        bgr = np.ascontiguousarray(host_bgr, dtype=np.uint8)
        height, width = bgr.shape[:2]
        out = ctypes.POINTER(ctypes.c_uint8)()
        length = ctypes.c_size_t()
        rc = self._lib.race_img_encode(
            self._handle,
            bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bgr.strides[0],
            width, height, int(quality),
            ctypes.byref(out), ctypes.byref(length),
        )
        if rc != 0:
            raise RuntimeError(f"race_img_encode failed: {self._shim_error()}")
        return ctypes.string_at(out, length.value)

    # -------------------------------------------------------------- life cycle
    def _shim_error(self) -> str:
        try:
            return self._lib.race_img_last_error().decode("utf-8", "replace")
        except Exception:
            return "<no error string>"

    def close(self) -> None:
        """Destroy the pipeline. Safe to call twice."""
        if getattr(self, "_handle", None):
            self._lib.race_img_destroy(self._handle)
            self._handle = None

    def __del__(self):  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass
