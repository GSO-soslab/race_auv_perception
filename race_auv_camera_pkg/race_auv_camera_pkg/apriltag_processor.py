"""Multi-family AprilTag detection backed by ``AprilRobotics/apriltag``.

This module also hosts :func:`annotate_detections`, the shared drawing
routine used by both the ``python`` backend (:class:`AprilTagDetector`)
and the ``cuda`` backend (``apriltag_cuda.CuAprilTagDetector``). The
``apriltag3`` Python binding is imported lazily in
:class:`AprilTagDetector.__init__`, so the CUDA backend can import this
module on hosts where ``apriltag3`` is not installed.

A single :class:`AprilTagDetector` owns one ``apriltag.apriltag``
instance *per family* configured for it. Because the upstream
``detect()`` call also does not span families in one pass, we run
one quad-scan per family and stitch the results. Per-tag size is
looked up *after* decode (each detection carries its ``family`` and
``id``), so a single detector instance can serve tags of mixed
sizes within the same family.

Decoded detections whose ``(family, id)`` is not in the configured
size table are dropped (they are real detections, just not tags we
care about). Pose is recovered per-detection via the upstream
library's ``estimate_tag_pose()`` -- no ``cv2.SOLVEPNP_IPPE_SQUARE``
or other external solver is involved.

On every detection the detector also draws:

* a colored bounding box (green = good pose, red = bad pose),
* the tag's axes via ``cv2.drawFrameAxes`` (RGB = X, Y, Z in camera frame),
* a label block stacked above the box: ``ID``, ``xyz``, ``rpy``.

All drawing happens on the caller-provided image in-place; ``detect``
returns structured data for the caller to publish as
``vision_msgs/Detection3DArray``.

Coordinate convention
---------------------
All poses returned by :meth:`AprilTagDetector.detect` are
``T_camera_to_tag`` in the *rectified* camera frame, i.e. the frame
the rectifiers in :mod:`image_processing` produce. The pose is a 4x4
homogeneous matrix with the rotation block in ``T[:3, :3]`` and the
translation column in ``T[:3, 3]`` (camera-frame coordinates of the
tag's centre).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# On-image drawing constants
# =============================================================================
# All colours are BGR (OpenCV convention).
_BOX_OK = (0, 255, 0)            # green box for usable poses
_BOX_BAD = (0, 0, 255)           # red box for rejected poses
_TEXT_PRIMARY = (255, 0, 255)    # magenta -- ID line
_TEXT_SECONDARY = (0, 255, 255)  # yellow  -- xyz / rpy lines

# Font and stroke sizing.
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_BIG = 0.55
_FONT_SCALE_SMALL = 0.5
_TEXT_THICKNESS = 1
_LINE_THICKNESS = 2

# Per-text vertical stride (pixels) so the stacked label block is evenly
# spaced. Roughly matches the rendered line height of ``_FONT_SCALE_BIG``
# with ``_TEXT_THICKNESS=1``.
_LABEL_STRIDE = 18

# Length of the axis arrows drawn by ``cv2.drawFrameAxes``, expressed
# in metres. Half the tag side is a good rule of thumb: short enough
# not to overlap neighbouring tags, long enough to read.
_AXIS_LENGTH_DIVISOR = 2.0


# =============================================================================
# Small math helpers
# =============================================================================
def _rotation_matrix_to_euler_xyz(Rmat: np.ndarray) -> tuple[float, float, float]:
    """Roll / pitch / yaw (degrees) of ``Rmat`` in scipy's ``'xyz'`` order.

    We delegate to scipy because its reference implementation is the
    source of truth for the on-image RPY labels -- the closed-form
    shortcut is tempting and the sign conventions are easy to get
    wrong, especially around gimbal lock.
    """
    e = R.from_matrix(Rmat).as_euler("xyz", degrees=True)
    return float(e[0]), float(e[1]), float(e[2])


class AprilTagDetector:
    """Detect AprilTags from one or more families with mixed per-tag sizes.

    One :class:`AprilTagDetector` is responsible for every tag in the
    configured tag list. The parent node (``apriltag_detector_node``)
    creates exactly one of these per pipeline, passing in the full set
    of families and a ``(family, id) -> size_m`` lookup.

    For each configured family we instantiate one
    ``apriltag.apriltag(family=...)`` and run its ``detect()`` once per
    frame. Decoded tags whose ``(family, id)`` is in ``id_to_size``
    are emitted with the corresponding size fed into the upstream
    library's ``estimate_tag_pose()`` for the pose solve.

    Parameters
    ----------
    families
        Sequence of family names (e.g. ``["tag25h9", "tag36h11"]``)
        to instantiate one ``apriltag`` detector for. Order doesn't
        matter; each family is scanned once per frame.
    id_to_size
        Mapping ``(family, tag_id) -> size_m``. Decoded detections
        whose key is not present here are dropped (they're tags we
        don't care about).
    camera_intrinsics
        Dict with keys ``fx``, ``fy``, ``cx``, ``cy`` (pixels) for the
        *rectified* image the detector will receive. If the parent
        pipeline runs ``process_scale < 1``, these are the intrinsics
        of the downscaled image, not the full-resolution one.
    camera_distortion
        Distortion coefficients for the rectified image. Pose
        estimation does not use these; they are kept here so
        :func:`cv2.drawFrameAxes` can label its axes.
    image_size
        Dict with ``img_width`` and ``img_height`` of the (possibly
        downscaled) image the detector will receive.
    logger
        A ``rclpy``-compatible logger.
    detector_params
        Dict of tuning parameters forwarded to ``apriltag.apriltag``
        (``nthreads``, ``quad_decimate``, ``quad_sigma``,
        ``refine_edges``).
    """

    def __init__(
        self,
        families: Sequence[str],
        id_to_size: Dict[Tuple[str, int], float],
        camera_intrinsics: dict,
        camera_distortion: Sequence[float],
        image_size: dict,
        logger,
        detector_params: dict,
    ):
        self.logger = logger
        self._id_to_size: Dict[Tuple[str, int], float] = {
            (str(f), int(i)): float(s)
            for (f, i), s in id_to_size.items()
        }
        self._families: Tuple[str, ...] = tuple(str(f) for f in families)
        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])
        self._camera_params = (fx, fy, cx, cy)

        # Distortion + camera matrix are needed for ``cv2.drawFrameAxes``
        # only. The upstream pose solve is intrinsics-only, but the
        # axis overlay projects the tag's local axes through this K/D
        # pair onto the image. The rectified image has zero distortion,
        # so the D we receive here should be all zeros in practice.
        self.camera_matrix = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.array(camera_distortion, dtype=np.float32)
        if self.dist_coeffs.shape[0] < 4:
            padded = np.zeros(5, dtype=np.float32)
            padded[: self.dist_coeffs.shape[0]] = self.dist_coeffs
            self.dist_coeffs = padded
        else:
            self.dist_coeffs = self.dist_coeffs[:5]

        self.img_width = int(image_size["img_width"])
        self.img_height = int(image_size["img_height"])

        # One detector per family. The upstream wrapper is also
        # single-family-per-instance, so a multi-family tag list still
        # requires one quad-scan per family. Within a single family,
        # tags of mixed sizes share that one scan: per-tag size is
        # looked up post-decode and fed into ``estimate_tag_pose``.
        try:
            from apriltag import apriltag as _apriltag_factory
        except ImportError as e:
            raise RuntimeError(
                "The 'python' AprilTag backend requires the apriltag3 Python "
                "binding (AprilRobotics/apriltag). Install it (for example "
                "'sudo apt install python3-apriltag') or set "
                "detector_backend: 'cuda' in the detector YAML."
            ) from e

        self._detectors: Dict[str, Any] = {}
        for fam in self._families:
            try:
                self._detectors[fam] = _apriltag_factory(
                    family=fam,
                    threads=int(detector_params.get("nthreads", 4)),
                    decimate=float(detector_params.get("quad_decimate", 2.0)),
                    blur=float(detector_params.get("quad_sigma", 0.0)),
                    refine_edges=bool(detector_params.get("refine_edges", True)),
                )
                self.logger.info(
                    f"apriltag3 ready: family='{fam}' "
                    f"image={self.img_width}x{self.img_height} "
                    f"params={dict(detector_params)}"
                )
            except Exception as e:
                self.logger.error(f"apriltag3 init failed for {fam}: {e}")

    # =====================================================================
    # detect
    # =====================================================================
    def detect(self, gray_image: np.ndarray) -> list[dict]:
        """Run detection on a grayscale image and return structured results.

        Each result is a dict with keys:

        * ``family``    -- ``str``, the canonical family name (e.g. ``"tag36h11"``).
        * ``tag_id``    -- ``int``, the numeric id within that family.
        * ``size``      -- ``float``, the physical side length in metres
          (looked up from the configured size table). Used by
          :meth:`annotate` for axis length; downstream consumers
          (``Detection3DArray``) don't need it.
        * ``T``         -- ``np.ndarray``, 4x4 ``T_camera_to_tag``.
        * ``corners``   -- ``np.ndarray``, ``Nx2 float32`` corner pixels
          in the upstream order (``lb-rb-rt-lt``: bottom-left,
          bottom-right, top-right, top-left).
        * ``bad_pose``  -- ``bool``, ``True`` when the pose could not be
          recovered. In that case ``T`` is the identity.

        Parameters
        ----------
        gray_image
            ``HxW`` ``uint8`` grayscale image. Must already be
            rectified (and optionally downscaled -- see
            ``process_scale``) so that ``self._camera_params`` matches
            the image coordinate system.
        """
        if gray_image is None:
            self.logger.warn("Received a null image for AprilTag detection.")
            return []

        fx, fy, cx, cy = self._camera_params
        out: list[dict] = []
        for family, det in self._detectors.items():
            try:
                raw = det.detect(gray_image)
            except Exception as e:
                self.logger.warn(
                    f"apriltag3 ({family}) detect failed: {e}; skipping."
                )
                continue
            for d in raw:
                try:
                    tag_id = int(d["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                key = (family, tag_id)
                size = self._id_to_size.get(key)
                if size is None:
                    continue

                T = np.eye(4)
                bad_pose = False
                try:
                    pose = det.estimate_tag_pose(
                        d, float(size), fx, fy, cx, cy,
                    )
                    R_mat = np.asarray(pose["R"], dtype=np.float64)
                    t_vec = np.asarray(pose["t"], dtype=np.float64).reshape(3)
                    T[:3, :3] = R_mat
                    T[:3, 3] = t_vec
                except Exception:
                    bad_pose = True

                corners = np.asarray(
                    d["lb-rb-rt-lt"], dtype=np.float32,
                ).reshape(4, 2)

                out.append({
                    "family": family,
                    "tag_id": tag_id,
                    "size": float(size),
                    "T": T,
                    "corners": corners,
                    "bad_pose": bad_pose,
                })
        return out

    # =====================================================================
    # annotate
    # =====================================================================
    def annotate(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes, axes, and id labels for ``detections`` in-place.

        Green boxes indicate a usable pose; red boxes indicate a pose
        that was rejected (very rare). The label block above each box
        shows:

        * ``ID: <family>:<id>``           (magenta, bigger)
        * ``xyz: (x, y, z)``              (yellow, metres)
        * ``rpy: (roll, pitch, yaw)``     (yellow, degrees)

        The 3D axes are drawn via ``cv2.drawFrameAxes`` (RGB = X, Y, Z)
        with length ``tag_size / 2`` so they stay compact. RPY / axes
        are only drawn for usable poses.

        Parameters
        ----------
        image
            BGR image to draw on (modified in place).
        detections
            List of dicts as returned by :meth:`detect`. ``corners``
            must already be in this image's coordinate system (the
            parent node scales them up from the detector's coordinate
            system when ``process_scale < 1``).

        Returns
        -------
        np.ndarray
            ``image``, returned for chaining convenience.
        """
        return annotate_detections(
            image, detections, self.camera_matrix, self.dist_coeffs, self.logger,
        )

# =============================================================================
# Shared annotation
# =============================================================================
def annotate_detections(
    image: np.ndarray,
    detections: list[dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    logger,
) -> np.ndarray:
    """Draw bounding boxes, axes, and id labels for ``detections`` in-place.

    Shared by :class:`AprilTagDetector` (python backend) and
    ``apriltag_cuda.CuAprilTagDetector`` (CUDA backend) so the two
    backends produce pixel-identical overlays.

    Green boxes indicate a usable pose; red boxes indicate a pose that
    was rejected (very rare). The label block above each box shows:

    * ``ID: <family>:<id>``           (magenta, bigger)
    * ``xyz: (x, y, z)``              (yellow, metres)
    * ``rpy: (roll, pitch, yaw)``     (yellow, degrees)

    The 3D axes are drawn via ``cv2.drawFrameAxes`` (RGB = X, Y, Z) with
    length ``tag_size / 2`` so they stay compact. RPY / axes are only
    drawn for usable poses.

    Parameters
    ----------
    image
        BGR image to draw on (modified in place).
    detections
        List of dicts as returned by each backend's ``detect``.
        ``corners`` must already be in this image's coordinate system.
    camera_matrix, dist_coeffs
        Pinhole model used to project the tag axes.
    logger
        A ``rclpy``-compatible logger.

    Returns
    -------
    np.ndarray
        ``image``, returned for chaining convenience.
    """
    if image is None:
        return image

    for det in detections:
        family = det["family"]
        tag_id = det["tag_id"]
        corners = det["corners"].astype(int)
        T = det["T"]
        bad = det["bad_pose"]
        tag_size = float(det.get("size", 0.0))

        if bad:
            _draw_bad_pose(image, corners, family, tag_id)
            continue

        tvec = T[:3, 3]
        Rmat = T[:3, :3]
        try:
            rvec, _ = cv2.Rodrigues(Rmat)
        except (ValueError, cv2.error) as e:
            logger.warn(
                f"Could not convert rotation for {family}:{tag_id}; "
                f"marking as bad pose. Error: {e}"
            )
            _draw_bad_pose(image, corners, family, tag_id)
            continue

        roll, pitch, yaw = _rotation_matrix_to_euler_xyz(Rmat)

        cv2.polylines(
            image, [corners], isClosed=True,
            color=_BOX_OK, thickness=_LINE_THICKNESS,
        )

        axis_length = tag_size / _AXIS_LENGTH_DIVISOR if tag_size > 0 else 0.05
        cv2.drawFrameAxes(
            image,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            axis_length,
        )

        anchor_x = int(corners[:, 0].min())
        anchor_y = int(corners[:, 1].min()) - _LABEL_STRIDE
        lines = [
            (f"ID: {family}:{tag_id}",  _TEXT_PRIMARY,   _FONT_SCALE_BIG,  _TEXT_THICKNESS),
            (f"xyz: ({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})",
                                            _TEXT_SECONDARY, _FONT_SCALE_SMALL, _TEXT_THICKNESS),
            (f"rpy: ({roll:.0f}, {pitch:.0f}, {yaw:.0f})",
                                            _TEXT_SECONDARY, _FONT_SCALE_SMALL, _TEXT_THICKNESS),
        ]
        _draw_label_block(image, anchor_x, anchor_y, lines)

    return image


def _draw_bad_pose(
    image: np.ndarray,
    corners: np.ndarray,
    family: str,
    tag_id: int,
) -> None:
    """Draw the red box + id label used when a detection's pose is rejected.

    Called from :func:`annotate_detections` when ``det["bad_pose"]`` is
    true or when :func:`cv2.Rodrigues` fails on the rotation matrix. No
    axes / xyz / rpy are drawn because we don't have a usable pose.
    """
    cv2.polylines(
        image, [corners], isClosed=True,
        color=_BOX_BAD, thickness=_LINE_THICKNESS,
    )
    anchor = tuple(int(v) for v in corners[0])
    cv2.putText(
        image,
        f"ID: {family}:{tag_id} (Bad Pose)",
        (anchor[0], anchor[1] - 5),
        _FONT, 0.7, _BOX_BAD, 2,
    )


def _draw_label_block(
    image: np.ndarray,
    x: int,
    start_y: int,
    lines: list[tuple[str, tuple[int, int, int], float, int]],
) -> None:
    """Stack ``lines`` vertically starting at ``(x, start_y)``.

    Each line is ``(text, color_bgr, font_scale, thickness)``. Each
    subsequent line is placed ``_LABEL_STRIDE`` pixels below the
    previous one. Anti-aliased (``cv2.LINE_AA``) for legibility on the
    JPEG-compressed output.
    """
    y = start_y
    for text, color, scale, thick in lines:
        cv2.putText(image, text, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)
        y += _LABEL_STRIDE
