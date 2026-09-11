"""CPU image rectification for the AprilTag pipeline.

This module provides a single ``ImageRectifier`` class that:

1. Builds pre-computed OpenCV remap maps once at construction time.
2. Rectifies every incoming frame via ``cv2.remap`` (CPU) and ROI-crops
   it to the largest rectangle of valid (non-black) pixels.
3. Exposes the rectified image's intrinsics (``fx``, ``fy``, ``cx``,
   ``cy``) and size, with zero distortion coefficients by construction.

GPU acceleration via ``cv2.cuda`` was removed because OpenCV's CUDA
support is not reliably available on the platforms this package is
deployed on (Jetson Orin, generic Linux, the CI runners): the
``opencv-python`` PyPI wheels do not include CUDA, and the JetPack
``python3-opencv`` system package either ships without NVCOMPRESS or
without CUDA runtime libraries in the configuration we use. The
``cv2.cuda`` modules therefore either raise ``AttributeError`` at import
time or fail at first use with "no CUDA-enabled devices". A failed
GPU path wastes CPU time on fallbacks and complicates the operator's
mental model, so it was removed entirely; everything runs on the CPU.

Public API
----------

* :class:`ImageRectifier` -- the CPU rectifier (see its docstring).
* :class:`RectifiedIntrinsics` -- dict subclass returned by
  :meth:`ImageRectifier.get_intrinsics`.
* :func:`build_rectifier` -- thin factory used by the detector node;
  always returns the CPU backend.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np


_logger = logging.getLogger(__name__)


# =============================================================================
# Public type returned by the rectifier
# =============================================================================
class RectifiedIntrinsics(dict):
    """Plain ``dict`` subclass carrying the rectified image's intrinsics.

    Keys:

    * ``fx``, ``fy`` -- focal lengths (pixels) for the rectified image.
    * ``cx``, ``cy`` -- principal point (pixels) for the rectified image.
    * ``img_width``, ``img_height`` -- size of the rectified image in
      pixels.
    * ``distortion`` -- always a length-5 array of zeros (a rectified
      image is, by construction, distortion-free).
    """


# =============================================================================
# CPU backend
# =============================================================================
class ImageRectifier:
    """CPU image rectifier.

    Pre-computes remap maps for either the fisheye or the standard
    (plumb-bob) distortion model and exposes ``rectify(image)`` that
    undistorts and ROI-crops an image in a single call.

    The CPU path uses OpenCV's ``cv2.remap`` (or ``cv2.undistort`` for
    the plumb-bob model), both of which are pure CPU operations. On a
    1600x1200 frame at 5 Hz this is ~10-15 ms / frame on a desktop and
    similar on a Jetson Orin -- acceptable for a single-camera AprilTag
    pipeline.

    Parameters
    ----------
    logger
        Logger used for init messages.
    camera_matrix
        ``3x3`` intrinsic matrix ``K`` for the *original* (distorted)
        image.
    dist_coeffs
        1-D distortion coefficients. Four for the fisheye model
        ``(k1, k2, k3, k4)`` and typically five for the standard
        plumb-bob model ``(k1, k2, p1, p2, k3)``.
    image_size
        ``(width, height)`` of the *original* (distorted) image.
    is_fisheye
        ``True`` for the fisheye model (``cv2.fisheye.*``),
        ``False`` for the standard plumb-bob model
        (``cv2.undistort`` / ``cv2.getOptimalNewCameraMatrix``).
    crop_to_valid_pixels
        ``True`` to crop the rectified image to the largest bounding
        rectangle of valid (non-black) pixels, removing the borders
        introduced by rectification. ``False`` to keep the full
        rectified frame (introduces black borders).
    """

    def __init__(
        self,
        logger,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: Tuple[int, int],
        is_fisheye: bool,
        crop_to_valid_pixels: bool,
    ) -> None:
        # ------------------------------------------------------------------ state
        self._logger = logger
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self._image_size = image_size
        self._is_fisheye = bool(is_fisheye)
        self._crop = bool(crop_to_valid_pixels)
        # The two remap maps are populated by ``_init_fisheye`` /
        # ``_init_standard`` below. They are ``np.float32`` arrays of
        # shape ``(H, W)`` (one entry per output pixel giving the
        # source pixel to read from).
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None

        self._logger.info(
            f"ImageRectifier init: backend=cpu fisheye={self._is_fisheye} "
            f"crop={self._crop} size={self._image_size[0]}x{self._image_size[1]}"
        )

        # ------------------------------------------------------------------ build
        if self._is_fisheye:
            self._init_fisheye()
        else:
            self._init_standard()

        # After ROI cropping the principal point shifts by ``(roi.x, roi.y)``
        # and the image size shrinks to ``(roi.w, roi.h)``.
        self._new_width = self.roi[2]
        self._new_height = self.roi[3]
        self._final_camera_matrix = self.new_camera_matrix.copy()
        self._final_camera_matrix[0, 2] -= self.roi[0]  # cx adjustment
        self._final_camera_matrix[1, 2] -= self.roi[1]  # cy adjustment

        self._logger.info(
            f"Rectifier ready: {self._new_width}x{self._new_height} "
            f"fx={self._final_camera_matrix[0, 0]:.2f} "
            f"fy={self._final_camera_matrix[1, 1]:.2f} "
            f"cx={self._final_camera_matrix[0, 2]:.2f} "
            f"cy={self._final_camera_matrix[1, 2]:.2f}"
        )

    # ------------------------------------------------------------------ init helpers
    def _init_fisheye(self) -> None:
        """Build the remap maps for the fisheye model.

        The fisheye API expects exactly 4 coefficients ``(k1, k2, k3, k4)``.
        If more are passed in (some camera drivers emit 8) we use the
        first four and warn about the rest.
        """
        d = np.asarray(self._dist_coeffs, dtype=np.float64).ravel()[:4]
        if d.size != 4:
            raise ValueError(
                f"Fisheye model requires exactly 4 distortion coefficients "
                f"(k1, k2, k3, k4); got {self._dist_coeffs.size}."
            )
        if self._dist_coeffs.size > 4:
            self._logger.warn(
                f"Fisheye model received {self._dist_coeffs.size} distortion "
                "coefficients; using the first 4 (k1, k2, k3, k4) and "
                "ignoring the rest."
            )

        # ``balance=0.0`` crops to valid pixels; ``balance=1.0`` keeps
        # all pixels (with black borders).
        balance = 0.0 if self._crop else 1.0
        self.new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self._camera_matrix, d, self._image_size, np.eye(3), balance=balance,
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self._camera_matrix, d, np.eye(3), self.new_camera_matrix,
            self._image_size, cv2.CV_32F,
        )

        # ROI: find the largest contour of valid pixels in an
        # all-ones mask after the same remap; its bounding rect is
        # the crop we want.
        if self._crop:
            mask = np.ones(self._image_size[::-1], dtype=np.uint8) * 255  # (H, W)
            undistorted_mask = cv2.remap(
                mask, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
            )
            contours, _ = cv2.findContours(
                undistorted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                self.roi = cv2.boundingRect(largest_contour)
            else:
                self._logger.warn(
                    "No contour in fisheye undistorted mask; not cropping."
                )
                self.roi = (0, 0, self._image_size[0], self._image_size[1])
        else:
            self.roi = (0, 0, self._image_size[0], self._image_size[1])

    def _init_standard(self) -> None:
        """Build the rectified camera matrix + ROI for the plumb-bob model.

        ``cv2.getOptimalNewCameraMatrix`` returns both at once: with
        ``alpha=0.0`` it crops to valid pixels, with ``alpha=1.0`` it
        keeps all pixels.
        """
        if len(self._dist_coeffs) < 4:
            self._logger.warn(
                f"Standard model with only {len(self._dist_coeffs)} distortion "
                "coefficients; expected at least 4 (k1, k2, p1, p2)."
            )
        alpha = 0.0 if self._crop else 1.0
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix, self._dist_coeffs, self._image_size,
            alpha, self._image_size,
        )

    # ------------------------------------------------------------------ public API
    def rectify(self, image: np.ndarray) -> np.ndarray:
        """Rectify and ROI-crop ``image``. Returns a new BGR array.

        The CPU ``cv2.remap`` call below reads from ``self.map1`` /
        ``self.map2`` and writes into a freshly allocated BGR buffer of
        size ``(self._new_width, self._new_height)``.
        """
        if self._is_fisheye:
            rect_img = cv2.remap(
                image, self.map1, self.map2,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            rect_img = cv2.undistort(
                image, self._camera_matrix, self._dist_coeffs,
                None, self.new_camera_matrix,
            )
        x, y, w, h = self.roi
        return rect_img[y:y + h, x:x + w]

    def get_intrinsics(self) -> RectifiedIntrinsics:
        """Return the rectified (and ROI-cropped) image's intrinsics.

        The principal point has already been shifted by the ROI origin,
        so a downstream caller can use ``fx``, ``fy``, ``cx``, ``cy``
        directly on the rectified image with no extra transform.
        """
        return RectifiedIntrinsics(
            fx=float(self._final_camera_matrix[0, 0]),
            fy=float(self._final_camera_matrix[1, 1]),
            cx=float(self._final_camera_matrix[0, 2]),
            cy=float(self._final_camera_matrix[1, 2]),
            img_width=int(self._new_width),
            img_height=int(self._new_height),
            distortion=np.zeros(5, dtype=np.float32),
        )

    def get_maps(self):
        """Return ``(map_x, map_y)`` (CV_32FC1) mapping output -> input.

        The maps are full-frame, i.e. *not* ROI-cropped; callers that
        want the rectifier's final image must slice them by ``self.roi``.
        They are used by the GPU (VPI) rectifier so it produces the exact
        same geometry as the CPU path.
        """
        if self._is_fisheye:
            return self.map1, self.map2
        map_x, map_y = cv2.initUndistortRectifyMap(
            self._camera_matrix, self._dist_coeffs, None,
            self.new_camera_matrix, self._image_size, cv2.CV_32FC1,
        )
        return map_x, map_y


# =============================================================================
# Selection helper
# =============================================================================
def build_rectifier(
    logger,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: Tuple[int, int],
    is_fisheye: bool,
    crop_to_valid_pixels: bool,
) -> Tuple[ImageRectifier, str]:
    """Build an :class:`ImageRectifier` (always CPU).

    Parameters
    ----------
    logger
        Logger passed through to the rectifier for init / ready lines.
    camera_matrix, dist_coeffs, image_size, is_fisheye, crop_to_valid_pixels
        Forwarded to :class:`ImageRectifier` -- see its docstring.

    Returns
    -------
    (rectifier, backend_name)
        ``backend_name`` is always ``"cpu"``; the second tuple element is
        kept so callers (and the HW-banner log) have a consistent
        surface.
    """
    r = ImageRectifier(
        logger=logger,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        is_fisheye=is_fisheye,
        crop_to_valid_pixels=crop_to_valid_pixels,
    )
    return r, "cpu"