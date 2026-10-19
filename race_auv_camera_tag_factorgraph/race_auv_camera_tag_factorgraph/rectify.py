"""Image rectification: distorted camera image -> distortion-free pinhole image.

The tag detector and the factor graph both assume a pinhole camera: a 3D
point ``p = (x, y, z)`` in the camera frame lands at pixel
``u = fx * x / z + cx``, ``v = fy * y / z + cy``. The real lens (a fisheye
behind a dome port) does not do that, so every image is first remapped into
a *rectified* image that does, with a new camera matrix ``K_rect``.

How it works
------------
1. Once, at start-up: choose ``K_rect`` and build two remap tables
   (``map1``, ``map2``). For every output pixel they hold the input pixel to
   read from (fixed-point ``CV_16SC2``, 1/32 px resolution, which
   ``cv2.remap`` processes about twice as fast as float maps).
2. Every frame: one ``cv2.remap``, then a crop to the valid pixels.

Choosing ``K_rect``
-------------------
* fisheye (``cv2.fisheye``, 4 coefficients k1..k4):
  ``estimateNewCameraMatrixForUndistortRectify`` with ``balance = 0``, which
  scales the output so (almost) every output pixel sees real image content;
* pinhole (plumb-bob, ``k1 k2 p1 p2 [k3]``): ``getOptimalNewCameraMatrix``
  with ``alpha = 0`` (same idea).

Crop: the rectified frame can still have black (invalid) borders. A white
mask is remapped the same way, and the image is cropped to the bounding
rectangle of its largest valid region. Cropping shifts the principal point:
``cx -= x0``, ``cy -= y0``. The result ``K`` is what downstream consumers
(the dock_fuser YAML) must use; the node publishes it as ``CameraInfo``.

This reproduces ``race_auv_camera_pkg.image_processing.ImageRectifier``
(same ``K_rect``, same crop), so existing dock_fuser configs stay valid.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


class Rectifier:
    """Precomputed undistortion for one camera (see the module docstring).

    Parameters
    ----------
    K
        3x3 camera matrix of the **raw** (distorted) image.
    D
        Distortion coefficients: ``k1 k2 k3 k4`` (fisheye) or
        ``k1 k2 p1 p2 [k3]`` (pinhole).
    size
        ``(width, height)`` of the raw image.
    fisheye
        True for the ``cv2.fisheye`` model.
    """

    def __init__(self, K: np.ndarray, D: Sequence[float], size: Tuple[int, int],
                 fisheye: bool) -> None:
        K = np.asarray(K, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64).ravel()
        size = (int(size[0]), int(size[1]))
        eye = np.eye(3)
        if fisheye:
            D = D[:4]   # cv2.fisheye takes exactly k1..k4
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D, size, eye, balance=0.0)
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                K, D, eye, K_new, size, cv2.CV_16SC2)
        else:
            K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 0.0, size)
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, D, None, K_new, size, cv2.CV_16SC2)

        # Crop: bounding rectangle of the largest region of valid pixels.
        mask = cv2.remap(np.full(size[::-1], 255, np.uint8), self.map1, self.map2,
                         cv2.INTER_LINEAR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        else:
            x, y, w, h = 0, 0, size[0], size[1]
        self.roi = (x, y, w, h)

        # Camera matrix of the cropped, rectified image (principal point
        # moves with the crop origin; the focal lengths do not change).
        self.K = K_new.copy()
        self.K[0, 2] -= x
        self.K[1, 2] -= y
        self.width, self.height = w, h

    def rectify(self, image: np.ndarray) -> np.ndarray:
        """Undistort and crop one image (grayscale or colour)."""
        x, y, w, h = self.roi
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)[y:y + h, x:x + w]

    def describe(self) -> str:
        """One-line summary for the start-up log (the values dock_fuser must use)."""
        K = self.K
        return (f"rectified {self.width}x{self.height}: fx {K[0, 0]:.4f} fy {K[1, 1]:.4f} "
                f"cx {K[0, 2]:.4f} cy {K[1, 2]:.4f}")
