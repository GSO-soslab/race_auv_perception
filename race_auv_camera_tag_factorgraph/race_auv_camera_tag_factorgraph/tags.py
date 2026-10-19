"""tag36h11 detection, per-tag pose and image-quality checks on a rectified image.

Pipeline per image
------------------
1. **Detect** (apriltag3, ``tag36h11`` only). Each detection has an ID and 4
   corner pixels in the order ``lb, rb, rt, lt`` (left-bottom, right-bottom,
   right-top, left-top in the tag's own frame). That is the *canonical*
   order used throughout the stack (see :func:`tag_corners`). Detection does
   not depend on the tag's physical size, so tags of several sizes are found
   in one pass.
2. **Size lookup.** The physical size comes from the configured
   ``{id: size}`` table, so one detector serves 12.5 cm and 4 cm tags alike.
   IDs not in the table are ignored (real tags, just not ours).
3. **Image checks** (the dock_fuser gate relies on them, see below):
   border distance, minimum side length, PnP fit error.
4. **Pose** ``T_cam_tag`` with ``cv2.solvePnPGeneric(SOLVEPNP_IPPE_SQUARE)``.
   IPPE (infinitesimal plane-based pose estimation) solves the planar-square
   pose in closed form and returns *both* solutions of the flip ambiguity;
   the better-fitting one is published as the pose, and its RMS
   reprojection error is the fit check of step 3.

Tag frame (canonical, shared with race_auv_factorgraph and the station URDF)
---------------------------------------------------------------------------
Origin at the tag centre, x right, y down, z into the tag (away from the
camera when the tag faces it). With ``s`` the black-square side::

    i   corner   tag-frame point
    0   lb       (-s/2, +s/2, 0)
    1   rb       (+s/2, +s/2, 0)
    2   rt       (+s/2, -s/2, 0)
    3   lt       (-s/2, -s/2, 0)

apriltag3 returns its corners in exactly this order, and it is the order
``SOLVEPNP_IPPE_SQUARE`` requires for its object points. So the published
corners and the published pose agree by construction, and no reordering is
needed (the old cuAprilTags path needed one).

Image checks (moved here from the dock_fuser gate)
--------------------------------------------------
These are properties of one detection in one image, so they belong at the
source; dock_fuser then only checks consistency over time.

* ``min_edge_px``: every corner at least this far from the rectified image
  border. A tag cut by the border has corners the detector guessed.
* ``min_side_px``: shortest side of the corner quadrilateral. Below ~15 px
  the corners are too noisy to be worth a factor (a 12 cm tag at ~9 m, a
  4 cm tag at ~3 m, with f ~ 1165 px).
* ``max_fit_rms_px``: RMS reprojection error of the best IPPE pose (per pixel
  coordinate). A clean tag fits to ~0.1-0.4 px; an occluded, blurred or
  badly segmented quad is not a square in perspective. The pose has 6
  parameters for 8 coordinates, so it absorbs small corner errors: this
  catches gross ones (one corner off by ~10 px or more). Tank data
  (2026-09-16, 7459 detections): median 0.08 px, p95 0.41 px.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

FAMILY = "tag36h11"


def tag_corners(size: float) -> np.ndarray:
    """The 4 corners of a tag of black-square side ``size`` in its own frame, ``(4, 3)``.

    Canonical order ``lb, rb, rt, lt`` (see the module docstring).
    """
    h = 0.5 * float(size)
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


@dataclass
class TagDetection:
    """One accepted tag in one rectified image."""

    tag_id: int
    size: float               # black-square side [m]
    corners: np.ndarray       # (4, 2) pixels, canonical order lb, rb, rt, lt
    T_cam_tag: np.ndarray     # (4, 4) pose of the tag in the camera (optical) frame
    fit_rms_px: float         # RMS reprojection error of that pose
    hamming: int = 0          # bits corrected while decoding (0 = exact code)
    margin: float = 0.0       # apriltag3 decision margin (higher = more contrast)


@dataclass
class FrameResult:
    """Everything one image produced: accepted tags and why the others were dropped."""

    tags: List[TagDetection] = field(default_factory=list)
    rejected: List[Tuple[int, str]] = field(default_factory=list)   # (tag id, reason)


def ippe_pose(corners: np.ndarray, size: float, K: np.ndarray) -> Tuple[np.ndarray, float]:
    """Best IPPE pose ``T_cam_tag`` (4x4) and its RMS fit error (px) for one tag.

    Solutions that put the tag behind the camera are discarded. Returns
    ``(None, inf)`` if no valid solution exists.
    """
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            tag_corners(size), corners.reshape(4, 1, 2).astype(np.float64), K, None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None, float("inf")
    best, best_err = None, float("inf")
    for i in range(int(n)):
        t = np.asarray(tvecs[i], dtype=np.float64).reshape(3)
        err = float(np.ravel(errs)[i])
        if t[2] > 0.0 and err < best_err:
            T = np.eye(4)
            T[:3, :3] = cv2.Rodrigues(np.asarray(rvecs[i], dtype=np.float64))[0]
            T[:3, 3] = t
            best, best_err = T, err
    return best, best_err


class TagDetector:
    """apriltag3 ``tag36h11`` detector with per-ID sizes and image checks.

    Parameters
    ----------
    sizes
        ``{tag id: black-square side [m]}``. Only these IDs are reported.
    K, width, height
        Camera matrix and size of the **rectified** image.
    threads, decimate, refine_edges
        apriltag3 settings. ``decimate`` > 1 finds quads on a downscaled image
        (faster; corners are still refined at full resolution when
        ``refine_edges`` is on). Lower it if small tags are missed at range.
    min_edge_px, min_side_px, max_fit_rms_px
        Image checks (see the module docstring).
    """

    def __init__(self, sizes: Dict[int, float], K: np.ndarray, width: int, height: int,
                 threads: int = 4, decimate: float = 2.0, refine_edges: bool = True,
                 min_edge_px: float = 10.0, min_side_px: float = 15.0,
                 max_fit_rms_px: float = 1.5) -> None:
        from apriltag import apriltag   # AprilRobotics/apriltag Python binding
        self._detector = apriltag(FAMILY, threads=int(threads), decimate=float(decimate),
                                  refine_edges=bool(refine_edges))
        self.sizes = {int(k): float(v) for k, v in sizes.items()}
        self.K = np.asarray(K, dtype=np.float64)
        self.width, self.height = int(width), int(height)
        self.min_edge_px = float(min_edge_px)
        self.min_side_px = float(min_side_px)
        self.max_fit_rms_px = float(max_fit_rms_px)

    def detect(self, gray: np.ndarray) -> FrameResult:
        """Detect, check and pose every configured tag in one rectified grayscale image."""
        out = FrameResult()
        for d in self._detector.detect(gray):
            tag_id = int(d["id"])
            size = self.sizes.get(tag_id)
            if size is None:
                continue
            corners = np.asarray(d["lb-rb-rt-lt"], dtype=np.float64).reshape(4, 2)
            reason = self._image_check(corners)
            T, err = (None, float("inf")) if reason else ippe_pose(corners, size, self.K)
            if not reason and T is None:
                reason = "PnP failed"
            elif not reason and err > self.max_fit_rms_px:
                reason = f"PnP fit {err:.1f} px > {self.max_fit_rms_px:.1f} (not a clean square)"
            if reason:
                out.rejected.append((tag_id, reason))
                continue
            out.tags.append(TagDetection(tag_id, size, corners, T, err,
                                         int(d.get("hamming", 0)), float(d.get("margin", 0.0))))
        return out

    def _image_check(self, corners: np.ndarray) -> str:
        """Border and size checks. Empty string if the detection passes."""
        m = self.min_edge_px
        u, v = corners[:, 0], corners[:, 1]
        if (u.min() < m or v.min() < m
                or u.max() > self.width - 1 - m or v.max() > self.height - 1 - m):
            return f"corner within {m:.0f} px of the image border"
        side = float(np.min(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))
        if side < self.min_side_px:
            return f"tag side {side:.0f} px < {self.min_side_px:.0f}"
        return ""
