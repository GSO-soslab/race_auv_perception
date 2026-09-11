"""Geometry / URDF helpers shared by the AprilTag pipeline nodes.

Both ``apriltag_detector_node`` (per-camera detector) and
``apriltag_fuser_node`` (multi-camera fuser) need the same primitives:

* SVD-based rotation sanitization (the detector occasionally returns
  slightly non-right-handed rotation matrices for noisy / degenerate
  detections, which crashes ``scipy.Rotation.from_matrix``).
* Bad-rotation heuristic to filter out untrustworthy detections.
* 4x4 homogeneous <-> ROS message conversion (Pose / TransformStamped).
* The Umeyama / SVD joint SE(3) solver used to fit one rigid
  ``T_camera_to_base`` to a set of (observed, known) tag-pose pairs.
  Supports an optional per-pair weight (e.g. ``1/d^2`` so closer tags
  dominate) and a RANSAC wrapper for outlier-robust fitting.
* The YAML-driven URDF path resolution used by the bridge launches.

Centralizing these keeps the detector and the fuser byte-for-byte aligned
on the math (and the small numerical edge cases), so a fix in one place
applies to both.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from geometry_msgs.msg import Pose, TransformStamped
from scipy.spatial.transform import Rotation as R


def sanitize_rotation(M: np.ndarray) -> np.ndarray:
    """Project a near-rotation 3x3 matrix onto SO(3) via SVD.

    The pose estimate returned by the detector's ``estimate_tag_pose``
    can be a slightly non-orthogonal rotation when the tag is noisy or
    its pose is degenerate; the matrix's determinant may end up
    negative or zero, which makes ``scipy.Rotation.from_matrix`` raise
    ``ValueError: Non-positive determinant``. This helper returns the
    closest proper rotation so downstream quaternion conversion never
    crashes.
    """
    M = np.asarray(M, dtype=np.float64)
    U, _, Vt = np.linalg.svd(M)
    R_fixed = U @ Vt
    if np.linalg.det(R_fixed) < 0.0:
        Vt[-1, :] *= -1.0
        R_fixed = U @ Vt
    return R_fixed


def is_bad_rotation(M: np.ndarray) -> bool:
    """Heuristic: this 3x3 is not a usable proper rotation.

    True if the matrix is non-finite, has det <= 0, or is far from
    orthogonal. Used to decide whether a detection's pose is trustworthy
    enough to be included in the joint base-pose solve.
    """
    M = np.asarray(M, dtype=np.float64)
    if not np.all(np.isfinite(M)):
        return True
    try:
        if np.linalg.det(M) <= 1e-6:
            return True
    except Exception:
        return True
    if not np.allclose(M @ M.T, np.eye(3), atol=1e-3):
        return True
    return False


def matrix_to_pose_msg(T: np.ndarray) -> Pose:
    """Homogeneous 4x4 -> geometry_msgs/Pose. Robust to noisy rotations."""
    p = T[:3, 3]
    R_clean = sanitize_rotation(T[:3, :3])
    q = R.from_matrix(R_clean).as_quat()  # xyzw
    msg = Pose()
    msg.position.x = float(p[0])
    msg.position.y = float(p[1])
    msg.position.z = float(p[2])
    msg.orientation.x = float(q[0])
    msg.orientation.y = float(q[1])
    msg.orientation.z = float(q[2])
    msg.orientation.w = float(q[3])
    return msg


def matrix_to_transform_stamped(
    T: np.ndarray, parent: str, child: str, stamp,
) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    p = T[:3, 3]
    R_clean = sanitize_rotation(T[:3, :3])
    q = R.from_matrix(R_clean).as_quat()
    msg.transform.translation.x = float(p[0])
    msg.transform.translation.y = float(p[1])
    msg.transform.translation.z = float(p[2])
    msg.transform.rotation.x = float(q[0])
    msg.transform.rotation.y = float(q[1])
    msg.transform.rotation.z = float(q[2])
    msg.transform.rotation.w = float(q[3])
    return msg


def solve_cam_to_base(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    weights: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Joint SE(3) solve: one rigid transform explaining all tag observations.

    Parameters
    ----------
    pairs
        List of ``(T_cam_to_tag_observed, T_base_to_tag_known)`` 4x4
        transforms. ``T_base_to_tag_known`` comes from the URDF.
    weights
        Optional per-pair weight, shape ``(len(pairs),)``. Used to
        scale the contribution of each tag to the SVD solve -- for
        example the fuser passes ``1/d^2`` so closer (and therefore
        tighter-pose) tags dominate. ``None`` (default) means every
        pair contributes equally, which is the original behaviour.

    Returns
    -------
    np.ndarray or None
        The single ``T_cam_to_base`` that best satisfies
        ``T_cam_to_tag_i ~= T_cam_to_base @ T_base_to_tag_i`` for every
        pair, in the weighted least-squares sense. ``None`` if fewer
        than 3 pairs are supplied (3 non-collinear points are needed
        for a unique SE(3) solution).

    Notes
    -----
    This is the classic Umeyama / ArUco ``estimatePoseBoard``
    formulation: point correspondences are the tag centers in the base
    and camera frames, and we solve for the rigid transform that aligns
    them. Doing it jointly over all detected tags is essential when the
    tags sit at different orientations on the base link, because
    averaging per-tag inverse solutions would mix inconsistent
    orientation estimates.

    Weighting is applied by scaling the centred points by ``sqrt(w)``
    before the SVD. Mathematically equivalent to minimising
    ``sum_i w_i * || Q_c_i - R P_c_i ||^2``.
    """
    if len(pairs) < 3:
        return None

    P = np.stack([T_b[:3, 3] for _, T_b in pairs], axis=0)  # base-frame points
    Q = np.stack([T_c[:3, 3] for T_c, _ in pairs], axis=0)  # cam-frame points
    p_bar = P.mean(axis=0)
    q_bar = Q.mean(axis=0)
    Pc = P - p_bar
    Qc = Q - q_bar

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != len(pairs):
            raise ValueError(
                f"weights has {w.shape[0]} entries, expected {len(pairs)}"
            )
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        s = np.sqrt(w).reshape(-1, 1)
        Pc = Pc * s
        Qc = Qc * s

    # Covariance H: minimizes ||Q_c - R P_c||_F^2 over R in SO(3).
    H = Qc.T @ Pc
    U, _, Vt = np.linalg.svd(H)
    R_sol = U @ Vt
    if np.linalg.det(R_sol) < 0.0:
        Vt[-1, :] *= -1.0
        R_sol = U @ Vt
    t_sol = q_bar - R_sol @ p_bar

    T = np.eye(4)
    T[:3, :3] = R_sol
    T[:3, 3] = t_sol
    return T


def solve_cam_to_base_ransac(
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    weights: Optional[np.ndarray] = None,
    *,
    n_iter: int = 30,
    residual_thresh: float = 0.05,
    rng_seed: Optional[int] = None,
) -> Optional[np.ndarray]:
    """RANSAC wrapper around :func:`solve_cam_to_base`.

    Draws ``n_iter`` random 3-pair subsets, solves each, scores by the
    number of inliers (pairs whose reprojection error under the solved
    ``T_cam_to_base`` is below ``residual_thresh``), and refits using
    all inliers of the best iteration. Falls back to the plain
    (non-RANSAC) solve when no iteration finds >= 3 inliers.

    Parameters
    ----------
    pairs
        List of ``(T_cam_to_tag_observed, T_base_to_tag_known)`` 4x4
        transforms.
    weights
        Optional per-pair weight forwarded to the inner solver (and
        used to weight the inlier count). ``None`` is uniform weight.
    n_iter
        Maximum number of random subsets to try. With >= 5 pairs the
        probability of an all-inlier subset in ``n_iter`` draws is
        ``1 - (1 - outlier_free_fraction^3) ** n_iter``; 30 iterations
        at 80% inlier rate gives ~99.9% success.
    residual_thresh
        Per-pair residual threshold, in metres + radians (the test is
        ``||t_residual|| + ||R_residual_as_angle|| < residual_thresh``).
        0.05 is a sensible default for camera-to-tag observations at
        0.5-3 m working distance.
    rng_seed
        Optional integer seed for the internal ``numpy`` RNG. Useful in
        unit tests where the result must be deterministic. ``None``
        means a fresh non-deterministic stream.

    Returns
    -------
    np.ndarray or None
        The best ``T_cam_to_base`` found, or ``None`` if there are
        fewer than 3 pairs (RANSAC needs at least 3 to fit + 1 to
        score -- the caller's responsibility is to gate on pair count).
    """
    n = len(pairs)
    if n < 3:
        return None
    rng = np.random.default_rng(rng_seed)

    weights_arr = (
        None
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )

    best_inliers: List[int] = []
    best_T: Optional[np.ndarray] = None

    for _ in range(n_iter):
        # Random 3-pair subset (distinct indices).
        subset_idx = rng.choice(n, size=3, replace=False)
        subset = [pairs[i] for i in subset_idx]
        subset_w = None if weights_arr is None else weights_arr[subset_idx]
        T = solve_cam_to_base(subset, weights=subset_w)
        if T is None:
            continue  # degenerate subset, try again

        # Score against every other pair. A pair (Tc, Tb) is an inlier
        # if Tc ≈ T @ Tb, where "≈" means both the translation
        # residual and the rotation-angle residual are below the
        # threshold.
        inliers: List[int] = []
        for j in range(n):
            Tc, Tb = pairs[j]
            t_pred = T[:3, :3] @ Tb[:3, 3] + T[:3, 3]
            t_res = float(np.linalg.norm(Tc[:3, 3] - t_pred))
            R_res = Tc[:3, :3] @ Tb[:3, :3].T @ T[:3, :3].T
            # Angle from the rotation matrix via the trace formula:
            # angle = arccos((tr(R) - 1) / 2), clipped for safety.
            cos_angle = (np.trace(R_res) - 1.0) * 0.5
            cos_angle = max(-1.0, min(1.0, cos_angle))
            r_res = float(np.arccos(cos_angle))
            if (t_res + r_res) < residual_thresh:
                inliers.append(j)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_T = T

    # Refit on the best inlier set, or fall back to the plain solve
    # if RANSAC didn't find a clean consensus.
    if best_T is not None and len(best_inliers) >= 3:
        inlier_pairs = [pairs[i] for i in best_inliers]
        inlier_w = (
            None
            if weights_arr is None
            else weights_arr[np.asarray(best_inliers, dtype=np.int64)]
        )
        return solve_cam_to_base(inlier_pairs, weights=inlier_w)
    return solve_cam_to_base(pairs, weights=weights)


def resolve_urdf_path(obj_cfg: dict) -> str:
    """Resolve a URDF path from ``object.urdf_package`` + ``object.urdf_filename``.

    Equivalent to ``description.launch.py``'s ``get_package_share_directory``
    pattern. Returns an empty string if the package name is missing.
    """
    pkg = obj_cfg.get("urdf_package") or obj_cfg.get("urdf_path_pkg")
    rel = obj_cfg.get("urdf_filename") or obj_cfg.get("urdf_path_in_pkg")
    if not pkg:
        return ""
    if not rel:
        rel = "urdf/base.urdf"
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory(str(pkg))
    except Exception as e:
        raise RuntimeError(
            f"Could not resolve ROS package '{pkg}' for URDF lookup: {e}"
        )
    return str(Path(share) / rel)


def load_yaml_config(path: str) -> dict:
    """Load the YAML config and return the inner ``apriltag:`` block.

    Falls back to a top-level dict if no ``apriltag:`` key is present.
    Returns an empty dict if ``path`` is empty or the file is missing.
    """
    if not path:
        return {}
    p = Path(str(path)).expanduser()
    if not p.is_file():
        return {}
    with open(p, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("apriltag", cfg) or {}