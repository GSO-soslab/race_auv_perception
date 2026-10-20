"""Tests on rendered tag36h11 images: corner order, multi-size poses, image checks, rectifier.

The tags are rendered with OpenCV's ``DICT_APRILTAG_36h11`` (the same codes
as apriltag3) and warped into a synthetic pinhole image at a known pose
``T_cam_tag``. The marker image is exactly the tag's black square.

Orientation: the stack's tag frame is the one apriltag3 reports (its
``lb-rb-rt-lt`` corners, which the old race_auv_camera_pkg python backend
published too and the station URDF tag frames follow). OpenCV renders the
same code rotated by 180 deg relative to that frame, so the renderer turns
the marker image by 180 deg first. Then its image corners map to the
canonical corners as

    marker image (after the turn)   top-left  top-right  bottom-right  bottom-left
    tag frame                       lt        rt         rb            lb

If apriltag3 ever changed its corner convention, every corner test fails.
"""

import math

import cv2
import numpy as np
import pytest

from race_auv_camera_tag_factorgraph.rectify import Rectifier
from race_auv_camera_tag_factorgraph.tags import TagDetector, ippe_pose, tag_corners

W, H = 1600, 1200
K = np.array([[1000.0, 0, 800.0], [0, 1000.0, 600.0], [0, 0, 1]])


def pose(xyz, rpy_deg):
    """4x4 T_cam_tag from a translation and fixed-axis roll/pitch/yaw (degrees)."""
    r, p, y = (math.radians(a) for a in rpy_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def project(T, size):
    """Canonical corners of a tag at T_cam_tag, projected with K. (4, 2)."""
    p = tag_corners(size) @ T[:3, :3].T + T[:3, 3]
    return (p[:, :2] / p[:, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def render(tags):
    """White image with each (tag_id, size, T_cam_tag) warped in at its pose."""
    img = np.full((H, W), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for tag_id, size, T in tags:
        n = 400
        # 180 deg turn: OpenCV's image -> apriltag3's tag frame (module docstring).
        # generateImageMarker is OpenCV >= 4.7; the Jetson's system OpenCV 4.6
        # still calls it drawMarker.
        gen = getattr(cv2.aruco, "generateImageMarker", None) or cv2.aruco.drawMarker
        marker = np.rot90(gen(dictionary, tag_id, n), 2).copy()
        uv = project(T, size)                      # lb, rb, rt, lt
        src = np.float32([[0, n], [n, n], [n, 0], [0, 0]])   # marker lb, rb, rt, lt
        Hm = cv2.getPerspectiveTransform(src, np.float32(uv))
        warped = cv2.warpPerspective(marker, Hm, (W, H), flags=cv2.INTER_AREA,
                                     borderValue=255)
        mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), Hm, (W, H)) > 0
        img[mask] = warped[mask]
    return img


def make_detector(sizes, **kw):
    return TagDetector(sizes, K, W, H, threads=2, decimate=1.0, **kw)


def pose_error(T_est, T_true):
    dt = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
    dR = T_est[:3, :3].T @ T_true[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
    return dt, ang


@pytest.mark.parametrize("rpy", [(0, 0, 0), (0, 0, 35), (20, -25, 100), (-30, 15, -150)])
def test_corners_are_canonical_and_pose_matches(rpy):
    """Detected corners land on the projected canonical corners, in that order."""
    T = pose([0.1, -0.05, 1.5], rpy)
    det = make_detector({146: 0.12})
    res = det.detect(render([(146, 0.12, T)]))
    assert [t.tag_id for t in res.tags] == [146] and not res.rejected
    t = res.tags[0]
    assert np.max(np.linalg.norm(t.corners - project(T, 0.12), axis=1)) < 1.0
    dt, ang = pose_error(t.T_cam_tag, T)
    assert dt < 0.01 and ang < 2.0
    assert t.fit_rms_px < 0.5


def test_two_sizes_in_one_image():
    """Per-ID size lookup: a 12.5 cm and a 4 cm tag, both posed with their own size.

    The range is the proof: PnP range scales with the assumed size, so a
    wrong size would be off by a factor of ~3 here. The rotation is checked
    for the big tag only. The 4 cm tag spans ~34 px, where the two IPPE
    solutions fit equally well (flip ambiguity) and the best-fitting one can
    be the flipped pose; dock_fuser keeps both solutions for that reason.
    """
    T_big = pose([-0.25, 0.0, 1.6], (10, 0, 0))
    T_small = pose([0.2, 0.05, 1.2], (0, 15, 20))
    det = make_detector({9: 0.125, 541: 0.04})
    res = det.detect(render([(9, 0.125, T_big), (541, 0.04, T_small)]))
    got = {t.tag_id: t for t in res.tags}
    assert set(got) == {9, 541}
    for tag_id, T in ((9, T_big), (541, T_small)):
        dt, ang = pose_error(got[tag_id].T_cam_tag, T)
        assert dt < 0.01, (tag_id, dt)
        if tag_id == 9:
            assert ang < 3.0, ang


def test_unknown_id_is_ignored():
    det = make_detector({146: 0.12})
    res = det.detect(render([(22, 0.12, pose([0, 0, 1.5], (0, 0, 0)))]))
    assert not res.tags and not res.rejected


def test_border_and_size_checks():
    det = make_detector({146: 0.12, 541: 0.04}, min_side_px=30.0)
    # Tag 146 with one corner 4 px from the left border; 541 far away (small).
    T_edge = pose([0, 0, 1.0], (0, 0, 0))
    T_edge[0, 3] = (4 + 60 - K[0, 2]) / K[0, 0] * 1.0   # tag side 120 px -> left edge at u = 4
    T_far = pose([0.3, 0.2, 2.0], (0, 0, 0))           # 4 cm at 2 m: side 20 px
    res = det.detect(render([(146, 0.12, T_edge), (541, 0.04, T_far)]))
    reasons = dict(res.rejected)
    assert not res.tags
    assert "border" in reasons[146] and "side" in reasons[541]


def test_fit_rms_grows_with_a_bad_corner():
    """The fit check catches a gross single-corner error, not a small one."""
    T = pose([0, 0, 1.5], (15, 10, 30))
    uv = project(T, 0.12)
    _, clean = ippe_pose(uv, 0.12, K)
    bad = uv.copy()
    bad[2] += (12.0, -9.0)                              # one corner 15 px off
    _, gross = ippe_pose(bad, 0.12, K)
    assert clean < 0.01 and gross > 1.5


def test_rectifier_matches_tank_calibration():
    """Same rectified K as race_auv_camera_pkg's rectifier (what dock_fuser's tank config uses).

    Tolerance 0.1 px: the fisheye new-camera-matrix estimate differs by ~0.04 px
    between OpenCV versions (4.11 on the desktop, 4.6 on the Jetson). Harmless
    at runtime: the detector publishes the K it actually rectified with in its
    camera_info, and dock_fuser takes K from there.
    """
    Kraw = np.array([[1262.74163, 0, 801.604094], [0, 1262.40141, 670.832446], [0, 0, 1]])
    D = [-0.0011265, -0.1213276, 0.444942, -0.388063]
    r = Rectifier(Kraw, D, (1600, 1200), fisheye=True)
    assert (r.width, r.height) == (1600, 1200)
    np.testing.assert_allclose([r.K[0, 0], r.K[1, 1], r.K[0, 2], r.K[1, 2]],
                               [1165.2372, 1164.9232, 801.9339, 679.6697], atol=0.1)


def test_rectifier_maps_to_the_distorted_pixel():
    """A rectified pixel samples the raw pixel that the fisheye model projects it to."""
    Kraw = np.array([[1262.74163, 0, 801.604094], [0, 1262.40141, 670.832446], [0, 0, 1]])
    D = np.array([-0.0011265, -0.1213276, 0.444942, -0.388063])
    r = Rectifier(Kraw, D, (1600, 1200), fisheye=True)
    mx, my = cv2.convertMaps(r.map1, r.map2, cv2.CV_32FC1)
    x0, y0 = r.roi[:2]
    for u, v in [(100, 100), (800, 600), (1500, 1100), (300, 900)]:
        n = np.array([[[(u - r.K[0, 2]) / r.K[0, 0], (v - r.K[1, 2]) / r.K[1, 1]]]])
        raw = cv2.fisheye.distortPoints(n, Kraw, D).ravel()
        assert np.hypot(mx[v + y0, u + x0] - raw[0], my[v + y0, u + x0] - raw[1]) < 0.05
