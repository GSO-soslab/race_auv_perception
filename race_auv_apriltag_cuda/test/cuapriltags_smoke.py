#!/usr/bin/env python3
"""Smoke test for the cuAprilTags ctypes shim.

Renders a synthetic tag36h11 with OpenCV's ArUco module, runs it through
race_at_create/race_at_detect, and checks the decoded id and pose against
the expected values for a pinhole camera. This validates that the
prebuilt libcuapriltags.a (built for JetPack 6.1 / CUDA 12) links and
runs against this host's CUDA runtime.

Run after building the package:

    source install/setup.bash
    python3 src/race_auv/race_auv_perception/race_auv_apriltag_cuda/test/cuapriltags_smoke.py

Exit code 0 on success, 1 on failure.
"""

import argparse
import ctypes
import math
import os
import sys

import numpy as np


TAG_ID = 0
TAG_SIZE_M = 0.125
SIDE_PIXELS = 200
MARGIN_PIXELS = 80
FX = FY = 800.0
MAX_TAGS = 64


def default_library_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(
            get_package_share_directory("race_auv_apriltag_cuda"),
            "lib",
            "librace_auv_apriltag_cuda.so",
        )
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..",
            "..",
            "..",
            "install",
            "race_auv_apriltag_cuda",
            "share",
            "race_auv_apriltag_cuda",
            "lib",
            "librace_auv_apriltag_cuda.so",
        )


def load_shim(path: str):
    lib = ctypes.CDLL(path)
    lib.race_at_create.restype = ctypes.c_void_p
    lib.race_at_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
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
    lib.race_at_destroy.restype = None
    lib.race_at_destroy.argtypes = [ctypes.c_void_p]
    lib.race_at_last_error.restype = ctypes.c_char_p
    lib.race_at_last_error.argtypes = []
    return lib


def render_tag_bgr(tag_id: int, side_pixels: int, margin_pixels: int) -> np.ndarray:
    import cv2

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, tag_id, side_pixels)
    else:  # OpenCV < 4.7
        marker = aruco.drawMarker(dictionary, tag_id, side_pixels)

    side = side_pixels + 2 * margin_pixels
    canvas = np.full((side, side), 255, dtype=np.uint8)
    canvas[
        margin_pixels:margin_pixels + side_pixels,
        margin_pixels:margin_pixels + side_pixels,
    ] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def detect(lib, handle, bgr: np.ndarray):
    bgr = np.ascontiguousarray(bgr)
    height, width = bgr.shape[:2]
    ids = (ctypes.c_uint16 * MAX_TAGS)()
    corners = (ctypes.c_float * (8 * MAX_TAGS))()
    orientation = (ctypes.c_float * (9 * MAX_TAGS))()
    translation = (ctypes.c_float * (3 * MAX_TAGS))()
    count = ctypes.c_int(0)
    status = lib.race_at_detect(
        handle,
        bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        bgr.strides[0],
        width, height, MAX_TAGS,
        ids, corners, orientation, translation, ctypes.byref(count),
    )
    if status != 0:
        raise RuntimeError(
            f"race_at_detect failed: {lib.race_at_last_error().decode()}"
        )
    detections = []
    for i in range(count.value):
        R = np.array(
            [orientation[i * 9 + k] for k in range(9)], dtype=np.float64
        ).reshape(3, 3, order="F")
        t = np.array(
            [translation[i * 3 + k] for k in range(3)], dtype=np.float64
        )
        detections.append({"id": int(ids[i]), "R": R, "t": t})
    return detections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", default=default_library_path())
    args = parser.parse_args()

    if not os.path.isfile(args.lib):
        print(f"FAIL: shim not found at {args.lib}")
        return 1

    import cv2
    print(f"OpenCV {cv2.__version__} from {cv2.__file__}")
    print(f"shim: {args.lib}")

    lib = load_shim(args.lib)
    image = render_tag_bgr(TAG_ID, SIDE_PIXELS, MARGIN_PIXELS)
    height, width = image.shape[:2]
    expected_z = FX * TAG_SIZE_M / SIDE_PIXELS
    print(f"image {width}x{height}, expected z ~ {expected_z:.3f} m")

    handle = lib.race_at_create(
        width, height, 4, TAG_SIZE_M, FX, FY, width / 2.0, height / 2.0
    )
    if not handle:
        print(f"FAIL: race_at_create: {lib.race_at_last_error().decode()}")
        return 1

    try:
        detections = detect(lib, handle, image)
        if not detections:
            print("FAIL: no detections in synthetic tag image")
            return 1
        det = next((d for d in detections if d["id"] == TAG_ID), None)
        if det is None:
            found = [d["id"] for d in detections]
            print(f"FAIL: tag id {TAG_ID} not found (got {found})")
            return 1
        z = det["t"][2]
        det_R = float(np.linalg.det(det["R"]))
        print(f"detected id={det['id']} t={det['t']} det(R)={det_R:.4f}")
        if not (0.5 * expected_z < z < 1.5 * expected_z):
            print(f"FAIL: z={z:.3f} outside 0.5..1.5x expected {expected_z:.3f}")
            return 1
        if not math.isfinite(det_R) or abs(det_R - 1.0) > 1e-2:
            print(f"FAIL: rotation is not orthonormal (det={det_R})")
            return 1

        blank = np.full((height, width, 3), 255, dtype=np.uint8)
        blank_detections = detect(lib, handle, blank)
        if blank_detections:
            print(f"FAIL: detected {len(blank_detections)} tags in blank image")
            return 1
        print("blank image -> 0 detections (ok)")
    finally:
        lib.race_at_destroy(handle)

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
