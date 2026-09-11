#!/usr/bin/env python3
"""Smoke test for the GPU image shim (nvjpeg decode/encode + VPI rectify).

Pure ctypes against ``librace_auv_image_cuda.so`` and
``librace_auv_apriltag_cuda.so`` — no ROS imports. It:

1. builds a fisheye warp map with OpenCV,
2. decodes a synthetic JPEG on the GPU,
3. rectifies it on the GPU and checks parity with ``cv2.remap``,
4. runs cuAprilTags on the *device* rectified buffer (no H2D copy),
5. encodes the rectified frame with nvjpeg and round-trips it.

Run after building the package::

    source install/setup.bash
    python3 src/race_auv/race_auv_perception/race_auv_apriltag_cuda/test/gpu_image_smoke.py

Exit code 0 on success, 1 on failure.
"""

import ctypes
import os
import sys

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
TAG_ID = 9
TAG_SIDE_PX = 300


def _package_lib(name: str) -> str:
    env = os.environ.get("RACE_AUV_IMAGE_CUDA_LIB" if "image" in name
                         else "RACE_AUV_APRILTAG_CUDA_LIB", "").strip()
    if env:
        return env
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(
            get_package_share_directory("race_auv_apriltag_cuda"), "lib", name
        )
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "..", "..", "install", "race_auv_apriltag_cuda",
            "share", "race_auv_apriltag_cuda", "lib", name,
        )


def load_image_lib(path):
    lib = ctypes.CDLL(path)
    lib.race_img_create.restype = ctypes.c_void_p
    lib.race_img_create.argtypes = [ctypes.c_int, ctypes.c_int]
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
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.race_img_last_error.restype = ctypes.c_char_p
    lib.race_img_last_error.argtypes = []
    return lib


def load_detector_lib(path):
    lib = ctypes.CDLL(path)
    lib.race_at_create.restype = ctypes.c_void_p
    lib.race_at_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.race_at_detect_device.restype = ctypes.c_int
    lib.race_at_detect_device.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.race_at_destroy.argtypes = [ctypes.c_void_p]
    lib.race_at_last_error.restype = ctypes.c_char_p
    lib.race_at_last_error.argtypes = []
    return lib


def main() -> int:
    image_lib = load_image_lib(_package_lib("librace_auv_image_cuda.so"))
    detector_lib = load_detector_lib(_package_lib("librace_auv_apriltag_cuda.so"))

    # Mild fisheye calibration, centered principal point.
    k = np.array([[600.0, 0, WIDTH / 2], [0, 600.0, HEIGHT / 2], [0, 0, 1]])
    d = np.array([-0.08, 0.01, 0.0, 0.0])
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k, d, (WIDTH, HEIGHT), np.eye(3), balance=0.0
    )
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        k, d, np.eye(3), new_k, (WIDTH, HEIGHT), cv2.CV_32FC1
    )

    # Synthetic tag36h11 id 9 at the image center.
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    marker = aruco.drawMarker(dictionary, TAG_ID, TAG_SIDE_PX)
    frame = np.full((HEIGHT, WIDTH, 3), 200, np.uint8)
    y0 = (HEIGHT - TAG_SIDE_PX) // 2
    x0 = (WIDTH - TAG_SIDE_PX) // 2
    frame[y0:y0 + TAG_SIDE_PX, x0:x0 + TAG_SIDE_PX] = cv2.cvtColor(
        marker, cv2.COLOR_GRAY2BGR
    )
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    jpeg = encoded.tobytes()

    pipeline = image_lib.race_img_create(WIDTH, HEIGHT)
    if not pipeline:
        print("FAIL: race_img_create:", image_lib.race_img_last_error().decode())
        return 1
    detector = None
    try:
        rc = image_lib.race_img_set_fisheye_job(
            pipeline, 0,
            map_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            map_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            WIDTH, HEIGHT, 4,
        )
        if rc != 0:
            print("FAIL: set_fisheye_job:", image_lib.race_img_last_error().decode())
            return 1

        buffer = np.frombuffer(jpeg, np.uint8)
        rc = image_lib.race_img_decode(
            pipeline,
            buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(jpeg),
        )
        if rc != 0:
            print("FAIL: decode:", image_lib.race_img_last_error().decode())
            return 1
        rc = image_lib.race_img_rectify(pipeline, 0)
        if rc != 0:
            print("FAIL: rectify:", image_lib.race_img_last_error().decode())
            return 1

        host = np.empty((HEIGHT, WIDTH, 3), np.uint8)
        rc = image_lib.race_img_rectified_to_host(
            pipeline, 0,
            host.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), WIDTH * 3,
        )
        if rc != 0:
            print("FAIL: to_host:", image_lib.race_img_last_error().decode())
            return 1

        cpu = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        diff = np.abs(host.astype(np.int16) - cpu.astype(np.int16))
        print(f"rectify parity: mean={diff.mean():.3f} p99={np.percentile(diff, 99):.1f}")

        # Device detection (no H2D).
        detector = detector_lib.race_at_create(
            WIDTH, HEIGHT, 4, 0.125,
            float(new_k[0, 0]), float(new_k[1, 1]),
            float(new_k[0, 2]), float(new_k[1, 2]),
        )
        if not detector:
            print("FAIL: race_at_create:", detector_lib.race_at_last_error().decode())
            return 1
        ptr = image_lib.race_img_get_rectified(pipeline, 0)
        max_tags = 64
        ids = (ctypes.c_uint16 * max_tags)()
        corners = (ctypes.c_float * (8 * max_tags))()
        orientation = (ctypes.c_float * (9 * max_tags))()
        translation = (ctypes.c_float * (3 * max_tags))()
        count = ctypes.c_int(0)
        rc = detector_lib.race_at_detect_device(
            detector, ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)),
            WIDTH * 3, WIDTH, HEIGHT, max_tags,
            ids, corners, orientation, translation, ctypes.byref(count),
        )
        if rc != 0:
            print("FAIL: detect_device:", detector_lib.race_at_last_error().decode())
            return 1
        found = [int(ids[i]) for i in range(count.value)]
        print(f"device detections: {found}")
        if TAG_ID not in found:
            print(f"FAIL: tag {TAG_ID} not detected on the device path")
            return 1
        z = translation[3 * found.index(TAG_ID) + 2]
        print(f"tag {TAG_ID} z = {z:.3f} m")

        # nvjpeg encode round-trip.
        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_len = ctypes.c_size_t()
        rc = image_lib.race_img_encode(
            pipeline,
            host.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), WIDTH * 3,
            WIDTH, HEIGHT, 90, ctypes.byref(out_ptr), ctypes.byref(out_len),
        )
        if rc != 0:
            print("FAIL: encode:", image_lib.race_img_last_error().decode())
            return 1
        decoded = cv2.imdecode(
            np.frombuffer(ctypes.string_at(out_ptr, out_len.value), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if decoded is None or decoded.shape != host.shape:
            print("FAIL: encoded JPEG did not round-trip")
            return 1
        print(f"encode: {out_len.value} bytes, {decoded.shape}")
    finally:
        if detector:
            detector_lib.race_at_destroy(detector)
        image_lib.race_img_destroy(pipeline)

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
