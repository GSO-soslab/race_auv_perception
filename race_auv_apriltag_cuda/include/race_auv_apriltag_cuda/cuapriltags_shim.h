// SPDX-License-Identifier: Apache-2.0
//
// C ABI shim around NVIDIA's cuAprilTags detector.
//
// ctypes cannot link against a static library (libcuapriltags.a), so this
// shim is built as a shared object and loaded by
// race_auv_camera_pkg/apriltag_cuda.py. It also keeps NVIDIA's struct
// padding (e.g. sizeof(cuAprilTagsID_t) == 88) out of the Python side:
// every output is flattened into caller-provided primitive arrays.
//
// Only NVAT_TAG36H11 is supported by the underlying library.

#ifndef RACE_AUV_APRILTAG_CUDA_CUAPRILTAGS_SHIM_H
#define RACE_AUV_APRILTAG_CUDA_CUAPRILTAGS_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Create a detector for a fixed image size and camera model.
//
// width/height   image dimensions in pixels (must match every race_at_detect call)
// tile_size      adaptive-threshold window size (Isaac ROS uses 4)
// tag_size       physical tag side length, metres; translations are returned
//                in the same units
// fx/fy/cx/cy    pinhole intrinsics of the (undistorted) input image
//
// Returns an opaque handle, or NULL on failure (see race_at_last_error()).
void* race_at_create(int width, int height, int tile_size, float tag_size,
                     float fx, float fy, float cx, float cy);

// Run detection on one host BGR (uchar3) image.
//
// host_bgr must point to width*height*3 bytes with the given row pitch.
// Output arrays must hold at least max_tags entries:
//   out_ids         [max_tags]
//   out_corners     [8 * max_tags]  x,y per corner (4 corners)
//   out_orientation [9 * max_tags]  column-major 3x3 rotation
//   out_translation [3 * max_tags]  camera-frame translation
//   out_count       [1]             number of detections written
//
// Returns 0 on success (including zero detections), non-zero on failure.
int race_at_detect(void* handle, const uint8_t* host_bgr, size_t pitch,
                   int width, int height, int max_tags,
                   uint16_t* out_ids,
                   float* out_corners,
                   float* out_orientation,
                   float* out_translation,
                   int* out_count);

// Same as race_at_detect(), but the input already lives in device memory
// (e.g. the output of race_img_get_rectified). No H2D copy is made.
int race_at_detect_device(void* handle, const uint8_t* device_bgr, size_t pitch,
                          int width, int height, int max_tags,
                          uint16_t* out_ids,
                          float* out_corners,
                          float* out_orientation,
                          float* out_translation,
                          int* out_count);

// Destroy a detector created by race_at_create(). NULL is a no-op.
void race_at_destroy(void* handle);

// Last error message for the calling thread ("" when none). The returned
// pointer is valid until the next call on the same thread.
const char* race_at_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  // RACE_AUV_APRILTAG_CUDA_CUAPRILTAGS_SHIM_H
