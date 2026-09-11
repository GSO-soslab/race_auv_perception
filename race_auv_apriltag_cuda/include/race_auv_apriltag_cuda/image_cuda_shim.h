// SPDX-License-Identifier: Apache-2.0
//
// In-process GPU image stages (nvjpeg decode/encode + VPI CUDA rectify).
// See src/image_cuda_shim.cpp for the pipeline description.

#ifndef RACE_AUV_APRILTAG_CUDA_IMAGE_CUDA_SHIM_H
#define RACE_AUV_APRILTAG_CUDA_IMAGE_CUDA_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Create a pipeline with a fixed decoded-image size.
void* race_img_create(int width, int height);

// Destroy a pipeline. NULL is a no-op. Safe to call twice.
void race_img_destroy(void* handle);

// Configure a fisheye rectification job (0..1).
//
// map_x/map_y are dense float32 maps of size out_width*out_height in the
// same convention as cv2.fisheye.initUndistortRectifyMap: map_x[y, x]
// gives the source x for output pixel (x, y). interval is the VPI warp
// grid spacing (power of two; 4 is a good default).
int race_img_set_fisheye_job(void* handle, int job_index,
    const float* map_x, const float* map_y,
    int out_width, int out_height, int interval);

// Decode a JPEG into the pipeline's device buffer (bgri, width*3 pitch).
int race_img_decode(void* handle, const uint8_t* jpeg, size_t length);

// Run the configured rectification job on the last decoded image.
int race_img_rectify(void* handle, int job_index);

// Device pointer to the decoded BGR image (race_at_detect_device can use it).
const void* race_img_get_decoded(void* handle);

// Device pointer to a rectified BGR image.
const void* race_img_get_rectified(void* handle, int job_index);

// Size of a configured rectification job's output.
int race_img_get_rectified_size(void* handle, int job_index,
    int* width, int* height);

// Copy a rectified image back to host memory (for annotation / display).
int race_img_rectified_to_host(void* handle, int job_index, uint8_t* host,
    size_t pitch);

// Encode a host BGR image as JPEG (quality 0..100). The returned pointer
// is owned by the pipeline and valid until the next call.
int race_img_encode(void* handle, const uint8_t* host_bgr, size_t pitch,
    int width, int height, int quality, const uint8_t** out_data,
    size_t* out_length);

// Last error message for the calling thread ("" when none).
const char* race_img_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  // RACE_AUV_APRILTAG_CUDA_IMAGE_CUDA_SHIM_H
