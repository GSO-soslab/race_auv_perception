// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the cuAprilTags ctypes shim. See cuapriltags_shim.h
// for the contract and race_auv_camera_pkg/apriltag_cuda.py for the
// Python consumer.

#include "race_auv_apriltag_cuda/cuapriltags_shim.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "cuAprilTags.h"

// The Python side never sees this struct, but ABI drift between the
// header and the prebuilt library would silently corrupt every output
// array, so fail the build instead.
static_assert(sizeof(cuAprilTagsID_t) == 88,
    "cuAprilTagsID_t layout changed; update the shim's flattening logic.");

namespace {

thread_local std::string g_last_error;

void clear_error()
{
    g_last_error.clear();
}

void set_error(const std::string& message)
{
    g_last_error = message;
}

struct ShimHandle
{
    cuAprilTagsHandle detector = nullptr;
    uint8_t* device_buffer = nullptr;
    std::size_t device_buffer_bytes = 0;
    int width = 0;
    int height = 0;
    std::vector<cuAprilTagsID_t> tags;
};

int detect_from_device(ShimHandle* handle, const uint8_t* device_bgr,
    std::size_t pitch, int width, int height, int max_tags,
    uint16_t* out_ids, float* out_corners, float* out_orientation,
    float* out_translation, int* out_count)
{
    if (handle->tags.size() < static_cast<std::size_t>(max_tags)) {
        handle->tags.resize(static_cast<std::size_t>(max_tags));
    }

    cuAprilTagsImageInput_t input;
    input.dev_ptr = reinterpret_cast<uchar3*>(const_cast<uint8_t*>(device_bgr));
    input.pitch = pitch;
    input.width = static_cast<uint16_t>(width);
    input.height = static_cast<uint16_t>(height);

    uint32_t num_tags = 0;
    const int status = cuAprilTagsDetect(
        handle->detector, &input, handle->tags.data(), &num_tags,
        static_cast<uint32_t>(max_tags), nullptr);
    if (status != 0) {
        set_error("cuAprilTagsDetect failed (status " + std::to_string(status) + ")");
        return -1;
    }

    for (uint32_t i = 0; i < num_tags; ++i) {
        const cuAprilTagsID_t& tag = handle->tags[i];
        out_ids[i] = tag.id;
        for (int corner = 0; corner < 4; ++corner) {
            out_corners[i * 8 + corner * 2 + 0] = tag.corners[corner].x;
            out_corners[i * 8 + corner * 2 + 1] = tag.corners[corner].y;
        }
        for (int k = 0; k < 9; ++k) {
            out_orientation[i * 9 + k] = tag.orientation[k];
        }
        for (int k = 0; k < 3; ++k) {
            out_translation[i * 3 + k] = tag.translation[k];
        }
    }
    *out_count = static_cast<int>(num_tags);
    return 0;
}

}  // namespace

extern "C" {

void* race_at_create(int width, int height, int tile_size, float tag_size,
    float fx, float fy, float cx, float cy)
{
    clear_error();
    if (width <= 0 || height <= 0 || tile_size <= 0 || tag_size <= 0.0f) {
        set_error("race_at_create: width/height/tile_size/tag_size must be positive");
        return nullptr;
    }

    ShimHandle* handle = new (std::nothrow) ShimHandle();
    if (handle == nullptr) {
        set_error("race_at_create: out of memory");
        return nullptr;
    }
    handle->width = width;
    handle->height = height;
    handle->device_buffer_bytes =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u;

    cudaError_t cuda_status = cudaMalloc(
        reinterpret_cast<void**>(&handle->device_buffer), handle->device_buffer_bytes);
    if (cuda_status != cudaSuccess) {
        set_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(cuda_status));
        delete handle;
        return nullptr;
    }

    const cuAprilTagsCameraIntrinsics_t camera{fx, fy, cx, cy};
    const int status = nvCreateAprilTagsDetector(
        &handle->detector,
        static_cast<uint32_t>(width), static_cast<uint32_t>(height),
        static_cast<uint32_t>(tile_size),
        NVAT_TAG36H11, &camera, tag_size);
    if (status != 0 || handle->detector == nullptr) {
        set_error("nvCreateAprilTagsDetector failed (status " +
                  std::to_string(status) + ")");
        cudaFree(handle->device_buffer);
        delete handle;
        return nullptr;
    }

    return handle;
}

int race_at_detect(void* raw_handle, const uint8_t* host_bgr, size_t pitch,
    int width, int height, int max_tags,
    uint16_t* out_ids,
    float* out_corners,
    float* out_orientation,
    float* out_translation,
    int* out_count)
{
    clear_error();
    if (raw_handle == nullptr) {
        set_error("race_at_detect: null handle");
        return -1;
    }
    if (host_bgr == nullptr || out_ids == nullptr || out_corners == nullptr ||
        out_orientation == nullptr || out_translation == nullptr ||
        out_count == nullptr)
    {
        set_error("race_at_detect: null input/output pointer");
        return -1;
    }
    if (max_tags <= 0) {
        set_error("race_at_detect: max_tags must be positive");
        return -1;
    }

    ShimHandle* handle = static_cast<ShimHandle*>(raw_handle);
    if (width != handle->width || height != handle->height) {
        set_error("race_at_detect: image size differs from the detector's "
                  "creation size");
        return -1;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3u;
    if (pitch < row_bytes) {
        set_error("race_at_detect: pitch smaller than width*3");
        return -1;
    }

    cudaError_t cuda_status = cudaMemcpy2D(
        handle->device_buffer, row_bytes,
        host_bgr, pitch,
        row_bytes, static_cast<std::size_t>(height),
        cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        set_error(std::string("cudaMemcpy2D failed: ") + cudaGetErrorString(cuda_status));
        return -1;
    }

    return detect_from_device(
        handle, handle->device_buffer, row_bytes, width, height, max_tags,
        out_ids, out_corners, out_orientation, out_translation, out_count);
}

int race_at_detect_device(void* raw_handle, const uint8_t* device_bgr,
    size_t pitch, int width, int height, int max_tags,
    uint16_t* out_ids, float* out_corners, float* out_orientation,
    float* out_translation, int* out_count)
{
    clear_error();
    if (raw_handle == nullptr) {
        set_error("race_at_detect_device: null handle");
        return -1;
    }
    if (device_bgr == nullptr || out_ids == nullptr || out_corners == nullptr ||
        out_orientation == nullptr || out_translation == nullptr ||
        out_count == nullptr)
    {
        set_error("race_at_detect_device: null input/output pointer");
        return -1;
    }
    if (max_tags <= 0) {
        set_error("race_at_detect_device: max_tags must be positive");
        return -1;
    }

    ShimHandle* handle = static_cast<ShimHandle*>(raw_handle);
    if (width != handle->width || height != handle->height) {
        set_error("race_at_detect_device: image size differs from the detector's "
                  "creation size");
        return -1;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3u;
    if (pitch < row_bytes) {
        set_error("race_at_detect_device: pitch smaller than width*3");
        return -1;
    }

    return detect_from_device(
        handle, device_bgr, pitch, width, height, max_tags,
        out_ids, out_corners, out_orientation, out_translation, out_count);
}

void race_at_destroy(void* raw_handle)
{
    clear_error();
    if (raw_handle == nullptr) {
        return;
    }
    ShimHandle* handle = static_cast<ShimHandle*>(raw_handle);
    if (handle->detector != nullptr) {
        cuAprilTagsDestroy(handle->detector);
    }
    if (handle->device_buffer != nullptr) {
        cudaFree(handle->device_buffer);
    }
    delete handle;
}

const char* race_at_last_error(void)
{
    return g_last_error.c_str();
}

}  // extern "C"
