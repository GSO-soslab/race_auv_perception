// SPDX-License-Identifier: Apache-2.0
//
// In-process GPU image stages for the AprilTag pipeline.
//
// Chains, without touching the CPU in between:
//
//   JPEG bytes --nvjpeg--> device BGR (decoded)
//              --VPI CUDA remap--> device BGR (rectified, N jobs)
//              --nvjpeg encode<-- host BGR (annotated)
//
// Rectify jobs let the caller produce more than one output size from the
// same decoded image (e.g. full resolution for display, process_scale
// resolution for detection). Each job owns its warp map, remap payload
// and output buffer.
//
// The warp map is built from the same dense fisheye maps OpenCV's
// cv2.fisheye.initUndistortRectifyMap produces, so the GPU output
// matches the CPU rectifier.

#include "race_auv_apriltag_cuda/image_cuda_shim.h"

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/VPI.h>
#include <vpi/WarpMap.h>
#include <vpi/algo/Remap.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void clear_error()
{
    g_last_error.clear();
}

int fail(const std::string& message)
{
    g_last_error = message;
    return -1;
}

constexpr int kMaxJobs = 2;

struct RectifyJob
{
    bool configured = false;
    int out_width = 0;
    int out_height = 0;
    VPIWarpMap warp{};
    VPIPayload payload = nullptr;
    uint8_t* device_out = nullptr;
    VPIImage vpi_out = nullptr;
    VPIImageData out_data{};
};

struct ImagePipeline
{
    int width = 0;
    int height = 0;

    uint8_t* device_decoded = nullptr;
    VPIImage vpi_in = nullptr;
    VPIImageData in_data{};

    nvjpegHandle_t njpeg = nullptr;
    nvjpegJpegState_t decode_state = nullptr;
    nvjpegEncoderState_t encode_state = nullptr;
    nvjpegEncoderParams_t encode_params = nullptr;
    cudaStream_t cuda_stream = nullptr;

    VPIStream vpi_stream = nullptr;

    uint8_t* device_encode = nullptr;
    std::size_t device_encode_bytes = 0;
    std::vector<uint8_t> bitstream;

    RectifyJob jobs[kMaxJobs];
    bool destroyed = false;
};

void destroy_job(RectifyJob& job)
{
    if (job.payload != nullptr) {
        vpiPayloadDestroy(job.payload);
        job.payload = nullptr;
    }
    if (job.vpi_out != nullptr) {
        vpiImageDestroy(job.vpi_out);
        job.vpi_out = nullptr;
    }
    if (job.device_out != nullptr) {
        cudaFree(job.device_out);
        job.device_out = nullptr;
    }
    if (job.warp.keypoints != nullptr) {
        vpiWarpMapFreeData(&job.warp);
    }
    job.configured = false;
    job.out_width = 0;
    job.out_height = 0;
}

ImagePipeline* as_pipeline(void* handle)
{
    return static_cast<ImagePipeline*>(handle);
}

bool valid_job_index(int job)
{
    return job >= 0 && job < kMaxJobs;
}

}  // namespace

extern "C" {

void* race_img_create(int width, int height)
{
    clear_error();
    if (width <= 0 || height <= 0) {
        fail("race_img_create: width/height must be positive");
        return nullptr;
    }

    ImagePipeline* pipeline = new (std::nothrow) ImagePipeline();
    if (pipeline == nullptr) {
        fail("race_img_create: out of memory");
        return nullptr;
    }
    pipeline->width = width;
    pipeline->height = height;

    if (nvjpegCreateSimple(&pipeline->njpeg) != NVJPEG_STATUS_SUCCESS ||
        nvjpegJpegStateCreate(pipeline->njpeg, &pipeline->decode_state) !=
            NVJPEG_STATUS_SUCCESS) {
        fail("race_img_create: nvjpeg init failed");
        race_img_destroy(pipeline);
        return nullptr;
    }
    if (nvjpegEncoderStateCreate(pipeline->njpeg, &pipeline->encode_state,
                                 nullptr) != NVJPEG_STATUS_SUCCESS ||
        nvjpegEncoderParamsCreate(pipeline->njpeg, &pipeline->encode_params,
                                  nullptr) != NVJPEG_STATUS_SUCCESS) {
        fail("race_img_create: nvjpeg encoder init failed");
        race_img_destroy(pipeline);
        return nullptr;
    }

    if (cudaStreamCreate(&pipeline->cuda_stream) != cudaSuccess) {
        fail("race_img_create: cudaStreamCreate failed");
        race_img_destroy(pipeline);
        return nullptr;
    }

    const std::size_t decoded_bytes =
        static_cast<std::size_t>(width) * height * 3u;
    if (cudaMalloc(reinterpret_cast<void**>(&pipeline->device_decoded),
                   decoded_bytes) != cudaSuccess) {
        fail("race_img_create: cudaMalloc(decoded) failed");
        race_img_destroy(pipeline);
        return nullptr;
    }

    if (vpiStreamCreate(VPI_BACKEND_CUDA, &pipeline->vpi_stream) != VPI_SUCCESS) {
        fail("race_img_create: vpiStreamCreate failed");
        race_img_destroy(pipeline);
        return nullptr;
    }

    std::memset(&pipeline->in_data, 0, sizeof(pipeline->in_data));
    pipeline->in_data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    pipeline->in_data.buffer.pitch.format = VPI_IMAGE_FORMAT_BGR8;
    pipeline->in_data.buffer.pitch.numPlanes = 1;
    pipeline->in_data.buffer.pitch.planes[0].width = width;
    pipeline->in_data.buffer.pitch.planes[0].height = height;
    pipeline->in_data.buffer.pitch.planes[0].pitchBytes = width * 3;
    pipeline->in_data.buffer.pitch.planes[0].pBase = pipeline->device_decoded;
    if (vpiImageCreateWrapper(&pipeline->in_data, nullptr, VPI_BACKEND_CUDA,
                              &pipeline->vpi_in) != VPI_SUCCESS) {
        fail("race_img_create: vpiImageCreateWrapper(input) failed");
        race_img_destroy(pipeline);
        return nullptr;
    }

    return pipeline;
}

void race_img_destroy(void* handle)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr) {
        return;
    }
    if (pipeline->destroyed) {
        return;
    }
    pipeline->destroyed = true;

    for (int i = 0; i < kMaxJobs; ++i) {
        destroy_job(pipeline->jobs[i]);
    }
    if (pipeline->vpi_in != nullptr) {
        vpiImageDestroy(pipeline->vpi_in);
    }
    if (pipeline->vpi_stream != nullptr) {
        vpiStreamDestroy(pipeline->vpi_stream);
    }
    if (pipeline->device_decoded != nullptr) {
        cudaFree(pipeline->device_decoded);
    }
    if (pipeline->device_encode != nullptr) {
        cudaFree(pipeline->device_encode);
    }
    if (pipeline->encode_params != nullptr) {
        nvjpegEncoderParamsDestroy(pipeline->encode_params);
    }
    if (pipeline->encode_state != nullptr) {
        nvjpegEncoderStateDestroy(pipeline->encode_state);
    }
    if (pipeline->decode_state != nullptr) {
        nvjpegJpegStateDestroy(pipeline->decode_state);
    }
    if (pipeline->njpeg != nullptr) {
        nvjpegDestroy(pipeline->njpeg);
    }
    if (pipeline->cuda_stream != nullptr) {
        cudaStreamDestroy(pipeline->cuda_stream);
    }
    delete pipeline;
}

int race_img_set_fisheye_job(void* handle, int job_index,
    const float* map_x, const float* map_y,
    int out_width, int out_height, int interval)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr) {
        return fail("race_img_set_fisheye_job: null handle");
    }
    if (!valid_job_index(job_index)) {
        return fail("race_img_set_fisheye_job: job index out of range");
    }
    if (map_x == nullptr || map_y == nullptr || out_width <= 0 ||
        out_height <= 0) {
        return fail("race_img_set_fisheye_job: invalid arguments");
    }
    if (interval < 1 || (interval & (interval - 1)) != 0) {
        return fail("race_img_set_fisheye_job: interval must be a power of two");
    }

    RectifyJob& job = pipeline->jobs[job_index];
    destroy_job(job);

    std::memset(&job.warp, 0, sizeof(job.warp));
    job.warp.grid.numHorizRegions = 1;
    job.warp.grid.regionWidth[0] = static_cast<int16_t>(out_width);
    job.warp.grid.horizInterval[0] = static_cast<int16_t>(interval);
    job.warp.grid.numVertRegions = 1;
    job.warp.grid.regionHeight[0] = static_cast<int16_t>(out_height);
    job.warp.grid.vertInterval[0] = static_cast<int16_t>(interval);

    if (vpiWarpMapAllocData(&job.warp) != VPI_SUCCESS) {
        return fail("race_img_set_fisheye_job: vpiWarpMapAllocData failed");
    }
    if (vpiWarpMapGenerateIdentity(&job.warp) != VPI_SUCCESS) {
        return fail("race_img_set_fisheye_job: vpiWarpMapGenerateIdentity failed");
    }

    for (int row = 0; row < job.warp.numVertPoints; ++row) {
        VPIKeypointF32* points = reinterpret_cast<VPIKeypointF32*>(
            reinterpret_cast<char*>(job.warp.keypoints) +
            static_cast<std::size_t>(row) * job.warp.pitchBytes);
        for (int col = 0; col < job.warp.numHorizPoints; ++col) {
            int x = static_cast<int>(std::lround(points[col].x));
            int y = static_cast<int>(std::lround(points[col].y));
            x = std::min(std::max(x, 0), out_width - 1);
            y = std::min(std::max(y, 0), out_height - 1);
            points[col].x = map_x[static_cast<std::size_t>(y) * out_width + x];
            points[col].y = map_y[static_cast<std::size_t>(y) * out_width + x];
        }
    }

    if (vpiCreateRemap(VPI_BACKEND_CUDA, &job.warp, &job.payload) != VPI_SUCCESS) {
        return fail("race_img_set_fisheye_job: vpiCreateRemap failed");
    }

    const std::size_t out_bytes =
        static_cast<std::size_t>(out_width) * out_height * 3u;
    if (cudaMalloc(reinterpret_cast<void**>(&job.device_out), out_bytes) !=
        cudaSuccess) {
        return fail("race_img_set_fisheye_job: cudaMalloc(output) failed");
    }

    std::memset(&job.out_data, 0, sizeof(job.out_data));
    job.out_data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    job.out_data.buffer.pitch.format = VPI_IMAGE_FORMAT_BGR8;
    job.out_data.buffer.pitch.numPlanes = 1;
    job.out_data.buffer.pitch.planes[0].width = out_width;
    job.out_data.buffer.pitch.planes[0].height = out_height;
    job.out_data.buffer.pitch.planes[0].pitchBytes = out_width * 3;
    job.out_data.buffer.pitch.planes[0].pBase = job.device_out;
    if (vpiImageCreateWrapper(&job.out_data, nullptr, VPI_BACKEND_CUDA,
                              &job.vpi_out) != VPI_SUCCESS) {
        return fail("race_img_set_fisheye_job: wrap output failed");
    }

    job.out_width = out_width;
    job.out_height = out_height;
    job.configured = true;
    return 0;
}

int race_img_decode(void* handle, const uint8_t* jpeg, size_t length)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr) {
        return fail("race_img_decode: null handle");
    }
    if (jpeg == nullptr || length == 0) {
        return fail("race_img_decode: empty JPEG");
    }

    int components = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT] = {0};
    int heights[NVJPEG_MAX_COMPONENT] = {0};
    nvjpegStatus_t status = nvjpegGetImageInfo(
        pipeline->njpeg, jpeg, length, &components, &subsampling,
        widths, heights);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return fail("race_img_decode: nvjpegGetImageInfo failed");
    }
    if (widths[0] != pipeline->width || heights[0] != pipeline->height) {
        return fail(
            "race_img_decode: JPEG is " + std::to_string(widths[0]) + "x" +
            std::to_string(heights[0]) + " but pipeline was created for " +
            std::to_string(pipeline->width) + "x" +
            std::to_string(pipeline->height));
    }

    nvjpegImage_t destination;
    std::memset(&destination, 0, sizeof(destination));
    destination.channel[0] = pipeline->device_decoded;
    destination.pitch[0] = static_cast<std::size_t>(pipeline->width) * 3u;

    status = nvjpegDecode(pipeline->njpeg, pipeline->decode_state, jpeg, length,
                          NVJPEG_OUTPUT_BGRI, &destination, pipeline->cuda_stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return fail("race_img_decode: nvjpegDecode failed (status " +
                    std::to_string(static_cast<int>(status)) + ")");
    }
    if (cudaStreamSynchronize(pipeline->cuda_stream) != cudaSuccess) {
        return fail("race_img_decode: cudaStreamSynchronize failed");
    }
    return 0;
}

int race_img_rectify(void* handle, int job_index)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr) {
        return fail("race_img_rectify: null handle");
    }
    if (!valid_job_index(job_index) || !pipeline->jobs[job_index].configured) {
        return fail("race_img_rectify: job not configured");
    }
    RectifyJob& job = pipeline->jobs[job_index];
    if (vpiSubmitRemap(pipeline->vpi_stream, VPI_BACKEND_CUDA, job.payload,
                       pipeline->vpi_in, job.vpi_out, VPI_INTERP_LINEAR,
                       VPI_BORDER_ZERO, 0) != VPI_SUCCESS) {
        return fail("race_img_rectify: vpiSubmitRemap failed");
    }
    if (vpiStreamSync(pipeline->vpi_stream) != VPI_SUCCESS) {
        return fail("race_img_rectify: vpiStreamSync failed");
    }
    return 0;
}

const void* race_img_get_decoded(void* handle)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    return pipeline == nullptr ? nullptr : pipeline->device_decoded;
}

const void* race_img_get_rectified(void* handle, int job_index)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr || !valid_job_index(job_index)) {
        return nullptr;
    }
    return pipeline->jobs[job_index].device_out;
}

int race_img_get_rectified_size(void* handle, int job_index, int* width, int* height)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr || !valid_job_index(job_index) ||
        width == nullptr || height == nullptr) {
        return fail("race_img_get_rectified_size: invalid arguments");
    }
    if (!pipeline->jobs[job_index].configured) {
        return fail("race_img_get_rectified_size: job not configured");
    }
    *width = pipeline->jobs[job_index].out_width;
    *height = pipeline->jobs[job_index].out_height;
    return 0;
}

int race_img_rectified_to_host(void* handle, int job_index, uint8_t* host,
    size_t pitch)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr || !valid_job_index(job_index) || host == nullptr) {
        return fail("race_img_rectified_to_host: invalid arguments");
    }
    RectifyJob& job = pipeline->jobs[job_index];
    if (!job.configured) {
        return fail("race_img_rectified_to_host: job not configured");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(job.out_width) * 3u;
    if (pitch < row_bytes) {
        return fail("race_img_rectified_to_host: pitch smaller than width*3");
    }
    cudaError_t status = cudaMemcpy2D(
        host, pitch, job.device_out, row_bytes, row_bytes,
        static_cast<std::size_t>(job.out_height), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        return fail(std::string("race_img_rectified_to_host: cudaMemcpy2D failed: ") +
                    cudaGetErrorString(status));
    }
    return 0;
}

int race_img_encode(void* handle, const uint8_t* host_bgr, size_t pitch,
    int width, int height, int quality, const uint8_t** out_data,
    size_t* out_length)
{
    clear_error();
    ImagePipeline* pipeline = as_pipeline(handle);
    if (pipeline == nullptr) {
        return fail("race_img_encode: null handle");
    }
    if (host_bgr == nullptr || width <= 0 || height <= 0 || out_data == nullptr ||
        out_length == nullptr) {
        return fail("race_img_encode: invalid arguments");
    }
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3u;
    if (pitch < row_bytes) {
        return fail("race_img_encode: pitch smaller than width*3");
    }

    const std::size_t needed = static_cast<std::size_t>(width) * height * 3u;
    if (needed > pipeline->device_encode_bytes) {
        if (pipeline->device_encode != nullptr) {
            cudaFree(pipeline->device_encode);
            pipeline->device_encode = nullptr;
            pipeline->device_encode_bytes = 0;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&pipeline->device_encode), needed) !=
            cudaSuccess) {
            return fail("race_img_encode: cudaMalloc(staging) failed");
        }
        pipeline->device_encode_bytes = needed;
    }

    cudaError_t cuda_status = cudaMemcpy2D(
        pipeline->device_encode, row_bytes, host_bgr, pitch, row_bytes,
        static_cast<std::size_t>(height), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        return fail(std::string("race_img_encode: cudaMemcpy2D failed: ") +
                    cudaGetErrorString(cuda_status));
    }

    nvjpegImage_t source;
    std::memset(&source, 0, sizeof(source));
    source.channel[0] = pipeline->device_encode;
    source.pitch[0] = row_bytes;

    if (nvjpegEncoderParamsSetQuality(pipeline->encode_params, quality,
                                      pipeline->cuda_stream) !=
        NVJPEG_STATUS_SUCCESS) {
        return fail("race_img_encode: nvjpegEncoderParamsSetQuality failed");
    }
    if (nvjpegEncoderParamsSetSamplingFactors(pipeline->encode_params,
                                              NVJPEG_CSS_420,
                                              pipeline->cuda_stream) !=
        NVJPEG_STATUS_SUCCESS) {
        return fail("race_img_encode: nvjpegEncoderParamsSetSamplingFactors failed");
    }
    nvjpegStatus_t status = nvjpegEncodeImage(
        pipeline->njpeg, pipeline->encode_state, pipeline->encode_params,
        &source, NVJPEG_INPUT_BGRI, width, height, pipeline->cuda_stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return fail("race_img_encode: nvjpegEncodeImage failed (status " +
                    std::to_string(static_cast<int>(status)) + ")");
    }

    size_t length = 0;
    status = nvjpegEncodeRetrieveBitstream(pipeline->njpeg,
                                           pipeline->encode_state, nullptr,
                                           &length, pipeline->cuda_stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return fail("race_img_encode: bitstream size query failed");
    }
    pipeline->bitstream.resize(length);
    status = nvjpegEncodeRetrieveBitstream(
        pipeline->njpeg, pipeline->encode_state, pipeline->bitstream.data(),
        &length, pipeline->cuda_stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        return fail("race_img_encode: bitstream retrieve failed");
    }
    if (cudaStreamSynchronize(pipeline->cuda_stream) != cudaSuccess) {
        return fail("race_img_encode: cudaStreamSynchronize failed");
    }

    *out_data = pipeline->bitstream.data();
    *out_length = length;
    return 0;
}

const char* race_img_last_error(void)
{
    return g_last_error.c_str();
}

}  // extern "C"
