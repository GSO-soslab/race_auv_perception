# AprilTag Backend Refactor Plan — In-Process CUDA Detector (cuAprilTags)

**Status:** implemented on `jazzy-devel-new-apriltag-isaac` (2026-09-10).
The CUDA shim, Python wrapper, node backend switch and config/launch
changes are in place; the shim smoke test passes on JetPack 7.2 /
CUDA 13.2 (Orin Nano) with the `lib_aarch64_jetpack61` library.
`detector_backend: "cuda"` is set in `race_auv_bringup/config/apriltag.yaml`;
simulation keeps `"python"`. tag25h9 tags are dropped (CUDA supports
tag36h11 only).

Follow-up implemented the same day: an in-process **GPU image pipeline**
(`image_pipeline: "cuda"`) adds nvjpeg decode and encode plus a VPI CUDA
fisheye rectify, keeping the detection-resolution frame in device memory
(see `image_cuda_shim.cpp` and `GpuImagePipeline` in
`race_auv_camera_pkg/apriltag_cuda.py`). Measured full chain:
~17 ms/frame at 1920x1080 with `process_scale: 0.5` vs ~99 ms on the
CPU path. Verify with `gpu_image_smoke.py`.

This document supersedes the earlier `pupil_apriltags` -> `apriltag3` plan.
The existing `AprilTagDetector` (`apriltag_processor.py`) stays in the tree as
the `python` fallback backend; the CUDA backend is added alongside it.

## Goal

Replace the CPU `AprilRobotics/apriltag` (`apriltag3`) detector with an
**in-process CUDA detector** based on NVIDIA's `cuAprilTags` library (the same
library `isaac_ros_apriltag` wraps) without adding any new ROS nodes, topics,
or processes:

```
CompressedImage -> cv_bridge -> CPU fisheye rectify -> BGR (existing)
  -> CuAprilTagDetector.detect(bgr)      # ctypes -> shim .so -> libcuapriltags + cudart
  -> filter (family, id) by YAML size table
  -> rescale translation by size / nominal_size
  -> annotate + publish CompressedImage + Detection3DArray (existing)
  -> existing apriltag_fuser_node (unchanged)
```

CUDA supports `tag36h11` only, so all `tag25h9` entries are dropped (locked
decision). The Python backend is kept for fallback and simulation.

## Why not launch `isaac_ros_apriltag`

* The Isaac ROS package exposes **no in-process API**: `isaac_ros_apriltag`
  ships only a composable node (`nvidia::isaac_ros::apriltag::AprilTagNode`).
* Its CUDA path (`CUAprilTagImpl` in `src/apriltag_node.cpp`) is a thin wrapper
  over three C functions exported by `libcuapriltags.a`.
* Running it as a node would force publishing a raw (rectified) image plus a
  `camera_info`, launching a container, and waiting on `tag_detections` -- all
  rejected by the user.
* Consequence: we do **not** need NITROS, VPI, `isaac_ros_common`, or a
  `colcon build` of `isaac_ros_apriltag`. Only the cuAprilTags static library
  and header from `isaac_ros_nitros` are used.

## Decisions locked

| Question | Decision |
|---|---|
| Integration | Existing Python node kept; new `CuAprilTagDetector` in `apriltag_cuda.py`; a thin C shim `.so` underneath (ctypes cannot load a static `.a`, and the shim hides NVIDIA's struct padding). |
| Library source | `isaac_ros_nitros` git submodule (branch `release-4.6`) under `third_party/`, with Git LFS + sparse checkout. No vendoring / redistribution. |
| Tag family | `tag36h11` only; all `tag25h9` configuration removed. |
| Tag size | One detector per camera created at `cuda_nominal_size` (0.125 m); translation rescaled per tag: `t_true = t_reported * (size_true / nominal_size)`; rotation unchanged. Exact per `cuAprilTags.h` ("translation ... expressed in the same units as the tag_size"). |
| Rectification | Stays CPU (`image_processing.ImageRectifier`); the Stellar cameras are fisheye and cuAprilTags requires an undistorted input. |
| Backend switch | `detector_backend: cuda|python` ROS/YAML param; `python` remains the default fallback. |
| Platform | Jetson Orin, JetPack 7.2, ROS 2 Jazzy (x86_64/Jazzy dev box for pre-testing). |
| Max tags | 64 per camera (matches the Isaac node default). |

## cuAprilTags contract

Header:
`third_party/isaac_ros_nitros/isaac_ros_nitros/lib/cuapriltags/cuapriltags/cuAprilTags.h`
(SPDX Apache-2.0).
Library: `lib_aarch64_jetpack61/libcuapriltags.a` (aarch64) /
`lib_x86_64_cuda_12_6/libcuapriltags.a` (x86_64). Both are Git LFS objects.

```c
int nvCreateAprilTagsDetector(cuAprilTagsHandle* h, uint32_t w, uint32_t h,
    uint32_t tile_size, cuAprilTagsFamily family,
    const cuAprilTagsCameraIntrinsics_t* cam, float tag_dim);
int cuAprilTagsDetect(cuAprilTagsHandle h, const cuAprilTagsImageInput_t* in,
    cuAprilTagsID_t* out, uint32_t* n, uint32_t max_tags, CUstream_st* stream);
int cuAprilTagsDestroy(cuAprilTagsHandle h);
```

* Input image must be **undistorted** with type `uchar3` (BGR, 3 bytes/px).
* Pose is solved inside the library from `cam` + `tag_dim`; `translation` is in
  the same units as `tag_dim`; `orientation[9]` is column-major.
* `cuAprilTagsID_t` is 88 bytes on the host (`float2` members are 8-byte
  aligned). The shim flattens everything so Python never mirrors this layout,
  and `static_assert(sizeof(cuAprilTagsID_t) == 88)` guards ABI drift.
* Only `NVAT_TAG36H11` is supported.
* Corner order may differ from `apriltag3`'s `lb-rb-rt-lt`; only drawing uses
  corners, so there is no consumer impact.

## Implementation

### 0. Prerequisites (Jetson, one-time)

* JetPack 7.2 with CUDA toolkit (`cudart`) and `cmake`.
* `git-lfs` (required: `.a` files are LFS objects; a bare clone yields a
  132-byte pointer).

### 1. Submodule + bootstrap

In the `race_auv` repo (branch `jazzy-devel-new-apriltag`):

```bash
git submodule add -b release-4.6 \
    https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git \
    third_party/isaac_ros_nitros
```

A full checkout would download every LFS blob in that repo (cuVSLAM, cuMotion,
hundreds of MB), so add `scripts/setup_third_party.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
SM="third_party/isaac_ros_nitros"
command -v git-lfs >/dev/null || { echo "install git-lfs" >&2; exit 1; }
git submodule update --init --no-checkout "$SM"
git -C "$SM" sparse-checkout init --cone
git -C "$SM" sparse-checkout set isaac_ros_nitros/lib/cuapriltags
git -C "$SM" checkout
git -C "$SM" lfs pull --include="isaac_ros_nitros/lib/cuapriltags/**"
```

Every fresh clone runs this before building; documented in `Jetson.md`.

### 2. New ament_cmake package `race_auv_apriltag_cuda/`

Sibling of `race_auv_camera_pkg` at the repo root (nested compiled packages
inside an `ament_python` package are avoided):

```
race_auv_apriltag_cuda/
|- CMakeLists.txt
|- package.xml
|- include/race_auv_apriltag_cuda/cuapriltags_shim.h
|- src/cuapriltags_shim.cpp
```

`CMakeLists.txt`:

* `find_package(ament_cmake REQUIRED)` and `find_package(CUDAToolkit REQUIRED)`.
* `CUAPRILTAGS_ROOT` cache var, default
  `${CMAKE_CURRENT_SOURCE_DIR}/../third_party/isaac_ros_nitros/isaac_ros_nitros/lib/cuapriltags`.
* Library dir: `lib_aarch64_jetpack61` on aarch64, `lib_x86_64_cuda_12_6` on
  x86_64. `FATAL_ERROR` when `${libdir}/libcuapriltags.a` is missing, with a
  message pointing at the bootstrap script (so a missing `git lfs pull` is
  obvious).
* `add_library(race_auv_apriltag_cuda SHARED src/cuapriltags_shim.cpp)`,
  include dirs `${CUAPRILTAGS_ROOT}/cuapriltags` + `${CUDAToolkit_INCLUDE_DIRS}`,
  link `"${libdir}/libcuapriltags.a"` + `CUDA::cudart`.
* Install the shared library to `share/race_auv_apriltag_cuda/lib/` so Python
  resolves it through `get_package_share_directory` (no `LD_LIBRARY_PATH`), and
  the header to `include/`.

`package.xml`: `ament_cmake` buildtool only; CUDA comes from JetPack (there is
no portable rosdep key for the toolkit).

Shim API (`cuapriltags_shim.h`):

```c
void* race_at_create(int w, int h, int tile_size, float tag_size,
                     float fx, float fy, float cx, float cy);
int   race_at_detect(void* handle, const uint8_t* host_bgr, size_t pitch,
                     int w, int h, int max_tags,
                     uint16_t* out_ids,
                     float* out_corners,      /* 8 * max_tags: x,y per corner */
                     float* out_orientation,  /* 9 * max_tags, column-major */
                     float* out_translation,  /* 3 * max_tags */
                     int* out_count);
void  race_at_destroy(void* handle);
const char* race_at_last_error(void);
```

`cuapriltags_shim.cpp`:

* Opaque state: `cuAprilTagsHandle`, `uchar3* d_buf` (single `cudaMalloc` at
  `w*h*3`, reused every frame), dimensions, `std::vector<cuAprilTagsID_t>`.
* `race_at_create`: `cudaMalloc` + `nvCreateAprilTagsDetector` with
  `NVAT_TAG36H11`, the intrinsics and `tag_size`; returns `nullptr` and sets the
  error string on failure.
* `race_at_detect`: `cudaMemcpy2D` H2D (`pitch = w*3`), fill
  `cuAprilTagsImageInput_t`, call `cuAprilTagsDetect(..., /*stream=*/0)`, then
  flatten ids / corners / column-major orientation / translation and the count
  into the caller's arrays.
* `race_at_destroy`: `cuAprilTagsDestroy` + `cudaFree`.
* Thread-local error string updated by every entry point; nonzero status is
  surfaced to Python.
* `static_assert(sizeof(cuAprilTagsID_t) == 88, "...");`

### 3. `race_auv_camera_pkg/race_auv_camera_pkg/apriltag_cuda.py`

`CuAprilTagDetector` mirrors `AprilTagDetector`'s public surface:

```python
class CuAprilTagDetector:
    def __init__(self, family, id_to_size, camera_intrinsics, image_size,
                 logger, nominal_size=0.125, tile_size=4, max_tags=64): ...
    def detect(self, bgr): ...                    # same dicts as AprilTagDetector
    def annotate(self, image, detections): ...    # delegates to shared annotator
```

* Loads `librace_auv_apriltag_cuda.so` from
  `get_package_share_directory("race_auv_apriltag_cuda")/lib/` via
  `ctypes.CDLL`.
* Allocates the output buffers once (`uint16[64]`, `float[8*64]`,
  `float[9*64]`, `float[3*64]`, `c_int`).
* `detect`:
  * `bgr = np.ascontiguousarray(bgr)`; pass `bgr.ctypes.data` and
    `pitch = w*3`.
  * Nonzero status -> log and return `[]`.
  * Per returned tag: drop when `(family, id)` is not in `id_to_size`;
    `R = np.asarray(orientation).reshape(3, 3, order="F")`;
    `t = translation * (size / nominal_size)`; build `T`; sanitize via
    `is_bad_rotation` / `sanitize_rotation`; set `bad_pose` when recovery
    fails or values are non-finite.
  * `corners` returned as `4x2 float32`; `size` included for axis length.
* Exposes `camera_matrix` / `dist_coeffs` exactly like `AprilTagDetector` so
  the node's `annotate(...)` call site is unchanged.
* Import is lazy in `apriltag_detector_node._build_pipeline`, so a workspace
  without the CUDA package still runs the `python` backend.

### 4. `apriltag_processor.py` -- shared annotation

* Extract `AprilTagDetector.annotate` + `_draw_bad_pose` +
  `_draw_label_block` into a module-level
  `annotate_detections(image, detections, camera_matrix, dist_coeffs, logger)`.
* `AprilTagDetector.annotate` delegates to it (no behavior change).
* `apriltag_cuda.py` imports the same function, so boxes, axes and labels stay
  pixel-identical between backends.

### 5. `apriltag_detector_node.py`

* New params: `detector_backend` (`"python"` default; YAML sets `"cuda"`),
  `cuda_nominal_size` (0.125), `cuda_tile_size` (4), `cuda_max_tags` (64).
* `_build_pipeline`:
  * Keep the rectifier and `process_scale` math as-is. The CUDA detector is
    created for the **processed** image size, so `process_scale < 1` still
    works; recommend `process_scale: 1.0` for the CUDA backend to regain
    full-resolution range (GPU detect is cheap).
  * `detector_backend == "cuda"` -> `CuAprilTagDetector(...)`, else
    `AprilTagDetector(...)` (today's path).
  * Log the chosen backend in the startup banner.
* `_process_frame`: branch only on what is fed to `detect()`:

  ```python
  if self._detector_backend == "cuda":
      detections = detector.detect(small)                            # BGR
  else:
      detections = detector.detect(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
  ```

  Everything else (corner upscale, `min_edge_dist`, sanitize, annotate,
  crosshair, JPEG, `Detection3DArray`, fuser) is untouched.
* `tags_override` / YAML tag loading unchanged.

### 6. Config

`race_auv_bringup/config/apriltag.yaml`:

* Delete every `tag25h9` entry (global list and per-camera comments).
* Add to `detector_defaults`:
  `detector_backend: "cuda"`, `cuda_tile_size: 4`,
  `cuda_nominal_size: 0.125`, `cuda_max_tags: 64`.
* The historical `use_cuda` / `jpeg_backend` keys stay ignored; the new
  `detector_backend` is the real switch.

`race_auv_bringup/config/simulation/apriltag.yaml`:

* Same 25h9 removal (sim tags become `tag36h11 @ {0.15, 0.05}`).
* Keep `detector_backend: "python"` for the sim (optional `"cuda"` on the RTX
  dev box; nominal size 0.15, rescale 0.05/0.15).

### 7. Launch

`camera_apriltag.launch.py` and `simulation/apriltag_sim.launch.py` only need to
forward the new params to `apriltag_detector_node`. No new nodes, no raw image
topic, no separate container.

### 8. `race_auv_camera_pkg/package.xml`

* Add `<exec_depend>race_auv_apriltag_cuda</exec_depend>`.
* Add `<exec_depend>ament_index_python</exec_depend>` (used to locate the
  shim `.so`).
* `apriltag3` (`AprilRobotics/apriltag`) stays documented for the `python`
  fallback only.

### 9. `Jetson.md`

* Replace section 0 "Install `apriltag3`" with:
  1. `git-lfs` + `scripts/setup_third_party.sh` (submodule bootstrap),
  2. build command
     (`colcon build --packages-up-to race_auv_apriltag_cuda race_auv_camera_pkg`),
  3. note that `apriltag3` is only needed for the `python` fallback.
* Update the per-stage timing table with the CUDA detect expectation (H2D copy
  + GPU detect) once measured.

## Verification

1. **Shim smoke test** (before node wiring): feed a known rectified BGR frame
   through `race_at_*`; confirm ids, corners and poses are sane.
2. **Parity**: run the same frame/tag with `detector_backend: python` and
   `cuda`; compare `T` (expect only IPPE branch-level differences, no gross
   scale errors).
3. **Size rescale**: a 4 cm tag next to a 12.5 cm tag; translations must match
   the Python backend within noise.
4. **Topic contract**: `apriltag_detection/detections3d` and
   `apriltag_detection/image` unchanged; the fuser still publishes
   `race_station/dock_point`.
5. **Performance**: 15 Hz per camera, `tegrastats` CPU headroom, detect-stage
   timing logged.

## Risks

* **aarch64 binary vintage**: 4.6 ships `lib_aarch64_jetpack61` (JetPack
  6.1 / CUDA 12). JetPack 7.2 should run it per NVIDIA's own pairing, but the
  link/run smoke test gates the approach. Fallback: `detector_backend: python`.
* **Git LFS**: bandwidth quota and the `git-lfs` requirement; mitigated by the
  sparse checkout bootstrap.
* **Closed binary**: `libcuapriltags.a` is not open source; we reference it via
  submodule and do not redistribute it.
* **H2D copy**: `w*h*3` bytes per frame is unavoidable while rectification is
  CPU; acceptable at 1600x1200 / 15 Hz.
* **Pose branch differences** vs `apriltag3`; the fuser's weighted joint
  Umeyama / RANSAC absorbs them.
* **Fixed image size per detector**: `tag_size` and image dimensions are set at
  creation; changing `process_scale` requires a node restart.
* **Corner order** differs from `apriltag3`; annotation only, no consumer
  impact.
* Thread safety: one detector handle per node/camera; the shim is not
  re-entrant and relies on the node's single executor thread per camera.

## Rollback

* Set `detector_backend: "python"` in `apriltag.yaml` -- no code revert needed.
* Optional full removal: drop the submodule, `race_auv_apriltag_cuda/`,
  `apriltag_cuda.py`, and the `package.xml` `exec_depend`.
