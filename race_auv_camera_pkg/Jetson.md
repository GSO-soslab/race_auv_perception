# Jetson Orin deployment notes

`race_auv_camera_pkg/apriltag_detector_node` runs in real time on a
Jetson Orin (Nano / AGX / Orin NX) with one or more camera streams.
This file is the operator's quick reference for the Jetson-specific
deployment pieces.

> For the complete from-zero setup (host dependencies, submodule +
> Git LFS bootstrap, CUDA shim build, verification, configuration
> reference and troubleshooting), see
> [`JETSON_ORIN_NANO_SETUP.md`](JETSON_ORIN_NANO_SETUP.md).

Rectification and JPEG encoding run on the CPU by default. The earlier
GPU paths based on OpenCV (`cv2.cuda.remap`, `cv2.cuda.encodeJpeg`) and
`pyNvJPEG` were removed because the OpenCV build bundled with JetPack
does not include `NVCOMPRESS` and `pyNvJPEG`'s wheel is not available
for our Python version. GPU image stages are now available through
in-process NVIDIA libraries instead (nvjpeg decode/encode + VPI CUDA
rectify, section 0.2). See the top-of-file comment in
`image_processing.py` for the CPU rationale.

Tag *detection* can also run on the GPU through the in-process
cuAprilTags backend (`detector_backend: "cuda"`, the default in
`race_auv_bringup/config/apriltag.yaml`). It decodes `tag36h11` only;
the `python` backend (apriltag3) remains available for all families
and for simulation.

---

## 0. Build the CUDA detector (cuAprilTags)

The CUDA backend links NVIDIA's prebuilt `libcuapriltags.a`, which
lives in the `isaac_ros_nitros` repository as Git LFS objects. Fetch
the sparse submodule plus only the cuapriltags LFS blobs:

```bash
sudo apt install -y git-lfs        # or put a git-lfs binary on PATH
scripts/setup_third_party.sh       # from the race_auv_perception repo root
```

Then build the shim package (uses the CUDA toolkit that ships with
JetPack; CMake >= 3.16):

```bash
cd ~/ros2_ws
colcon build --packages-up-to race_auv_apriltag_cuda race_auv_camera_pkg
source install/setup.bash
```

The detector node also needs ROS' `vision_msgs` message package
(declared in `race_auv_camera_pkg/package.xml`); on a fresh Jetson:

```bash
sudo apt install ros-jazzy-vision-msgs
```

Smoke-test the shim before running the stack (renders a synthetic
tag36h11 and checks the decoded id + pose):

```bash
python3 src/race_auv/race_auv_perception/race_auv_apriltag_cuda/test/cuapriltags_smoke.py
```

If the CUDA backend cannot be built (missing submodule, no CUDA
device), set `detector_backend: "python"` in
`race_auv_bringup/config/apriltag.yaml` and install apriltag3 instead
(section 0.1).

---

## 0.1 Install `apriltag3` (python fallback only)

The python backend needs the upstream `AprilRobotics/apriltag` Python
wrapper **with pose estimation** (`estimate_tag_pose`). Note that the
Ubuntu `python3-apriltag` package (3.3.0) does *not* ship that method:
using it makes every detection fall back to an identity pose. Build the
upstream repo instead. It is **not** available via `pip install`
(upstream ships no wheel), so the canonical install is a CMake build:

```bash
sudo apt install -y cmake build-essential python3-dev python3-numpy

git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release
sudo cmake --build build --target install
sudo ldconfig
```

`cmake` defaults the install prefix to `/usr/local`, which puts
`libapriltag.so.3` in `/usr/local/lib` and the
`apriltag.cpython-*.so` Python wrapper in
`/usr/local/lib/python3.12/site-packages/`. `ldconfig` then makes the
shared library discoverable to any process on the system with no
`LD_LIBRARY_PATH` gymnastics.

After the install, symlink the shared library into the same directory
as the Python wrapper. Upstream's CMakeLists installs the wrapper one
directory deeper than `libapriltag.so.3` (it's at
`${prefix}/lib/python3.12/site-packages/`), so the loader's default
search path won't find the shared library when the wrapper is
imported. The symlink fixes that and survives across rebuilds:

```bash
PY_SITE=/usr/local/lib/python3.12/site-packages
sudo ln -sf ../../libapriltag.so.3 $PY_SITE/libapriltag.so.3
```

(If you can't `sudo`, install to your home instead with
`cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local` and adjust the
paths above. The symlink trick is identical.)

`ninja` works in place of the default Makefiles if installed
(`sudo apt install ninja-build`):

```bash
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
sudo cmake --build build --target install
sudo ldconfig
```

The `cv2.cuda` / `pyNvJPEG` image-pipeline GPU paths remain off; the
GPU image stages are implemented with in-process NVIDIA libraries
instead (see section 0.2).

---

## 0.2 GPU image pipeline (nvjpeg + VPI)

`image_pipeline: "cuda"` replaces the CPU image stages with in-process
NVIDIA ones:

* **decode** — nvjpeg (`libnvjpeg.so`), ~10-11 ms at 1920x1080
* **rectify** — VPI 4 CUDA (`WarpMap.fisheye_correction`), ~1-2 ms
* **encode** — nvjpeg, ~2 ms for the annotated frame

Both ship with a full `nvidia-jetpack` install
(`nvidia-jetpack-dev` → `nvidia-vpi-dev`; `cuda-libraries-dev-13-2` →
`libnvjpeg-dev-13-2`). For a minimal install see
`JETSON_ORIN_NANO_SETUP.md` §2; without VPI the CMake configure warns
and skips `librace_auv_image_cuda.so`, and the node must fall back to
`image_pipeline: "cpu"`.

The detection-resolution frame stays in device memory and is consumed
directly by cuAprilTags; only the full-resolution rectified frame is
copied back for the CPU annotation overlay. Measured full chain
(decode + rectify + detect at `process_scale: 0.5` + encode):
**~17 ms/frame** vs ~99 ms on the CPU path.

Requirements: `image_transport: "compressed"` and
`detector_backend: "cuda"` (the node warns and falls back to the CPU
image path otherwise). Config knobs: `image_pipeline`,
`gpu_rectify_interval`. Verify with
`race_auv_apriltag_cuda/test/gpu_image_smoke.py`.

`Failed to open PVA device node` on stderr is harmless: VPI probes the
PVA unit at init and falls back to CUDA.

---

## 1. Power / clocks

```bash
sudo nvpmodel -m 0                  # MAXN (or -m 2 for 25 W)
sudo jetson_clocks                  # lock CPU/GPU to max frequencies
```

Verify with `jtop` (recommended) or `tegrastats`.

---

## 2. Expected per-stage timings (1600x1200, CPU pipeline)

| Stage                            | process_scale=1.0 | process_scale=0.5 |
|----------------------------------|-------------------|-------------------|
| JPEG decode (compressed)         |  5-10 ms          |  5-10 ms          |
| Rectify + ROI crop               | 10-15 ms          | 10-15 ms          |
| Detect (full image)              | 15-30 ms          | --                |
| Detect (process_scale=0.5)       | --                |  5-10 ms          |
| Annotate (boxes + axes + labels) |  3-5 ms           |  3-5 ms           |
| JPEG encode (annotated)          |  8-15 ms          |  8-15 ms          |
| **Total (process_scale=1.0)**    | **~45-80 ms**     | --                |
| **Total (process_scale=0.5)**    | --                | **~35-55 ms**     |

`process_scale: 0.5` cuts the apriltag3 detection time by ~3-5x on the
CPU cores with negligible accuracy loss for dock-sized tags at typical
working distances. Even at full resolution the pipeline keeps up with
5 Hz on the Orin; at 0.5 it has ~2x headroom.

With `detector_backend: "cuda"` the "Detect" stages above are replaced
by a host-to-device copy of the processed BGR image plus the GPU
detect. Measured on the Orin Nano at 1600x1200 (full frame, including
the H2D copy): **~23 ms/frame (~43 fps of headroom)**, with the CPU
cores free. The hardware YAML runs `process_scale: 1.0` so full
resolution reaches the GPU; lower it if you need more margin (the
detector is created for a fixed size, so that requires a restart).

---

## 3. Topside / Foxglove

`apriltag_detection/image` is published as
`sensor_msgs/CompressedImage` (the standard `image/compressed`
transport). `foxglove_bridge` relays it byte-for-byte -- no extra
config on the topside side. Set the Foxglove panel's "Compression
quality" to match `jpeg_quality` for best results.

---

## 4. Known upstream issues (not in this package)

These come from `dwe_camera_driver` and the rclpy / v4l2 stack. They
are **not** bugs in `race_auv_camera_pkg`; they're listed here so you
don't waste time blaming the wrong layer.

### 4.1. `/dev/video2` "can't open camera by index"

Symptom:

```
[camera_node-N] Found camera 'usb-3610000.usb-2.X' at /dev/video2
[camera_node-N] [ WARN:0@0.626] open VIDEOIO(V4L2:/dev/video2): can't open camera by index
[camera_node-N] RuntimeError: Failed to open video device /dev/video2
```

What it actually means: `dwe_camera_driver` uses `v4l2-ctl` to
locate the camera by product name (returns `/dev/videoN`) and then
asks OpenCV's `cv2.VideoCapture(N)` to open it. On Linux, USB UVC
cameras expose two `/dev/videoN` nodes per physical camera:

| Node           | What it is        |
|----------------|-------------------|
| `/dev/video0`  | cam0 **capture**  |
| `/dev/video1`  | cam0 metadata     |
| `/dev/video2`  | cam1 **capture**  |
| `/dev/video3`  | cam1 metadata     |

`v4l2-ctl --list-devices` returns the capture node as soon as it
exists, but the UVC driver hasn't necessarily finished probing the
capture endpoint yet. `cv2.VideoCapture(N)` returns `ENODEV` /
`EBUSY` when called before that probe completes.

**Why `multi_camera.launch.py` works and the original
`camera_apriltag.launch.py` didn't:** launch order.

* `multi_camera.launch.py` lists `stellar_camera_node_1` first
  (maps to `usb-3610000.usb-2.1` -> `/dev/video0`). `/dev/video0`
  opens successfully because the UVC probe for the lower-indexed
  device finishes first. Then `stellar_camera_node_2` opens
  `/dev/video2` and the kernel has had time to finish its probe.
* `camera_apriltag.launch.py` (removed; superseded by
  `multi_camera.launch.py` + `apriltag_detection.launch.py`, chained
  by `bringup_camera_perception.launch.py`) originally iterated the
  YAML cameras list (`cam_front` first, which mapped to
  `usb-3610000.usb-2.3` -> `/dev/video2`) and interleaved each
  detector with its driver. Result: `/dev/video2` was opened first --
  before the UVC probe finished -- and failed; `/dev/video0` opened
  second and won.

**The fix:** `multi_camera.launch.py` keeps the correct driver order
(lowest `/dev/videoN` first) with no interleaved detectors, and
`bringup_camera_perception.launch.py` delays
`apriltag_detection.launch.py` by a few seconds so both cameras are
already publishing before any detector subscribes.

If you can't wait for an upstream fix in `dwe_camera_driver`, the
underlying issue (no retry on `ENODEV`) is still there -- the
workaround just keeps the failure window off the critical path. The
real fix is in `dwe_camera_driver`:

1. Add a small `time.sleep(0.2)` or retry loop between the
   `v4l2-ctl` lookup and the `cv2.VideoCapture` open in
   `dwe_camera_driver/camera_node.py::setup_camera_device`.
2. Or pin the video device path explicitly and open it with
   `cv2.VideoCapture("/dev/videoN", cv2.CAP_V4L2)` instead of by
   index, which sidesteps the OpenCV enumeration race entirely.

### 4.2. rclpy `exc_info` logging crash

```
TypeError: parameter "exc_info" is not one of the recognized logging
options "['throttle_duration_sec', 'throttle_time_source_type',
'skip_first', 'once']"
```

`dwe_camera_driver/camera_node.py` calls
`self.get_logger().fatal(..., exc_info=True)`. rclpy in Jazzy rejects
`exc_info` on its logger. The crash is benign (the node has already
exited), but it pollutes the launch log. Fix is to remove
`exc_info=True` in `dwe_camera_driver`. Out of scope for this package.

### 4.3. `camera.power_line_frequency` not supported

```
[WARN] power_line_frequency: Failed to set control 'power_line_frequency' to 0
[WARN] Control 'power_line_frequency' not supported. Setting parameter
       'camera.power_line_frequency' to read-only.
```

The Stellar cam driver reports this as a warning and the parameter is
marked read-only -- harmless. The `hardware_controls.yaml` value of 0
(Disabled) was correct for desktop testing but is not supported on the
Stellar Air underwater variant. Set it to 2 (60 Hz disabled) or
remove the line in
`race_auv_bringup/config/camera/hardware_controls.yaml` if you want a
clean log.