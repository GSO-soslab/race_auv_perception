# Race AUV AprilTag stack — Jetson Orin Nano bring-up guide

Complete, from-zero instructions for running the AprilTag detection
pipeline of `race_auv_camera_pkg` on a Jetson Orin Nano, both on the
real robot (two DWE Stellar cameras) and in Stonefish simulation.

This document is the authoritative setup guide. `Jetson.md` remains the
quick reference for power/clock setup, per-stage timings, and known
upstream driver issues; this file covers everything needed to get the
stack built and running.

---

## 0. What the pipeline is

```
 DWE camera driver            apriltag_detector_node (per camera)
 ─────────────────           ───────────────────────────────────
 /race/stellar2/image/  ───►  subscribe CompressedImage
   compressed                 ├─ image stages (image_pipeline):
                              │    "cuda"   nvjpeg decode -> VPI CUDA
                              │             fisheye rectify -> nvjpeg encode
                              │    "cpu"    cv_bridge/cv2 + CPU rectify
                              │             + cv2.imencode
                              ├─ detector backend:
                              │    "cuda"   cuAprilTags (race_auv_apriltag_cuda)
                              │    "python" apriltag3
                              ├─ per-tag size rescale + pose sanitize
                              ├─ annotate boxes / axes / labels
                              └─ publish:
                                   <ns>/apriltag_detection/detections3d
                                   <ns>/apriltag_detection/image          (JPEG)
```

* Two cameras (`cam_front`, `cam_down`), one detector process each.
* The detector itself publishes **no TF**.
* In simulation, `apriltag_fuser_node` additionally subscribes to all
  `detections3d` topics and fuses them into `race_station/dock_point`.

Two detector backends:

| Backend | Library | Tag families | Where used |
|---|---|---|---|
| `cuda` | NVIDIA cuAprilTags (in-process, via ctypes shim) | **`tag36h11` only** | default on hardware |
| `python` | `AprilRobotics/apriltag` Python binding | all families | simulation, fallback |

Two image-stage backends (`image_pipeline`):

| Backend | Decode | Rectify | Encode | Where used |
|---|---|---|---|---|
| `cpu` | `cv_bridge`/`cv2` | CPU `ImageRectifier` | `cv2.imencode` | simulation, fallback |
| `cuda` | nvjpeg | VPI CUDA (`WarpMap.fisheye`) | nvjpeg | default on hardware |

> **Important:** cuAprilTags cannot decode `tag25h9`. All `tag25h9`
> entries were removed from both configs. Only `tag36h11` tags are
> detected now.

---

## 1. Verified baseline

This guide was validated on:

| Component | Value | Check command |
|---|---|---|
| Board | Jetson Orin Nano (generic) | `cat /proc/device-tree/model` |
| L4T | R39.2.1 | `cat /etc/nv_tegra_release` |
| JetPack | 7.2.1-b49 | `dpkg-query -W nvidia-jetpack` |
| CUDA runtime | 13.2.86 | `nvcc --version` (after adding to PATH) |
| CMake | 3.28.3 | `cmake --version` |
| ROS 2 | Jazzy | `echo $ROS_DISTRO` |
| Python | 3.12.3 | `python3 --version` |
| OpenCV | apt 4.6.0 (no CUDA needed; see §9.3) | `python3 -c "import cv2; print(cv2.__version__)"` |
| VPI | 4.1.4 (`/opt/nvidia/vpi4`, CUDA backend) | `dpkg-query -W libnvvpi4` |
| nvjpeg | 13.1.0 (`libnvjpeg-13-2`) | `dpkg-query -W libnvjpeg-13-2` |
| git-lfs | 3.4.1 (user-local `~/.local/bin`) | `git lfs version` |

The prebuilt `libcuapriltags.a` shipped by `isaac_ros_nitros` is built
for JetPack 6.1 / CUDA 12, but its cubins include `sm_87` (Orin) and it
links and runs against this machine's CUDA 13.2. This was verified with
the smoke test in §6.1. If a future JetPack drops `sm_87` support, the
fallback is `detector_backend: "python"`.

---

## 2. Host dependencies

Required:

* ROS 2 Jazzy (`ros-base` or `desktop`) with `cv_bridge`, `image_transport`,
  `sensor_msgs`, `geometry_msgs`, `tf2_ros`, `vision_msgs`.
* CUDA toolkit (ships with JetPack), including the nvjpeg development
  headers (`libnvjpeg-dev-13-2`).
* VPI development package (`nvidia-vpi-dev`) for the GPU image stages.
  A full `nvidia-jetpack` install already includes both:
  `nvidia-jetpack-dev` → `nvidia-vpi-dev` → `libnvvpi4`/`vpi4-dev`, and
  `cuda-libraries-dev-13-2` → `libnvjpeg-dev-13-2`.
* `cmake`, a C++ compiler, `python3-dev`, `python3-numpy`, `python3-scipy`,
  `python3-yaml`, OpenCV Python bindings.
* `git` and `git-lfs`.

Install everything that is missing:

```bash
sudo apt update
sudo apt install -y \
    git-lfs cmake build-essential python3-dev \
    ros-jazzy-vision-msgs \
    nvidia-vpi-dev libnvjpeg-dev-13-2 \
    python3-opencv python3-numpy python3-scipy python3-yaml
git lfs install --skip-repo
```

Notes:

* `ros-jazzy-vision-msgs` is a hard dependency of
  `apriltag_detector_node` (it publishes `vision_msgs/Detection3DArray`).
  A freshly flashed Jetson does **not** have it.
* `git-lfs` is what makes the cuAprilTags static library download real
  bytes instead of a 132-byte pointer (see §12.1).
* `nvidia-vpi-dev` / `libnvjpeg-dev-13-2` are only needed for
  `image_pipeline: "cuda"` (the `-13-2` suffix tracks your CUDA version;
  adjust if you run a different JetPack). Verify with `ls /opt/nvidia/vpi4`
  and `ls /usr/local/cuda/include/nvjpeg.h`. Without VPI the build skips
  `librace_auv_image_cuda.so` (see §5 and the GPU troubleshooting
  section in §12).
* `python3-apriltag` (apt) is **not** usable for the `python` backend:
  version 3.3.0 lacks `estimate_tag_pose`, so every pose would come back
  as identity. See §8.1 for the correct install.

### 2.1 If you cannot use `sudo` (git-lfs without root)

```bash
cd /tmp
apt-get download git-lfs
mkdir -p "$HOME/.local/git-lfs-extract" "$HOME/.local/bin"
dpkg-deb -x git-lfs_*.deb "$HOME/.local/git-lfs-extract"
cp "$HOME/.local/git-lfs-extract/usr/bin/git-lfs" "$HOME/.local/bin/"
chmod +x "$HOME/.local/bin/git-lfs"

# Persist in every new shell:
grep -q 'local/bin' "$HOME/.bashrc" || \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
export PATH="$HOME/.local/bin:$PATH"

git lfs version
git lfs install --skip-repo
```

`scripts/setup_third_party.sh` checks for `git-lfs` on `PATH` and fails
with a clear message if it is missing.

### 2.2 Make `nvcc`/CMake see CUDA

`/usr/local/cuda` points at CUDA 13.2, and CMake's `FindCUDAToolkit`
finds it. If a build ever reports CUDA as missing:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda-13.2
```

The shim is compiled with `g++` and only uses CUDA headers plus
`libcudart`; `nvcc` is not needed on the happy path.

---

## 3. Workspace layout

```
~/ros2_ws/src/
├── race_auv/                          # main repo (branch: jazzy-devel-perception)
│   ├── race_auv_perception/           # camera + apriltag CUDA packages (submodule)
│   └── race_auv_sim/                  # sim utilities (submodule, not needed here)
├── dwe_camera/                        # dwe_camera_driver + dwe_camera_interfaces
└── mvp_msgs/                          # MVP message definitions
```

Confirm the branch and submodule:

```bash
git -C ~/ros2_ws/src/race_auv status
git -C ~/ros2_ws/src/race_auv branch --show-current
# -> jazzy-devel-perception
git -C ~/ros2_ws/src/race_auv submodule update --init race_auv_perception
```

---

## 4. Fetch NVIDIA cuAprilTags (once per clone)

The CUDA detector links NVIDIA's prebuilt `libcuapriltags.a`, which is a
Git LFS object inside the `isaac_ros_nitros` repository. It is wired in
as a sparse submodule at `third_party/isaac_ros_nitros`.

```bash
cd ~/ros2_ws/src/race_auv/race_auv_perception
git submodule status
scripts/setup_third_party.sh
```

The script:

1. `git submodule update --init` for `third_party/isaac_ros_nitros`
   (with `GIT_LFS_SKIP_SMUDGE=1` so unrelated LFS blobs are not pulled),
2. cone-mode sparse checkout of only `isaac_ros_nitros/lib/cuapriltags`,
3. `git lfs pull --include="...cuapriltags/**"` to fetch just those blobs.

A full checkout of `isaac_ros_nitros` would pull hundreds of MB of
cuVSLAM/cuMotion LFS objects; the sparse cone avoids that.

Verify the library is real (not an LFS pointer):

```bash
ls -l third_party/isaac_ros_nitros/isaac_ros_nitros/lib/cuapriltags/\
lib_aarch64_jetpack61/libcuapriltags.a
# expect ~1.26 MB. A 132-byte file is a pointer -> re-run the script.
```

`third_party/COLCON_IGNORE` keeps colcon from trying to build the
vendored tree.

---

## 5. Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-up-to race_auv_apriltag_cuda race_auv_camera_pkg race_auv_bringup \
    --event-handlers console_direct+
source install/setup.bash
```

What gets built / installed:

| Package | Type | Installs |
|---|---|---|
| `race_auv_apriltag_cuda` | `ament_cmake` | `librace_auv_apriltag_cuda.so` (cuAprilTags) **and** `librace_auv_image_cuda.so` (nvjpeg + VPI), both in `share/race_auv_apriltag_cuda/lib/`, plus the shim headers |
| `race_auv_camera_pkg` | `ament_python` | `apriltag_detector_node`, `apriltag_fuser_node`, `apriltag_cuda.py` |
| `race_auv_bringup` | `ament_cmake` | configs and launch files |

`librace_auv_image_cuda.so` is only built when VPI is found. During
configure you should see:

```
-- race_auv_apriltag_cuda: VPI found, building race_auv_image_cuda
```

If instead it warns *"VPI not found; race_auv_image_cuda ... will not be
built"*, install the VPI/nvjpeg packages from §2 and rebuild. Verify both
libraries exist:

```bash
ls install/race_auv_apriltag_cuda/share/race_auv_apriltag_cuda/lib/
# librace_auv_apriltag_cuda.so  librace_auv_image_cuda.so
```

Platform note: the CMake selects the prebuilt cuAprilTags library by
architecture — `lib_aarch64_jetpack61` on Jetson,
`lib_x86_64_cuda_12_6` on x86_64 dev boxes. Override the root with
`-DCUAPRILTAGS_ROOT=/path/to/cuapriltags`. VPI is Jetson-only, so the
GPU image library is skipped on x86_64 hosts and the node falls back to
`image_pipeline: "cpu"`.

If you change source Python files later, rebuild just that package:

```bash
colcon build --packages-select race_auv_camera_pkg && source install/setup.bash
```

---

## 6. Verify the detector before running the robot

### 6.1 Shim smoke test (go/no-go for CUDA)

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
python3 src/race_auv/race_auv_perception/race_auv_apriltag_cuda/test/cuapriltags_smoke.py
```

Expected output ends with:

```
detected id=0 t=[-3.14070348e-04 -3.14070348e-04  5.02512574e-01] det(R)=1.0000
blank image -> 0 detections (ok)
PASS
```

This renders a synthetic `tag36h11` and checks id, translation
(expected ≈ 0.50 m) and rotation. If it fails with a link/loader error,
the CUDA 12-built library is incompatible with your runtime — switch to
the `python` backend (§10) or check §12.2.

### 6.2 Python wrapper test

```bash
cd ~/ros2_ws
source install/setup.bash
python3 - <<'PY'
import cv2, numpy as np
from race_auv_camera_pkg.apriltag_cuda import CuAprilTagDetector

class Log:
    def info(self, m): print(m)
    def warn(self, m): print("warn:", m)
    def error(self, m): print("error:", m)

aruco = cv2.aruco
d = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
marker = aruco.drawMarker(d, 0, 200)
img = np.full((360, 360), 255, np.uint8)
img[80:280, 80:280] = marker
bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

det = CuAprilTagDetector(
    id_to_size={("tag36h11", 0): 0.125},
    camera_intrinsics={"fx": 800.0, "fy": 800.0, "cx": 180.0, "cy": 180.0},
    image_size={"img_width": 360, "img_height": 360},
    logger=Log(),
)
dets = det.detect(bgr)
assert len(dets) == 1 and abs(dets[0]["T"][2, 3] - 0.5) < 0.05, dets
print("WRAPPER OK:", dets[0]["tag_id"], round(float(dets[0]["T"][2, 3]), 3))
det.close()
PY
```

### 6.3 Optional backend parity

If the upstream apriltag binding is installed (§8.1), a 12.5 cm tag at
0.5 m should agree between backends within a few millimetres and a
fraction of a degree. On the verified machine: 2.55 mm / 0.000°.

### 6.4 GPU image pipeline smoke test (nvjpeg + VPI)

Only relevant when `image_pipeline: "cuda"`. Checks decode parity,
device-memory detection and nvjpeg encode without ROS:

```bash
cd ~/ros2_ws
source install/setup.bash
python3 src/race_auv/race_auv_perception/race_auv_apriltag_cuda/test/gpu_image_smoke.py
```

Expected:

```
rectify parity: mean=0.008 p99=0.0
device detections: [9]
tag 9 z = 0.239 m
encode: 20591 bytes, (720, 1280, 3)
PASS
```

---

## 7. Run on the real robot (two Stellar cameras)

### 7.1 Power / clocks

```bash
sudo nvpmodel -m 0        # MAXN (or -m 2 for 25 W)
sudo jetson_clocks        # lock clocks to max
```

Verify with `jtop` or `tegrastats`.

### 7.2 Launch

`bringup_camera_perception.launch.py` starts the two camera drivers
(`multi_camera.launch.py`), then -- after a few seconds -- the
detectors (`apriltag_detection.launch.py`, which reads
`race_auv_bringup/config/apriltag.yaml`).

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch race_auv_bringup bringup_camera_perception.launch.py
```

Startup order is staggered on purpose:

```
t = 0 s   stellar_camera_node_1   (/dev/video0)
t = 0 s   stellar_camera_node_2   (/dev/video2)
t = 3 s   apriltag_detector_cam_front
t = 3 s   apriltag_detector_cam_down
```

`multi_camera.launch.py` lists `stellar_camera_node_1` (`/dev/video0`)
first, which works around a UVC probe race in `dwe_camera_driver`; see
`Jetson.md` §4.1. The detectors are then delayed 3 s so both cameras
have started publishing before anything subscribes.

Note: the CUDA detector/image backends (`detector_backend: "cuda"`,
`image_pipeline: "cuda"`) run through in-process cuAprilTags/nvjpeg/VPI
shims, not `cv2.cuda` -- there is no OpenCV PYTHONPATH override to
worry about here.

### 7.3 Topics

| Topic | Type | Content |
|---|---|---|
| `/race/stellar2/image/compressed` | `sensor_msgs/CompressedImage` | cam_front raw feed |
| `/race/stellar1/image/compressed` | `sensor_msgs/CompressedImage` | cam_down raw feed |
| `/cam_front/apriltag_detection/detections3d` | `vision_msgs/Detection3DArray` | `Detection3D.id = "tag36h11:<id>"`, pose in camera frame |
| `/cam_front/apriltag_detection/image` | `sensor_msgs/CompressedImage` | annotated JPEG (boxes, axes, xyz/rpy) |
| `/cam_down/apriltag_detection/...` | same | cam_down outputs |

### 7.4 Verify it is working

```bash
# camera feeds alive?
ros2 topic hz /race/stellar2/image/compressed
ros2 topic hz /race/stellar1/image/compressed

# detections published at the configured rate?
ros2 topic hz /cam_front/apriltag_detection/detections3d

# one detection message
ros2 topic echo --once /cam_front/apriltag_detection/detections3d

# annotated image for Foxglove / rqt
ros2 topic hz /cam_front/apriltag_detection/image
```

In the detector's startup log, confirm the backend line:

```
[apriltag_detector]: === Pipeline ===
  detector backend: cuda
  rectify backend : cpu
  jpeg   backend  : cpu (cv2.imencode)
  process_scale   : 1.0
  families        : ['tag36h11'] (10 tags)
```

If an AprilTag is in view and nothing is published:

* confirm the tag is `tag36h11` (25h9 is no longer supported);
* confirm its `(id, size)` tuple is in the camera's `tags:` list;
* check `ros2 topic echo /cam_front/apriltag_detection/image` — the
  annotated feed shows boxes for anything the detector decoded, even if
  the id was filtered out.

### 7.5 Foxglove / topside

`apriltag_detection/image` is a standard `CompressedImage`; `foxglove_bridge`
relays it unchanged. Set the panel's JPEG quality to match `jpeg_quality`
in the YAML for best results.

The in-driver AprilTag processor (`dwe_camera_driver` parameter
`apriltag.enable`, default `false`) is intentionally not enabled by this
launch file; leave it off to avoid duplicate detection topics.

### 7.6 Single-camera test (USB/UVC, e.g. exploreHD)

For bench testing one camera without the full robot stack, use the
dedicated config-driven launch:

```bash
ros2 launch race_auv_bringup explore_cam_apriltag.launch.py
```

Everything is read from
`race_auv_bringup/config/explore_cam_apriltag.yaml`: video id/format,
intrinsics, V4L2 controls, detector backend, tick rate and tag list.
`tags: []` (the default) means *all* tags — the detector falls back to
the global tag list in `config/apriltag.yaml`. The only launch argument
is `config:=<path>` to use a different YAML.

The shipped explore config uses the GPU image pipeline
(`image_pipeline: "cuda"`: nvjpeg + VPI). If the VPI/nvjpeg packages
from §2 are missing, set `image_pipeline: "cpu"` in the config (the node
also warns and falls back automatically if `librace_auv_image_cuda.so`
was not built).

Default topics: `/explore/image/compressed`,
`/explore/apriltag_detection/detections3d`,
`/explore/apriltag_detection/image`.

---

## 8. Run the simulation

The simulator publishes raw `Image` + `CameraInfo`, and its config uses
`detector_backend: "python"` (apt's 4.6/4.8 OpenCV has no CUDA, and the
sim runs on dev boxes too). That backend needs the upstream apriltag3
binding with pose estimation.

### 8.1 Install apriltag3 (only needed for `python` / simulation)

```bash
sudo apt install -y cmake build-essential python3-dev python3-numpy

git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release
sudo cmake --build build --target install
sudo ldconfig

# Upstream installs the Python wrapper one directory deeper than
# libapriltag.so.3, so symlink it next to the wrapper:
PY_SITE=/usr/local/lib/python3.12/site-packages
sudo ln -sf ../../libapriltag.so.3 "$PY_SITE/libapriltag.so.3"

python3 -c "from apriltag import apriltag; print('apriltag3 OK')"
```

Without root, install to a prefix and export both paths:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$HOME/.local/apriltag-upstream"
cmake --build build -j"$(nproc)"
cmake --install build

export PYTHONPATH="$HOME/.local/apriltag-upstream/lib/python3.12/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$HOME/.local/apriltag-upstream/lib:$LD_LIBRARY_PATH"
python3 -c "from apriltag import apriltag; print('apriltag3 OK')"
```

> Do **not** use `sudo apt install python3-apriltag`: Ubuntu's 3.3.0
> binding has only `detect()` and no `estimate_tag_pose`. Against that
> package every detection gets an identity pose.

### 8.2 Launch

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch race_auv_bringup bringup_simulation.launch.py
```

The sim station currently carries:

| Tags | Family | Size |
|---|---|---|
| 146, 176, 185 | `tag36h11` | 15 cm |
| 541, 558 | `tag36h11` | 5 cm |

`apriltag_fuser_node` fuses the per-camera detections and publishes the
fused `race_station/dock_point` TF plus
`race_station/dock_point/pose` (`PoseStamped`).

To run the sim with CUDA on a dev box that has a GPU, set
`detector_backend: "cuda"` (and `cuda_nominal_size: 0.15`) in
`race_auv_bringup/config/simulation/apriltag.yaml`.

---

## 9. Configuration reference — `race_auv_bringup/config/apriltag.yaml`

### 9.1 `detector_defaults`

| Key | Default | Meaning |
|---|---|---|
| `detector_backend` | `"cuda"` (hw) / `"python"` (sim) | detector implementation |
| `image_pipeline` | `"cuda"` (hw) / `"cpu"` (sim) | image stages: nvjpeg decode + VPI CUDA rectify + nvjpeg encode, or CPU |
| `gpu_rectify_interval` | `4` | VPI warp-grid spacing (power of two) for the GPU rectifier |
| `cuda_tile_size` | `4` | cuAprilTags adaptive-threshold window |
| `cuda_nominal_size` | `0.125` | size the CUDA detector is created with; per-tag translations are rescaled |
| `cuda_max_tags` | `64` | max detections per frame |
| `publish_rate` | `15.0` (hw) | detector tick rate (Hz) |
| `min_edge_dist` | `10` | drop detections within N px of the rectified border |
| `nthreads`, `quad_decimate`, `quad_sigma`, `refine_edges` | — | **python backend only** |

`image_pipeline: "cuda"` requires `detector_backend: "cuda"` and
`image_transport: "compressed"`; otherwise the node logs a warning and
falls back to the CPU image path. The GPU image stages need VPI and
nvjpeg — a full JetPack install provides both (`nvidia-vpi-dev`,
`libnvjpeg-dev-13-2`; see §2) — and the node needs
`librace_auv_image_cuda.so`, which is built only when VPI is present
(§5).

### 9.2 Per-camera keys

| Key | Example | Meaning |
|---|---|---|
| `name` / `enabled` | `cam_front` / `true` | camera identity / on-off |
| `detector_namespace` | `cam_front` | ROS namespace for detector pubs |
| `driver:` | `node_name`, `product_name`, remaps | `dwe_camera_driver` block |
| `image_transport` | `compressed` | `compressed` or `raw` |
| `image_topic` | `/race/stellar2/image/compressed` | input feed |
| `info_topic` | `""` | optional `CameraInfo`; empty → YAML intrinsics |
| `camera_frame` | `race_auv/cam_front` | frame id stamped on detections |
| `output_image_topic` | `apriltag_detection/image` | annotated JPEG |
| `output_detections_topic` | `apriltag_detection/detections3d` | `Detection3DArray` |
| `tags` | list of `{id, family, size}` | per-camera override of the global list |
| `intrinsics` | `fx, fy, cx, cy, width, height, distortion, fisheye` | fallback when `info_topic` is empty |
| `process_scale` | `1.0` | detect on a downscaled image; **CUDA: restart required to change** |
| `jpeg_quality` | `90` | annotated image quality |

### 9.3 Tag list

```yaml
tags:
  - { id: 9,  family: "tag36h11", size: 0.125 }
  - { id: 22, family: "tag36h11", size: 0.125 }
```

* `size` is the physical side length in metres and must match the tag.
* IDs are per-family; the CUDA backend ignores everything except
  `tag36h11`.
* For the CUDA backend, translations are rescaled from
  `cuda_nominal_size`:
  `t_true = t_reported * (size / cuda_nominal_size)`.
  Set `cuda_nominal_size` to a tag size that is commonly present; the
  default (0.125) matches the station's medium tags.

---

## 10. Switching backends / rollback

To disable CUDA (e.g. if the prebuilt library will not run on a future
JetPack), set the backend to `python` for the affected cameras or for
all of them:

```yaml
detector_defaults:
  detector_backend: "python"
```

This requires the upstream apriltag3 binding (§8.1) and currently only
works for tags present in the list; re-add any `tag25h9` entries if the
physical tags exist on your station.

To point at a shim built elsewhere without rebuilding the workspace:

```bash
export RACE_AUV_APRILTAG_CUDA_LIB=/path/to/librace_auv_apriltag_cuda.so
```

---

## 11. Performance and tuning

Measured on this Orin Nano, full 1600×1200 frame, including the
host-to-device copy:

| Stage | Value |
|---|---|
| cuAprilTags detect (blank frame) | ~23 ms/frame |
| cuAprilTags detect (one tag) | ~23 ms/frame |
| Implied headroom | ~43 fps / camera |

Measured with `image_pipeline: "cuda"` at 1920x1080 (full frame):

| Stage | CPU path | GPU path |
|---|---:|---:|
| JPEG decode | 25.4 ms | ~10-11 ms (nvjpeg) |
| Fisheye rectify | 14-22 ms | ~0.9-1.9 ms (VPI, device-resident) |
| cuAprilTags detect @1920x1080 | 41 ms | 41 ms (no H2D copy) |
| cuAprilTags detect @960x540 | 8 ms | 8 ms |
| JPEG encode (annotated) | ~18 ms | ~2 ms (nvjpeg) |
| Full chain decode+rectify+detect@0.5+encode | ~99 ms | **~17 ms** |

With `image_pipeline: "cuda"` the CPU cores are free during
decode/rectify/encode and the detection-resolution frame never leaves
the device. Knobs:

* `process_scale` — detect on a downscaled image. The CUDA detector is
  created for the processed size, so changing it requires a node
  restart. Start at `1.0`; try `0.75`/`0.5` for more margin (the GPU
  rectifier produces the downscaled detection frame directly).
* `publish_rate` — detector tick rate (hardware default 15 Hz).
* `jpeg_quality` — annotated-stream bandwidth vs. detail.
* `gpu_rectify_interval` — VPI warp-grid spacing; `4` is the tested
  default, `1` is densest/most accurate and slowest.

The detector logs no per-stage timing; measure with `tegrastats` and
`ros2 topic hz` while watching `top`/`jtop`.

---

## 12. Troubleshooting

### 12.1 `git lfs` / 132-byte `libcuapriltags.a`

**Symptom:** build fails with `libcuapriltags.a: file format not
recognized`, or `scripts/setup_third_party.sh` exits with "missing or
still an LFS pointer".

**Cause:** the submodule was checked out without `git-lfs`, so the `.a`
is a text pointer.

**Fix:**

```bash
sudo apt install git-lfs        # or the user-local route in §2.1
git lfs install --skip-repo
cd ~/ros2_ws/src/race_auv/race_auv_perception
scripts/setup_third_party.sh
```

### 12.2 CMake cannot find CUDA (or `libcuapriltags.a`)

**Not an error anymore.** `CUDAToolkit` is optional: on hosts without a
CUDA toolkit (e.g. the sim computer) `colcon build` still succeeds and
`race_auv_apriltag_cuda` is installed without the native shims. The
detector node then falls back to `detector_backend: "python"` /
`image_pipeline: "cpu"` automatically.

To get the CUDA backend on a Jetson, make CMake find the toolkit and
rebuild:

```bash
which nvcc || export PATH=/usr/local/cuda/bin:$PATH
export CUDAToolkit_ROOT=/usr/local/cuda-13.2
cd ~/ros2_ws/src/race_auv/race_auv_perception
scripts/setup_third_party.sh
cd ~/ros2_ws && colcon build --packages-select race_auv_apriltag_cuda
```

### 12.3 `race_at_create` fails / link errors against `libcuapriltags.a`

**Symptom:** smoke test fails at `nvCreateAprilTagsDetector` or the
build fails linking CUDA registration symbols.

**Cause:** the prebuilt library targets JetPack 6.1 / CUDA 12. If the
link fails, force a CUDA device link by building the shim with
`enable_language(CUDA)` (needs `nvcc` on PATH); if the runtime fails,
fall back to `detector_backend: "python"`.

### GPU image pipeline (`image_pipeline: "cuda"`) issues

* The node logs *"image_pipeline='cuda' requires ..."* and uses the CPU
  path: `image_pipeline: "cuda"` needs `detector_backend: "cuda"` **and**
  `image_transport: "compressed"`.
* `librace_auv_image_cuda.so` missing: the CMake configure warned
  *"VPI not found; race_auv_image_cuda ... will not be built"*. Install
  the GPU packages and rebuild:

  ```bash
  sudo apt install nvidia-vpi-dev libnvjpeg-dev-13-2
  ls /opt/nvidia/vpi4 && ls /usr/local/cuda/include/nvjpeg.h
  colcon build --packages-select race_auv_apriltag_cuda
  ```

  The library is intentionally skipped on non-Jetson (x86_64) hosts.
* `Failed to open PVA device node` on stderr is harmless — VPI tries
  PVA at init and falls back to CUDA.
* If the GPU pipeline errors at runtime, the node logs
  *"GPU image pipeline failed: ..."* once per tick; set
  `image_pipeline: "cpu"` to fall back immediately.

### 12.4 `ModuleNotFoundError: No module named 'vision_msgs'`

```bash
sudo apt install ros-jazzy-vision-msgs
```

### 12.5 `No module named 'apriltag'` or identity poses

* Missing binding → install upstream apriltag (§8.1).
* Binding present but every pose is identity, rotation ~180° from
  expected → you installed Ubuntu's `python3-apriltag` 3.3.0, which has
  no `estimate_tag_pose`. Remove it and install upstream:

```bash
sudo apt remove python3-apriltag
# then §8.1
```

### 12.6 Detector starts but never publishes detections

* The detector only publishes after the pipeline is built. With an
  empty `info_topic` this happens immediately from YAML intrinsics;
  with an `info_topic` it waits for the first `CameraInfo`.
* `process_scale` mismatch: the CUDA detector logs
  `created for WxH but received WxH; skipping` if the rectified size
  changes under it (restart the node).
* CUDA device busy / no device: watch for `cudaMalloc failed` or
  `cuAprilTags detector creation failed` in the node log.
* Tag is `tag25h9` → silently filtered; use `tag36h11`.

### 12.7 `Failed to parse parameter override rule` on the CLI

ROS 2 rejects empty values (`-p info_topic:=""`) and some JSON/flow
values on the command line. Use a params file for complex types:

```bash
ros2 run race_auv_camera_pkg apriltag_detector_node --ros-args \
    --params-file /path/to/params.yaml
```

### 12.8 Camera-open race (`/dev/video2`)

Symptom and full explanation: `Jetson.md` §4.1. The launch file already
stagger-sorts drivers by `driver.node_name` and inserts 2 s between
spawns. Don't reorder the `cameras:` list to "fix" it — the sort is
intentional.

### 12.9 `rclpy exc_info` crash / `power_line_frequency` warning

Both are known `dwe_camera_driver` issues and do not affect detection;
see `Jetson.md` §4.2 and §4.3.

---

## 13. Verification checklist

Copy/paste as you go:

- [ ] `cat /etc/nv_tegra_release` shows R39.x, `dpkg-query -W nvidia-jetpack` shows 7.2.x
- [ ] `git lfs version` works (system or user-local)
- [ ] `dpkg -l ros-jazzy-vision-msgs` (or `python3 -c "import vision_msgs"`)
- [ ] `git -C ~/ros2_ws/src/race_auv/race_auv_perception submodule status` shows the nitros submodule
- [ ] `race_auv_perception/scripts/setup_third_party.sh` prints `cuapriltags ready: ...`
- [ ] `ls -l .../lib_aarch64_jetpack61/libcuapriltags.a` ≈ 1.26 MB
- [ ] `colcon build --packages-up-to ...` finishes with 3 packages
- [ ] `ls .../share/race_auv_apriltag_cuda/lib/` lists **both** `librace_auv_apriltag_cuda.so` and `librace_auv_image_cuda.so` (GPU path)
- [ ] `ls /opt/nvidia/vpi4` and `ls /usr/local/cuda/include/nvjpeg.h` (GPU path, §2)
- [ ] `python3 .../test/cuapriltags_smoke.py` prints `PASS`
- [ ] `python3 .../test/gpu_image_smoke.py` prints `PASS` (GPU path)
- [ ] `ros2 launch race_auv_bringup bringup_camera_perception.launch.py` starts 4 processes
- [ ] Detector log shows `detector backend: cuda` and `image  backend  : cuda` (GPU path)
- [ ] `ros2 topic hz /cam_front/apriltag_detection/detections3d` is non-zero with a tag in view
- [ ] Annotated image visible in Foxglove
- [ ] (Sim) `ros2 launch race_auv_bringup bringup_simulation.launch.py`; dock point TF appears

---

## 14. File map

```
race_auv_perception/                    # this repository
├── race_auv_apriltag_cuda/             # in-process CUDA shims
│   ├── CMakeLists.txt                  # picks lib_aarch64_jetpack61 / x86_64; CUDA/VPI optional
│   ├── package.xml
│   ├── include/race_auv_apriltag_cuda/
│   │   ├── cuapriltags_shim.h          # cuAprilTags C ABI (host + device detect)
│   │   └── image_cuda_shim.h           # nvjpeg decode/encode + VPI rectify
│   ├── src/
│   │   ├── cuapriltags_shim.cpp
│   │   └── image_cuda_shim.cpp
│   └── test/
│       ├── cuapriltags_smoke.py        # §6.1
│       └── gpu_image_smoke.py          # §6.4
├── race_auv_camera_pkg/
│   ├── race_auv_camera_pkg/
│   │   ├── apriltag_detector_node.py   # per-camera node (detector + image backends)
│   │   ├── apriltag_fuser_node.py      # sim multi-camera fuser
│   │   ├── apriltag_cuda.py            # ctypes wrappers: CuAprilTagDetector + GpuImagePipeline
│   │   ├── apriltag_processor.py       # apriltag3 backend + shared annotate
│   │   ├── apriltag_geom.py            # rotation sanitize / message helpers
│   │   ├── urdf_tag_parser.py
│   │   └── image_processing.py         # CPU rectifier (+ maps for the GPU rectifier)
│   ├── Jetson.md                       # quick reference + upstream driver issues
│   └── JETSON_ORIN_NANO_SETUP.md       # this file
├── third_party/isaac_ros_nitros/       # sparse submodule (cuapriltags only)
└── scripts/setup_third_party.sh

# Still in the main race_auv repository:
race_auv_bringup/
├── config/apriltag.yaml                # real hardware (Stellar, GPU image pipeline)
├── config/simulation/apriltag.yaml     # simulation (raw images, CPU pipeline)
├── config/explore_cam_apriltag.yaml    # single exploreHD/USB camera test (§7.6)
├── launch/explore_cam_apriltag.launch.py
├── launch/bringup_camera_perception.launch.py   # multi_camera + apriltag_detection
└── launch/include/
    ├── multi_camera.launch.py          # 2x dwe_camera_driver camera_node
    ├── apriltag_detection.launch.py    # 2x apriltag_detector_node
    └── simulation/apriltag_sim.launch.py
```

---

## 15. Known limitations

* **`tag36h11` only on CUDA.** The four 21 cm `tag25h9` tags on hardware
  and three 27 cm ones in sim are no longer detected. Migrate physical
  tags to `tag36h11` (and update the URDF/SCN) if long-range detection
  is needed, or run a mixed-backend setup.
* **Fixed detector size.** The CUDA detector is created for one image
  size at startup; changing `process_scale` or the camera resolution
  requires a node restart.
* **Prebuilt binary vintage.** `libcuapriltags.a` is a JetPack 6.1 /
  CUDA 12 build redistributed by NVIDIA; it is not built from source
  here. The `python` backend is the escape hatch.
* **GPU image stages are Jetson-only.** `image_pipeline: "cuda"` needs
  JetPack VPI (`nvidia-vpi-dev`) and CUDA nvjpeg
  (`libnvjpeg-dev-13-2`); on x86_64 hosts the image library is skipped
  and the node must use the CPU image path. The nvjpeg/VPI output matches
  the CPU path within a few grey levels (JPEG IDCT differences), so tag
  IDs and poses are unaffected.
* **CPU fallback cost.** With `image_pipeline: "cpu"`, fisheye
  undistortion and JPEG encode run per frame at full resolution and are
  the dominant cost; use `process_scale < 1` or the GPU path when the
  CPU is saturated.
