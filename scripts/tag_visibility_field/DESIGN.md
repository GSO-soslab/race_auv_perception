# AprilTag Visibility-Cone Probability Field — Design

**Status:** design only, not yet implemented. No code in this directory yet.

## Context

The docking station carries several AprilTags at known positions. Today that layout is
only used reactively — the multi-camera fuser (`race_auv_camera_pkg/apriltag_fuser_node.py`)
combines *live* detections with the known tag geometry to estimate the dock's pose.
There's no tool that answers the *planning* question: "given only tag placement, tag
size, and water conditions, where in the water column would a tag plausibly be
detectable at all?"

Explicit scope: **no camera FOV/intrinsics, no occlusion modeling, no ROS** — just
tag-centric geometric visibility cones, simple enough to run as a standalone Python
script. Decoupling the tag layout into its own flat YAML (rather than parsing the real
URDF live) also makes this a general "try a hypothetical tag layout, see the coverage"
design tool — useful for evaluating *where to place* tags, not just analyzing the
current station.

This is a first pass; a natural (not-built-here) follow-up is feeding high-probability
regions into `bhv_path_following`'s waypoint format
(`race_auv_bringup/config/go_to_list/*.yaml`: `{frame_id, waypoints: [{x,y,z,u}]}`), or,
separately, wrapping the same math in a thin ROS node (see "Why standalone, not ROS"
below) if this ever needs to sit alongside live system data.

## Location and rationale

`race_auv_perception/scripts/tag_visibility_field/` — matches the existing convention
for standalone, non-ROS-registered tooling in this repo
(`race_auv_perception/scripts/setup_third_party.sh` already lives at this level). No ROS
package, no `setup.py`/`package.xml`/`CMakeLists.txt` changes, no colcon build needed to
use it — just `python3 tag_visibility_field.py ...` with `numpy` + `pyyaml` (no `scipy`,
no `rclpy`).

Planned files (not yet created):
```
race_auv_perception/scripts/tag_visibility_field/
  DESIGN.md                # this file
  config.yaml              # tags + grid + cone model + output settings (one file)
  tag_visibility_field.py  # the whole tool: load config, compute field, write outputs
```

## Why standalone, not ROS/RViz

Considered and rejected (for this pass) publishing `PointCloud2`/`MarkerArray` from a
ROS node and viewing it in RViz/Foxglove. Trade-off:

- **Standalone wins here**: no colcon build, no sourcing, no launch file, no QoS/
  durability gotchas (a ROS approach needs a manual "set Transient Local on the RViz
  display" step just to see a latched topic — an easy-to-miss footgun). Iteration loop
  is `edit config.yaml → rerun script → refresh browser tab`. Output is a single
  self-contained HTML file, shareable with anyone without the ROS workspace at all.
- **ROS/RViz would win** if this needs to sit alongside *live* system data (real vehicle
  pose, live detections, TF tree, bag playback) — a custom viewer can't cheaply grow
  into that. If that need arises later, the fix is a thin ROS wrapper node that reuses
  the same `compute_field()` math to publish `PointCloud2`/`MarkerArray` — not a
  rewrite, since the math here has zero ROS dependency by design.
- The decoupled YAML (vs. the real URDF) trades staying in sync with the real dock for
  the ability to try hypothetical layouts and for zero ROS/package dependency. If it
  ever drifts from the real station and that becomes a problem, an alternate input mode
  that parses `race_station_description/urdf/base.urdf` via the existing
  `race_auv_camera_pkg/race_auv_camera_pkg/urdf_tag_parser.py` (`extract_tag_transforms()`)
  is a natural, low-effort addition.

## Existing building blocks (context, not reused directly)

Tag poses today live in `race_station_description/urdf/base.urdf` (sibling repo
`src/race_station`), parsed by `race_auv_camera_pkg/race_auv_camera_pkg/urdf_tag_parser.py`
(`extract_tag_transforms()` → `{(family, id): TagTransform(T_base_to_tag)}`). Tag sizes
come separately from a YAML `tags:` list (e.g. `race_auv_bringup/config/simulation/apriltag.yaml`),
keyed by `(family, id)`. This design deliberately does NOT reuse that parser — see "Why
standalone, not ROS" above — but the `config.yaml` below was seeded by copying the
current values out of that URDF once, by hand.

## 1. `config.yaml` — flat, decoupled tag layout + model params

All tag poses are relative to `dock_point` (the file's own implicit origin `(0,0,0)`) —
no URDF/joint-chain parsing. A header comment notes these are a manual snapshot, meant
to be edited freely for hypothetical layouts, not kept in lockstep with the real URDF.

```yaml
# Tag layout is a decoupled, hand-edited snapshot (originally copied from
# race_station_description/urdf/base.urdf, poses relative to dock_point).
# Edit freely to try hypothetical layouts -- this file has no link back to
# the real URDF/race_station repo.
tags:
  - { id: 146, family: "tag36h11", size: 0.15, xyz: [0.430, 0.0, 0.405],  rpy: [-1.5708, 0.0, -1.5708] }
  - { id: 176, family: "tag36h11", size: 0.15, xyz: [0.0, 0.25, -0.012],  rpy: [3.1416, 0.0, -1.5708] }
  - { id: 185, family: "tag36h11", size: 0.15, xyz: [0.0, -0.25, -0.012], rpy: [3.1416, 0.0, -1.5708] }
  - { id: 541, family: "tag36h11", size: 0.05, xyz: [0.470, 0.0, 0.160],  rpy: [-1.5708, 0.0, -1.5708] }
  - { id: 558, family: "tag36h11", size: 0.05, xyz: [0.470, 0.0, 0.095],  rpy: [-1.5708, 0.0, -1.5708] }
  - { id: 2,   family: "tag25h9",  size: 0.27, xyz: [-0.81, 0.0, -0.21],  rpy: [-1.5708, 0.0, -1.5708] }
  - { id: 13,  family: "tag25h9",  size: 0.27, xyz: [-0.26, 0.37, -0.21], rpy: [-1.5708, 0.0, 3.1416] }
  - { id: 17,  family: "tag25h9",  size: 0.27, xyz: [-0.26, -0.37, -0.21], rpy: [-1.5708, 0.0, 0.0] }

tag_normal_axis: "+z"     # local axis (post-rotation) pointing out of the tag face.
                          # UNVERIFIED guess (common AprilTag-library convention) --
                          # see Verification #2, the viewer's normal arrows are the check.

grid:
  bounds: { x: [-1.5, 1.5], y: [-1.5, 1.5], z: [-1.0, 2.0] }   # metres, dock_point-relative
  resolution: 0.05                # metres/voxel, isotropic
  max_voxels: 2000000             # hard cap; script raises rather than silently truncating

cone_model:
  range_per_tag_size: 10.0        # max_range (pre-water-cap) = tag size * this
  water:
    attenuation_coefficient: 0.3  # 1/m, Beer-Lambert cap: range_water = -ln(contrast_threshold)/c
    contrast_threshold: 0.03
    jerlov_override: null         # if set, overrides attenuation_coefficient via table below
    jerlov_to_c_table:
      - { jerlov: 0.0, c: 0.05 }
      - { jerlov: 3.0, c: 0.15 }
      - { jerlov: 6.0, c: 0.35 }
      - { jerlov: 9.0, c: 0.60 }
  max_incidence_deg: 40.0
  range_falloff_shape: "cosine"   # "cosine" | "linear" | "smoothstep"
  range_falloff_start_frac: 0.7   # start taper at 70% of max_range
  angle_falloff_power: 1.0        # score *= cos(theta)^power inside the cone

combination:
  rule: "max"                     # "max" | "prob_or" (1-prod(1-p_i)) | "sum_clip"

output:
  probability_threshold: 0.02     # drop points below this from saved data + viewer
  formats: ["npz", "csv"]
  max_render_points: 200000       # random-subsample (fixed seed) the viewer's embedded
                                   # points if the thresholded set exceeds this
  normal_arrow_length: 0.3        # metres, viewer-only debug arrow length
```

Effective max range per tag = `min(size * range_per_tag_size, range_water)`.

## 2. `tag_visibility_field.py` — the whole tool, no ROS

Pure `numpy` + `pyyaml` + stdlib (`json`, `argparse`, `csv`, `math`). No `scipy`: roll/
pitch/yaw → rotation matrix is ~10 lines of numpy trig (`Rz @ Ry @ Rx`), not worth a new
dependency.

```python
def rpy_to_matrix(roll, pitch, yaw) -> np.ndarray
    # Rz(yaw) @ Ry(pitch) @ Rx(roll), matching URDF's <origin rpy="r p y"/>
    # convention (fixed-axis XYZ, i.e. scipy's Rotation.from_euler('xyz', ...)
    # semantics) -- same convention urdf_tag_parser.py already uses, so pasted
    # rpy values from a URDF behave identically here.

def tag_pose(tag_cfg, normal_axis) -> tuple[origin: np.ndarray, normal: np.ndarray]
    # origin = tag_cfg['xyz']; normal = rpy_to_matrix(*tag_cfg['rpy']) @ axis_vector,
    # where axis_vector is +/-e_x/e_y/e_z selected by `normal_axis`.

def resolve_max_range(tag_size, cone_cfg, jerlov_override) -> float
    # size_range = tag_size * cone_cfg['range_per_tag_size']
    # c = lookup(jerlov_override, jerlov_to_c_table) if jerlov_override is not None
    #     else cone_cfg['water']['attenuation_coefficient']
    # water_range = -ln(contrast_threshold) / c
    # return min(size_range, water_range)

def build_grid(bounds, resolution, max_voxels) -> np.ndarray  # (N,3) via np.meshgrid + reshape
    # raises ValueError (not silent truncation) if the implied voxel count > max_voxels

def score_grid_for_tag(points, origin, normal, size, cone_cfg) -> np.ndarray  # (N,)
def combine_scores(per_tag_scores: np.ndarray, rule: str) -> np.ndarray       # (n_tags,N)->(N,)
def compute_field(cfg: dict) -> tuple[points, probability, tags_resolved, warnings]
def probability_to_rgb(p: np.ndarray) -> np.ndarray
    # (N,3) uint8, simple 2-stop lerp blue(low) -> red(high); same colormap
    # used for both any optional CLI plot and the viewer's legend, so the
    # two never show inconsistent colors for the same value.

def write_outputs(points, probability, tags_resolved, cfg, out_dir: Path) -> None
    # .npz: savez_compressed(points=points, probability=probability) (thresholded set)
    # .csv: header "x,y,z,probability", thresholded set
    # viewer.html: thresholded + max_render_points-capped (fixed-seed random
    #   subsample if over the cap) subset, embedded as:
    #     const FIELD = {
    #       points: [[x,y,z], ...],       // capped subset
    #       probability: [p, ...],        // same length/order as points
    #       tags: [{id, family, size, origin: [x,y,z], normal: [x,y,z]}, ...],  // ALL tags, uncapped
    #       normal_arrow_length: <from output.normal_arrow_length, default 0.3>
    #     };

def main() -> None   # argparse: --config, --out-dir (default ./tag_visibility_output),
                      # --jerlov (overrides cfg.cone_model.water.jerlov_override)
                      # prints: grid shape, voxel count, prob min/max/mean, per-tag
                      # point-count above threshold, and each tag's origin+normal in
                      # plain text (pre-viewer sanity check)
```

`score_grid_for_tag`: `r = ||p - origin||`; `cos_theta = dot(unit(p-origin), normal)`;
angle score = `0` if `theta > max_incidence` else `cos_theta ** power`; range score = the
configured falloff shape from 1.0 at `r=0` to 0.0 at `max_range` (taper starting at
`range_falloff_start_frac * max_range`); product of the two, clipped to `[0,1]`.

## 3. `viewer.html` — self-contained browser viewer (generated by the script)

"Real-time" here means **interactively explorable** (orbit/zoom/pan, live threshold
slider) — the field itself is static, computed once from a fixed config, not a live data
feed. One HTML file, data embedded inline, opens directly via `file://` — no server, no
build step.

- **Three.js** loaded via `<script src="https://cdn.jsdelivr.net/npm/three@<pinned>/build/three.min.js">`
  (plus OrbitControls) — since this file is meant to be opened locally (`file://`), it
  is not subject to any Claude-Artifact CDN allowlist; use whatever pinned Three.js
  version/CDN works. (If this viewer is later also published as a Claude Artifact for
  sharing, an allowlist and pinned-version rule would apply then — that's a separate,
  optional step from the file the Python script writes to disk.)
- Data embedded as `<script>const FIELD = {points, probability, tags: [{id, family,
  size, origin, normal}], ...};</script>` via `json.dumps` from Python.
- Scene: `THREE.Points` (BufferGeometry; position from `points`, per-point color via a
  simple blue→red lerp on `probability`), one small box/plane per tag sized to `size`
  and oriented from its rotation (gray, so tags read as distinct from the field), one
  `THREE.ArrowHelper` per tag along its `normal` (**this is the axis-sign verification
  aid**), an `AxesHelper`/`GridHelper` for scale, `OrbitControls` (drag-rotate,
  scroll-zoom, right-drag-pan).
- UI overlay (plain HTML/CSS, no framework): a probability-threshold range slider that
  re-filters the already-embedded points client-side (no recompute), a checkbox to
  toggle tag markers/normal arrows, a small colormap legend.

## Verification plan (once implemented)

1. **Sanity check (stdout, no viewer needed):**
   `python3 tag_visibility_field.py --config config.yaml`
   — point count ≤ `max_voxels`; probability ∈ [0,1]; each tag's printed origin matches
   its `xyz` in `config.yaml`; nonzero-but-not-everything score coverage per tag.

2. **Axis-sign check (mandatory before trusting the field):** open `viewer.html` in a
   browser. Confirm every tag's normal arrow points away from the dock body into open
   water, not into the mounting plate. If backwards, flip `tag_normal_axis` in
   `config.yaml` and rerun. If only some tags are backwards, that means the copied `rpy`
   values are inconsistent per-tag — not fixable by one global flag; add a per-tag
   `normal_axis` override only if this is actually observed.

3. **Field visual check:** in the viewer, confirm the point cloud concentrates in front
   of tag faces, respects `max_incidence_deg`/effective `max_range`, and combines
   overlapping cones per `combination.rule`. Use the threshold slider to inspect the
   high-confidence core vs. the tapered edges.

4. **Water-attenuation sweep:** rerun with `--jerlov 0.0/3.0/6.0/9.0` and confirm
   `max_range` (printed per tag) shrinks monotonically as water gets murkier.

## Constants: best-guess (tune later) vs. structural (design choice)

**Best-guess — validate against real data before trusting quantitatively:**
`range_per_tag_size` (10.0), `water.attenuation_coefficient` (0.3/m),
`contrast_threshold` (0.03), `jerlov_to_c_table` anchors, `max_incidence_deg` (40°),
`range_falloff_start_frac` (0.7), `angle_falloff_power` (1.0), `grid.resolution`/
`max_voxels` (performance, not physics), `output.probability_threshold` (0.02),
`output.max_render_points` (200000, browser performance only).

**Structural (design decisions, not physics):** the existence and default (`"+z"`) of
`tag_normal_axis`, gated by the mandatory viewer-based verification step;
`combination.rule` options and `"max"` as default (a point is "detectable" if *any* tag
sees it well — `prob_or`/`sum_clip` model "how many tags agree," a different question,
available but not default); the separable `range_falloff(r) * angle_falloff(theta)`
score model (simpler to implement/vectorize; real decode probability isn't necessarily
separable); the flat, decoupled `config.yaml` tag layout instead of live URDF parsing
(see "Why standalone, not ROS" above).

## Next step

This document is the full design; `config.yaml` and `tag_visibility_field.py` have not
been written yet. Implement directly from the function signatures and config schema
above when ready to proceed.
