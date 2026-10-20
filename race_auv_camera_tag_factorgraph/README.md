# race_auv_camera_tag_factorgraph

Minimal tag36h11 detector for the factor-graph `dock_fuser` (`race_auv_factorgraph`).
One node per camera: rectify the image (cv2), detect tag36h11 (apriltag3, CPU),
look up each tag's size by ID, drop poor detections, and publish the 4 corners
and the IPPE pose at the image stamp. One message per processed image, also
when no tag is in view (dock_fuser makes one keyframe per message).

| File | What |
|---|---|
| `rectify.py` | Fisheye / pinhole undistortion maps, crop, rectified camera matrix |
| `tags.py` | Detection, per-ID size, image checks, IPPE pose; the canonical corner order |
| `detector_node.py` | ROS node: newest image at `publish_rate`, publishes the outputs below |

## Outputs (in the camera namespace)

| Topic | Type | Content |
|---|---|---|
| `apriltag_detection/tags` | `apriltag_msgs/AprilTagDetectionArray` | 4 corners per tag, rectified image, order `lb, rb, rt, lt` |
| `apriltag_detection/detections3d` | `vision_msgs/Detection3DArray` | IPPE pose `T_cam_tag` per tag |
| `apriltag_detection/camera_info` | `sensor_msgs/CameraInfo` (latched) | Rectified intrinsics; dock_fuser's `cameras.*` must equal them |
| `apriltag_detection/image/compressed` | `sensor_msgs/CompressedImage` | Debug overlay, only with `publish_image: true` |

All stamped with the image's `header.stamp`; `frame_id` = `camera_frame`.

## Image checks

Applied before publishing, so dock_fuser only checks consistency over time:

- `min_edge_px` (10): every corner this far from the rectified border;
- `min_side_px` (15): shortest tag side in the image;
- `max_fit_rms_px` (1.5): RMS error of the best IPPE pose (catches occluded,
  blurred or badly segmented quads).

## Run

```bash
# Vehicle: cameras + this detector (config/tag_factorgraph_detector.yaml in race_auv_bringup)
ros2 launch race_auv_bringup bringup_camera_tag_factorgraph.launch.py

# Re-run on a recorded bag together with dock_fuser
ros2 launch race_auv_factorgraph bag_test.launch.py bag:=<image bag> start_offset:=95 duration:=60

# Tests (rendered tags at known poses; no ROS needed)
python3 -m pytest -q test
```

Requires the apriltag3 Python binding (AprilRobotics/apriltag, `import apriltag`).
