# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-time scene geometry calibration: lock camera / robot / tooth layout.

This script captures the fixed spatial relationships that must remain stable
between data-collection / training sessions of the ACT policy:

  1. **Base marker pose in the robot-base frame** (T_base_robotbase_marker):
     averaged over N frames while nothing in the scene is moving. This
     defines the world reference for azimuth computation.
  2. **Canonical tooth pose**: the operator places the dental model at the
     canonical start position used during ACT demos, then the script
     records the tooth marker pose in the base marker frame and the
     corresponding shoulder_pan reading. The align_base controller will
     drive the arm back to this shoulder_pan every run, so the policy
     always sees the same initial visual input.

Outputs (SCENE_GEOMETRY_FILE, JSON):
  - base_marker_pose_mean: 4x4 T_cam_base averaged over N frames (list).
  - base_marker_pose_std_rot_deg / std_trans_mm: per-frame dispersion,
    so the user can verify the scene was actually static.
  - canonical_tooth_in_base: 4x4 T_base_tooth at the canonical pose.
  - canonical_azimuth_deg: atan2(y,x) of the canonical tooth position.
  - canonical_shoulder_pan: joint reading captured alongside.
  - camera_index, marker_size_m, base/tooth ids: provenance for audit.

Usage:
    python -m dental_robot.calibrate_scene_geometry

Procedure:
  - Mount the camera, robot and base marker; do not move them afterwards.
  - Run the script. It will sample BASE_SAMPLE_FRAMES frames for the base
    marker pose, then prompt you to place the tooth model at the canonical
    start pose and press SPACE to capture the canonical pair.
  - The script verifies that the pan-mapping file exists (needed later by
    align_base) but does not refit it; run calibrate_pan_mapping first.

NOTE: this calibration must be redone whenever the camera, robot base, or
base marker is physically moved. The ACT policy is only valid for the
geometry recorded here.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from dental_robot.aruco_locator import ArucoLocator, invert_rigid_transform
from dental_robot.config import (
    BASE_MARKER_ID,
    MARKER_SIZE_M,
    PAN_MAPPING_FILE,
    SCENE_CAM_INDEX,
    SCENE_GEOMETRY_FILE,
    TOOTH_MARKER_ID,
    connect_follower,
    make_follower_config,
    make_scene_camera_config,
)
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

# Number of frames averaged for the base-marker pose. At 30 FPS this is
# ~3 s of sampling; sufficient for sub-millimetre stability on a static
# scene while keeping the calibration step brief.
BASE_SAMPLE_FRAMES = 100
# Same, for the canonical tooth pose (the model is held by hand briefly,
# so we sample fewer frames once the operator confirms it is steady).
TOOTH_SAMPLE_FRAMES = 30


def _pose_to_list(pose: np.ndarray) -> list:
    return np.asarray(pose, dtype=np.float64).tolist()


def _geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    delta = R1 @ R2.T
    trace = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def _sample_marker_pose(
    locator: ArucoLocator, camera: OpenCVCamera, marker_id: int, n_frames: int
) -> tuple[np.ndarray, float, float]:
    """Return (mean_pose, std_rot_deg, std_trans_mm) over n_frames.

    Resets the temporal smoother first so the mean is not biased by prior
    state. Drops frames where the marker is not visible; raises RuntimeError
    if fewer than half the frames succeed.
    """
    locator.reset_temporal_state()
    poses: list[np.ndarray] = []
    print(f"  sampling {n_frames} frames for marker {marker_id}...")
    for i in range(n_frames):
        frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = locator.detector.detectMarkers(gray)
        if ids is None or marker_id not in ids.flatten():
            time.sleep(0.02)
            continue
        idx = int(np.where(ids.flatten() == marker_id)[0][0])
        raw = locator._solve_marker_pose(corners[idx], marker_id=marker_id)  # noqa: SLF001
        if raw is None:
            time.sleep(0.02)
            continue
        poses.append(raw)
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{n_frames}  ({len(poses)} valid)")

    if len(poses) < n_frames // 2:
        raise RuntimeError(
            f"Marker {marker_id} visible in only {len(poses)}/{n_frames} frames. "
            "Check lighting, marker angle and that nothing is occluding it."
        )

    # Average translation element-wise; average rotation via quaternion mean
    # (approximate: arithmetically average the rotation matrices and
    # re-orthogonalise via SVD -- sufficient for a tightly-clustered set).
    rots = np.stack([p[:3, :3] for p in poses])
    trans = np.stack([p[:3, 3] for p in poses])
    mean_rot = np.mean(rots, axis=0)
    U, _, Vt = np.linalg.svd(mean_rot)
    mean_rot = U @ Vt
    mean_trans = np.mean(trans, axis=0)

    # Dispersion: max pairwise rotation angle and translation std (mm).
    max_rot_deg = 0.0
    for R in rots:
        max_rot_deg = max(max_rot_deg, _geodesic_angle_deg(R, mean_rot))
    std_trans_mm = float(np.max(np.std(trans, axis=0))) * 1000.0

    mean_pose = np.eye(4, dtype=np.float64)
    mean_pose[:3, :3] = mean_rot
    mean_pose[:3, 3] = mean_trans
    return mean_pose, max_rot_deg, std_trans_mm


def main():
    # Pre-flight: pan mapping must already exist, because align_base will
    # later map the canonical azimuth back to shoulder_pan.
    if not PAN_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Pan mapping not found: {PAN_MAPPING_FILE}\n"
            "Run `python -m dental_robot.calibrate_pan_mapping` first."
        )

    locator = ArucoLocator()
    # Calibration needs RAW per-frame poses, not smoothed ones.
    locator.enable_temporal_smoothing(False)
    camera = OpenCVCamera(make_scene_camera_config())
    camera.connect()
    robot = connect_follower(make_follower_config(with_cameras=False))

    print("=" * 60)
    print("Scene geometry calibration")
    print("=" * 60)
    print("Step 1: keep the scene STILL. Sampling base marker pose...")
    try:
        base_mean, base_rot_std, base_trans_std = _sample_marker_pose(
            locator, camera, BASE_MARKER_ID, BASE_SAMPLE_FRAMES
        )
        print(f"  base marker rotation spread: {base_rot_std:.3f} deg")
        print(f"  base marker translation spread: {base_trans_std:.3f} mm")
        if base_rot_std > 1.0 or base_trans_std > 1.0:
            print("  [WARN] base marker pose is not stable. Tighten the mount,")
            print("         improve lighting, then re-run this script.")

        print()
        print("Step 2: place the dental model at the CANONICAL start pose")
        print("        (the same position used for ACT demonstrations).")
        print("        Keep it still, then press SPACE to sample.")
        print("        q = abort.")

        # Wait for SPACE, showing a live debug view.
        while True:
            frame = camera.read()
            display = locator.draw_debug(frame)
            pan = robot.bus.sync_read("Present_Position")["shoulder_pan"]
            cv2.putText(
                display,
                f"shoulder_pan: {pan:+.2f}   SPACE=sample  q=abort",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("scene geometry calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                print("Aborted by user.")
                return

        tooth_mean, tooth_rot_std, tooth_trans_std = _sample_marker_pose(
            locator, camera, TOOTH_MARKER_ID, TOOTH_SAMPLE_FRAMES
        )
        print(f"  tooth marker rotation spread: {tooth_rot_std:.3f} deg")
        print(f"  tooth marker translation spread: {tooth_trans_std:.3f} mm")

        # Record the shoulder_pan alongside (the ACT demos start from this
        # value, so align_base drives back to it every run).
        canonical_pan = float(robot.bus.sync_read("Present_Position")["shoulder_pan"])

        # Canonical tooth pose expressed in the base marker frame.
        tooth_in_base = invert_rigid_transform(base_mean) @ tooth_mean
        cx, cy = float(tooth_in_base[0, 3]), float(tooth_in_base[1, 3])
        canonical_az = float(np.degrees(np.arctan2(cy, cx)))

        geometry = {
            "version": 1,
            "base_marker_id": BASE_MARKER_ID,
            "tooth_marker_id": TOOTH_MARKER_ID,
            "marker_size_m": MARKER_SIZE_M,
            "scene_camera_index": SCENE_CAM_INDEX,
            "base_marker_pose_mean": _pose_to_list(base_mean),
            "base_marker_pose_std_rot_deg": base_rot_std,
            "base_marker_pose_std_trans_mm": base_trans_std,
            "canonical_tooth_in_base": _pose_to_list(tooth_in_base),
            "canonical_azimuth_deg": canonical_az,
            "canonical_shoulder_pan": canonical_pan,
            "tooth_marker_pose_std_rot_deg": tooth_rot_std,
            "tooth_marker_pose_std_trans_mm": tooth_trans_std,
            "notes": (
                "Redo this calibration whenever the camera, robot base, or "
                "base marker is moved. The canonical pose is the ACT policy "
                "start configuration; align_base drives shoulder_pan back to "
                "canonical_shoulder_pan before handing control to the policy."
            ),
        }

        SCENE_GEOMETRY_FILE.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
        print()
        print("=" * 60)
        print(f"Saved scene geometry to {SCENE_GEOMETRY_FILE}")
        print(f"  canonical azimuth:    {canonical_az:+.2f} deg")
        print(f"  canonical shoulder_pan: {canonical_pan:+.2f}")
        print(f"  base pose spread:     {base_rot_std:.3f} deg / {base_trans_std:.3f} mm")
        print(f"  tooth pose spread:    {tooth_rot_std:.3f} deg / {tooth_trans_std:.3f} mm")
        print()
        print("IMPORTANT: do NOT move the camera / robot / base marker")
        print("after this point. The ACT policy is only valid for this layout.")
        print("=" * 60)
    finally:
        camera.disconnect()
        robot.bus.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
