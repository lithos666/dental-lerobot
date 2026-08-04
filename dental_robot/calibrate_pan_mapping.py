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

"""Calibrate the azimuth -> shoulder_pan mapping (one-time, empirical).

Usage:
    python -m dental_robot.calibrate_pan_mapping

Procedure (repeat for 4-6 tooth-model positions across the working sector):
  1. Place the dental model somewhere in the sector.
  2. Jog the base with a/d (hold Shift via A/D for fine steps) until the arm
     is aimed at the model exactly the way your ACT demos start.
  3. Press SPACE to record the (azimuth, shoulder_pan) pair.
Then press f to fit a linear mapping and save it, q to quit without saving.

The mapping is linear because both quantities are rotations about the same
vertical axis; the fit absorbs the camera/marker mounting offset and the
normalized-unit scale of shoulder_pan.

Mount the base marker so the working sector does NOT cross the +/-180 deg
azimuth seam (the fit below is linear and cannot bridge the discontinuity).
"""

import json

import cv2
import numpy as np

from dental_robot.aruco_locator import ArucoLocator
from dental_robot.config import (
    PAN_MAPPING_FILE,
    connect_follower,
    make_follower_config,
    make_scene_camera_config,
)
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

COARSE_STEP = 2.0  # normalized shoulder_pan units per keypress
FINE_STEP = 0.5
# A working sector wider than this suggests samples straddle the +/-180 deg
# seam, where a linear fit is invalid (see module docstring).
MAX_AZIMUTH_SPAN_DEG = 180.0


def fit_pan_mapping(samples: list[tuple[float, float]]) -> dict:
    """Least-squares linear fit pan = slope * azimuth + intercept, with diagnostics."""
    az_arr = np.array([s[0] for s in samples], dtype=np.float64)
    pan_arr = np.array([s[1] for s in samples], dtype=np.float64)

    az_span = float(np.ptp(az_arr))
    if az_span > MAX_AZIMUTH_SPAN_DEG:
        raise ValueError(
            f"Azimuth samples span {az_span:.1f} deg (> {MAX_AZIMUTH_SPAN_DEG:.0f}): they likely "
            "straddle the +/-180 deg seam. Remount the base marker so the working "
            "sector avoids the discontinuity, then recalibrate."
        )
    if az_span < 1.0:
        raise ValueError(
            f"Azimuth samples span only {az_span:.2f} deg: the fit would be ill-conditioned. "
            "Spread the tooth model across the working sector."
        )

    slope, intercept = np.polyfit(az_arr, pan_arr, 1)
    residuals = pan_arr - (slope * az_arr + intercept)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "rmse_pan_units": float(np.sqrt(np.mean(residuals**2))),
        "max_abs_residual_pan_units": float(np.max(np.abs(residuals))),
        "samples": samples,
    }


def main():
    locator = ArucoLocator()
    # Calibration needs RAW per-frame poses, not smoothed ones: smoothing
    # against stale state would bias the (azimuth, pan) samples.
    locator.enable_temporal_smoothing(False)
    camera = OpenCVCamera(make_scene_camera_config())
    camera.connect()
    # No robot cameras here: the scene camera is opened standalone above.
    robot = connect_follower(make_follower_config(with_cameras=False))

    samples = []  # (azimuth_deg, shoulder_pan_normalized)
    print("a/d = jog base, A/D = fine jog, SPACE = record pair, f = fit & save, q = quit")

    try:
        while True:
            frame_rgb = camera.read()
            display = locator.draw_debug(frame_rgb)
            pan = robot.bus.sync_read("Present_Position")["shoulder_pan"]
            cv2.putText(
                display,
                f"shoulder_pan: {pan:+.2f}   samples: {len(samples)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("pan mapping calibration", display)

            key = cv2.waitKey(1) & 0xFF
            step = 0.0
            if key == ord("a"):
                step = COARSE_STEP
            elif key == ord("d"):
                step = -COARSE_STEP
            elif key == ord("A"):
                step = FINE_STEP
            elif key == ord("D"):
                step = -FINE_STEP
            elif key == ord(" "):
                az = locator.averaged_azimuth_deg(camera.read, n_frames=10)
                samples.append((az, pan))
                print(f"Recorded: azimuth={az:+.2f} deg -> pan={pan:+.2f}")
            elif key == ord("f"):
                if len(samples) < 2:
                    print("Need at least 2 samples (4-6 recommended)")
                    continue
                try:
                    mapping = fit_pan_mapping(samples)
                except ValueError as e:
                    print(f"Fit rejected: {e}")
                    continue
                PAN_MAPPING_FILE.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
                print(f"Fit: pan = {mapping['slope']:.4f} * azimuth + {mapping['intercept']:.4f}")
                print(
                    f"RMSE: {mapping['rmse_pan_units']:.3f} pan units, "
                    f"max residual: {mapping['max_abs_residual_pan_units']:.3f}"
                )
                print(f"Saved to {PAN_MAPPING_FILE}")
                break
            elif key == ord("q"):
                break

            if step != 0.0:
                robot.send_action({"shoulder_pan.pos": pan + step})
    finally:
        camera.disconnect()
        try:
            robot.disconnect()
        except RuntimeError as exc:
            print(f"[pan_mapping] robot disconnect error (non-fatal): {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
