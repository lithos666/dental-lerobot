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

"""Phase 1: rotate the base so the dental model sits at the canonical pose.

Usage (standalone test, no ACT involved):
    python -m dental_robot.align_base

Reusable entry point for the full pipeline: `align_base(robot, grab_frame)`.
"""

import json
import time
from collections.abc import Callable

import numpy as np

from dental_robot.aruco_locator import ArucoLocator
from dental_robot.config import (
    PAN_MAPPING_FILE,
    SCENE_GEOMETRY_FILE,
    connect_follower,
    make_follower_config,
    make_scene_camera_config,
)
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

STEP_LIMIT = 5.0  # max pan increment per command (stay below max_relative_target)
SETTLE_S = 0.15  # wait between increments so the servo settles
TOLERANCE = 0.5  # acceptable |target - present| in normalized pan units
# Termination guard: worst case is a full sweep of the normalized range
# (200 units / STEP_LIMIT) plus margin for servo settling imprecision.
MAX_STEPS = 60


def load_mapping() -> dict:
    if not PAN_MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"Pan mapping not found: {PAN_MAPPING_FILE}\n"
            "Run `python -m dental_robot.calibrate_pan_mapping` first."
        )
    return json.loads(PAN_MAPPING_FILE.read_text(encoding="utf-8"))


def load_scene_geometry() -> dict | None:
    """Return the saved scene geometry, or None if not yet calibrated.

    The geometry file is optional for legacy compatibility, but when present
    it is used to verify that the measured base pose is consistent with the
    calibrated layout (catches a moved camera/marker before ACT runs).
    """
    if not SCENE_GEOMETRY_FILE.exists():
        return None
    return json.loads(SCENE_GEOMETRY_FILE.read_text(encoding="utf-8"))


def align_base(robot, grab_frame: Callable[[], np.ndarray], locator: ArucoLocator | None = None) -> float:
    """Measure the model azimuth, rotate shoulder_pan to the mapped target.

    `grab_frame` must return an un-flipped RGB frame from the scene camera.
    Returns the final shoulder_pan position.

    If scene geometry was calibrated (calibrate_scene_geometry.py), the
    target shoulder_pan is taken directly from the saved canonical value,
    which guarantees the ACT policy starts from the exact configuration
    used during demonstrations. Otherwise the azimuth->pan mapping is used.
    """
    locator = locator or ArucoLocator()
    # Live tracking: enable temporal smoothing to suppress per-frame jitter.
    locator.enable_temporal_smoothing(True)
    mapping = load_mapping()
    geometry = load_scene_geometry()

    azimuth = locator.averaged_azimuth_deg(grab_frame, n_frames=10)
    mapped_pan = mapping["slope"] * azimuth + mapping["intercept"]
    # Clamp to the normalized joint range: a bad mapping extrapolation must
    # never command the servo beyond its calibrated travel.
    mapped_pan = float(np.clip(mapped_pan, -100.0, 100.0))

    if geometry is not None:
        target_pan = float(geometry["canonical_shoulder_pan"])
        print(
            f"[align] azimuth={azimuth:+.2f} deg (mapped pan {mapped_pan:+.2f}); "
            f"using calibrated canonical shoulder_pan={target_pan:+.2f}"
        )
    else:
        target_pan = mapped_pan
        print(f"[align] azimuth={azimuth:+.2f} deg -> target shoulder_pan={target_pan:+.2f}")

    # Approach the target in bounded increments (safety + smoothness).
    for _ in range(MAX_STEPS):
        present = robot.bus.sync_read("Present_Position")["shoulder_pan"]
        error = target_pan - present
        if abs(error) <= TOLERANCE:
            break
        step = max(-STEP_LIMIT, min(STEP_LIMIT, error))
        robot.send_action({"shoulder_pan.pos": present + step})
        time.sleep(SETTLE_S)
    else:
        raise RuntimeError(
            f"Base alignment did not converge within {MAX_STEPS} steps "
            f"(target {target_pan:+.2f}, last error {error:+.2f}). "
            "Check for obstructions or joint range limits."
        )

    final = robot.bus.sync_read("Present_Position")["shoulder_pan"]
    print(f"[align] done, shoulder_pan={final:+.2f} (error {target_pan - final:+.2f})")
    return final


def main():
    camera = OpenCVCamera(make_scene_camera_config())
    camera.connect()
    robot = connect_follower(make_follower_config(with_cameras=False))
    try:
        align_base(robot, camera.read)
    finally:
        camera.disconnect()
        robot.bus.disconnect()


if __name__ == "__main__":
    main()
