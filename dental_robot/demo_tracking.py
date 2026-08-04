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

"""Demo: move the dental model around, watch the arm track its azimuth.

This is a pure visualization of the Phase-1 geometry pipeline (no ACT policy,
no canonical-pose locking). It shows the live ArUco azimuth measurement and
the shoulder_pan value the align_base controller WOULD command, so you can
verify the mapping end-to-end before recording episodes or running the policy.

Controls:
    SPACE = rotate the arm to the mapped target for the current azimuth
    r     = reset temporal smoother state (after large scene changes)
    q/Esc = quit

Usage:
    python -m dental_robot.demo_tracking

Move the dental model (with the tooth marker) to different positions in the
working sector. The on-screen HUD shows:
  - measured azimuth (deg, in the base-marker frame)
  - mapped target shoulder_pan  (= slope * azimuth + intercept)
  - present shoulder_pan
  - error (target - present)
  - whether both markers are currently visible
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dental_robot.align_base import load_mapping
from dental_robot.aruco_locator import ArucoLocator
from dental_robot.config import (
    BASE_MARKER_ID,
    TOOTH_MARKER_ID,
    connect_follower,
    make_follower_config,
    make_scene_camera_config,
)
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

# Same safety constants as align_base.py (kept local to avoid a circular import).
STEP_LIMIT = 5.0
SETTLE_S = 0.15
TOLERANCE = 0.5
MAX_STEPS = 60


def draw_hud(
    frame_bgr: np.ndarray,
    azimuth: float | None,
    target_pan: float | None,
    present_pan: float,
    both_visible: bool,
    aligning: bool,
) -> np.ndarray:
    """Draw the heads-up display on a BGR frame."""
    lines = []
    if both_visible and azimuth is not None:
        lines.append((f"azimuth: {azimuth:+.2f} deg", (0, 255, 0)))
    else:
        lines.append(("azimuth: markers not visible", (0, 0, 255)))

    if target_pan is not None:
        lines.append((f"target shoulder_pan: {target_pan:+.2f}", (255, 255, 0)))
    lines.append((f"present shoulder_pan: {present_pan:+.2f}", (255, 255, 255)))
    if target_pan is not None:
        err = target_pan - present_pan
        col = (0, 255, 0) if abs(err) < TOLERANCE else (0, 165, 255)
        lines.append((f"error: {err:+.2f}", col))
    lines.append((
        f"mode: {'ALIGNING...' if aligning else 'idle (SPACE=align, r=reset, q=quit)'}",
        (200, 200, 200),
    ))

    y = 25
    for text, color in lines:
        cv2.putText(frame_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
    return frame_bgr


def align_once(robot, locator, camera) -> bool:
    """Run one full alignment pass. Returns True if converged."""
    mapping = load_mapping()
    # Average azimuth over a few frames for a stable target.
    try:
        azimuth = locator.averaged_azimuth_deg(camera.read, n_frames=10)
    except RuntimeError as e:
        print(f"[align] {e}")
        return False
    target_pan = float(np.clip(mapping["slope"] * azimuth + mapping["intercept"], -100.0, 100.0))
    print(f"[align] azimuth={azimuth:+.2f} deg -> target shoulder_pan={target_pan:+.2f}")

    for step in range(MAX_STEPS):
        present = robot.bus.sync_read("Present_Position")["shoulder_pan"]
        error = target_pan - present
        if abs(error) <= TOLERANCE:
            print(f"[align] converged at {present:+.2f} after {step} steps")
            return True
        cmd = max(-STEP_LIMIT, min(STEP_LIMIT, error))
        robot.send_action({"shoulder_pan.pos": present + cmd})
        time.sleep(SETTLE_S)
    print(f"[align] did NOT converge within {MAX_STEPS} steps")
    return False


def main():
    print("=" * 60)
    print("Demo: azimuth tracking")
    print("=" * 60)
    print("Move the dental model to different positions in the working sector.")
    print("The HUD shows the measured azimuth and the mapped target pan.")
    print("Press SPACE to rotate the arm to the target; q/Esc to quit.")
    print()

    locator = ArucoLocator()
    locator.enable_temporal_smoothing(True)
    camera = OpenCVCamera(make_scene_camera_config())
    camera.connect()
    robot = connect_follower(make_follower_config(with_cameras=False))
    mapping = load_mapping()

    aligning = False
    try:
        while True:
            frame_rgb = camera.read()
            # Live azimuth for the HUD (single-frame, no averaging).
            az = locator.tooth_azimuth_deg(frame_rgb)
            both_visible = az is not None
            target_pan = (
                float(np.clip(mapping["slope"] * az + mapping["intercept"], -100.0, 100.0))
                if az is not None
                else None
            )
            present_pan = robot.bus.sync_read("Present_Position")["shoulder_pan"]

            # Draw ArUco detections + axes on the display frame.
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = locator.detector.detectMarkers(gray)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
                # Annotate which marker is which.
                if corners:
                    flat = ids.flatten()
                    for c, mid in zip(corners, flat):
                        center = c.reshape(4, 2).mean(axis=0).astype(int)
                        label = "BASE" if mid == BASE_MARKER_ID else "TOOTH"
                        cv2.putText(bgr, f"ID{mid} {label}", tuple(center),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            bgr = draw_hud(bgr, az, target_pan, present_pan, both_visible, aligning)
            cv2.imshow("demo tracking (SPACE=align, r=reset, q=quit)", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                aligning = True
                # Redraw once to show ALIGNING state, then run.
                bgr = draw_hud(bgr, az, target_pan, present_pan, both_visible, aligning)
                cv2.imshow("demo tracking (SPACE=align, r=reset, q=quit)", bgr)
                cv2.waitKey(1)
                ok = align_once(robot, locator, camera)
                print(f"[demo] alignment {'OK' if ok else 'FAILED'}")
                aligning = False
            elif key == ord("r"):
                locator.reset_temporal_state()
                print("[demo] temporal state reset")
            elif key == ord("q") or key == 27:
                break
    finally:
        camera.disconnect()
        robot.bus.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
