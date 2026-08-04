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

"""Capture and save the canonical start pose for the follower arm (ID 2-6).

The start pose defines where the arm joints (shoulder_lift through gripper)
should be at the beginning of every episode. Combined with ArUco alignment
(which handles shoulder_pan / ID 1), this guarantees a fully deterministic
starting configuration for ACT training and deployment.

Usage:
    1. Connect the follower arm and leader arm.
    2. Run this script.
    3. Manually move the leader arm to the desired starting pose
       (the follower follows via teleoperation).
    4. When satisfied, press SPACE to save. The follower's current joint
       positions (ID 2-6) are written to start_pose.json.

The saved file is consumed by record_episodes.py and run_pipeline.py
to auto-reset the arm before each episode.
"""

import json

import cv2
import numpy as np

from dental_robot.config import (
    START_POSE_FILE,
    connect_follower,
    connect_leader,
    make_follower_config,
)
from lerobot.utils.utils import init_logging

# Joints that are auto-reset (everything except shoulder_pan which ArUco handles).
ARM_JOINTS = [
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main():
    init_logging()

    robot = connect_follower(make_follower_config(with_cameras=False))
    teleop = connect_leader()

    print("=" * 60)
    print("  Start Pose Calibration")
    print("=" * 60)
    print()
    print("Move the LEADER arm to the desired starting pose.")
    print("The follower arm follows in real time.")
    print()
    print("  SPACE  = save current follower pose (ID 2-6)")
    print("  ESC    = cancel without saving")
    print()

    cv2.namedWindow("start_pose", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Teleoperate: leader controls follower
            action = teleop.get_action()
            robot.send_action(action)

            # Show current follower positions
            obs = robot.get_observation()
            lines = []
            for j in ARM_JOINTS:
                val = obs.get(f"{j}.pos", float("nan"))
                lines.append(f"  {j:16s} = {val:+7.2f}")

            display = np.zeros((220, 500, 3), dtype=np.uint8)
            cv2.putText(display, "Move leader to start pose", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i, line in enumerate(lines):
                cv2.putText(display, line, (20, 70 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(display, "SPACE=save  ESC=cancel", (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("start_pose", display)

            key = cv2.waitKey(30) & 0xFF
            if key == 32:  # Space
                break
            if key == 27:  # Esc
                print("[calibrate] Cancelled.")
                return

        # Read the follower's current arm joint positions (ID 2-6)
        # BEFORE disconnecting (bus must still be open).
        positions = robot.bus.sync_read("Present_Position")
        start_pose = {j: float(positions[j]) for j in ARM_JOINTS}
    finally:
        cv2.destroyAllWindows()
        try:
            robot.disconnect()
        except RuntimeError as exc:
            print(f"[calibrate] robot disconnect error (non-fatal): {exc}")
        try:
            teleop.disconnect()
        except RuntimeError as exc:
            print(f"[calibrate] leader disconnect error (non-fatal): {exc}")

    # Save to file.
    START_POSE_FILE.write_text(json.dumps(start_pose, indent=2), encoding="utf-8")

    print()
    print("[calibrate] Saved start pose:")
    for j, v in start_pose.items():
        print(f"  {j:16s} = {v:+7.2f}")
    print(f"\nFile: {START_POSE_FILE}")
    print("This pose will be used by record_episodes.py and run_pipeline.py")
    print("to auto-reset the arm (ID 2-6) after each ArUco alignment.")


if __name__ == "__main__":
    main()
