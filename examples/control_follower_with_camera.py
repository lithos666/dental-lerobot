# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Control the SO-101 follower arm with the keyboard while displaying its camera feed.

Keys (press in the OpenCV window):
    a / d : shoulder_pan  - / +
    w / s : shoulder_lift + / -
    e / r : elbow_flex    + / -
    t / g : wrist_flex    + / -
    y / h : wrist_roll    + / -
    o / c : gripper open / close
    q     : quit

Run:
    python examples/control_follower_with_camera.py
"""

import time

import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.robot_utils import busy_wait

FPS = 30
STEP = 2.0  # increment per key press, in normalized units [-100, 100]

KEY_TO_MOTOR = {
    ord("a"): ("shoulder_pan.pos", -STEP),
    ord("d"): ("shoulder_pan.pos", +STEP),
    ord("w"): ("shoulder_lift.pos", +STEP),
    ord("s"): ("shoulder_lift.pos", -STEP),
    ord("e"): ("elbow_flex.pos", +STEP),
    ord("r"): ("elbow_flex.pos", -STEP),
    ord("t"): ("wrist_flex.pos", +STEP),
    ord("g"): ("wrist_flex.pos", -STEP),
    ord("y"): ("wrist_roll.pos", +STEP),
    ord("h"): ("wrist_roll.pos", -STEP),
    ord("o"): ("gripper.pos", +STEP),
    ord("c"): ("gripper.pos", -STEP),
}


def main():
    config = SO101FollowerConfig(
        port="COM24",
        id="my_awesome_follower_arm",
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=0, width=640, height=480, fps=FPS, flip_horizontal=True
            ),
        },
        # Safety: limit how far a single command can move each joint
        # (must be a float: ensure_safe_goal_position rejects int)
        max_relative_target=10.0,
    )

    robot = SO101Follower(config)
    # Bypass the flaky per-motor ping handshake (unreliable on CH343 serial adapters):
    # connect the bus directly, push the calibration file to the motors, then configure.
    robot.bus.connect(handshake=False)
    robot.bus.write_calibration(robot.calibration)
    for cam in robot.cameras.values():
        cam.connect()
    robot.configure()

    # Start from the current position so the arm does not jump on the first command
    obs = robot.get_observation()
    action = {k: v for k, v in obs.items() if k.endswith(".pos")}

    try:
        while True:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            frame = cv2.cvtColor(obs["front"], cv2.COLOR_RGB2BGR)
            cv2.imshow("front camera (q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in KEY_TO_MOTOR:
                motor, delta = KEY_TO_MOTOR[key]
                low, high = (0, 100) if motor == "gripper.pos" else (-100, 100)
                action[motor] = min(max(action[motor] + delta, low), high)

            robot.send_action(action)

            positions = "  ".join(f"{k.removesuffix('.pos')}: {v:6.1f}" for k, v in action.items())
            print(f"\r{positions}", end="", flush=True)

            busy_wait(1 / FPS - (time.perf_counter() - loop_start))
    except KeyboardInterrupt:
        pass
    finally:
        print()
        robot.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
