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

"""Full Plan-C pipeline: ArUco base alignment, then ACT fine manipulation.

Usage:
    python -m dental_robot.run_pipeline --policy_path outputs/train/act_dental/checkpoints/last/pretrained_model
    python -m dental_robot.run_pipeline --policy_path ... --skip_align   # phase 2 only

Phase 1 uses the robot's own "fixed" camera stream (no second device handle),
phase 2 replicates the official lerobot-record inference loop.
"""

import argparse
import time

import cv2

from dental_robot.align_base import align_base, move_arm_to_start_pose
from dental_robot.aruco_locator import ArucoLocator
from dental_robot.config import FPS, connect_follower, make_follower_config
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.control_utils import predict_action
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device


def run_policy(robot, policy: ACTPolicy, duration_s: float):
    """Run the ACT policy in a closed loop on the robot for `duration_s` seconds."""
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    device = get_safe_torch_device(policy.config.device)
    policy.reset()

    print(f"[act] running policy for {duration_s:.0f}s at {FPS} Hz (Ctrl+C to stop)")
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        loop_t = time.perf_counter()

        observation = robot.get_observation()
        observation_frame = build_dataset_frame(obs_features, observation, prefix="observation")
        action_values = predict_action(
            observation_frame,
            policy,
            device,
            policy.config.use_amp,
            robot_type=robot.robot_type,
        )
        action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        robot.send_action(action)

        busy_wait(1 / FPS - (time.perf_counter() - loop_t))
    print("[act] done")


def main():
    """CLI entry point: align the base (phase 1) then run the ACT policy (phase 2)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", required=True, help="Trained ACT checkpoint directory")
    parser.add_argument("--duration", type=float, default=20.0, help="Policy run time (s)")
    parser.add_argument("--skip_align", action="store_true", help="Skip phase 1 (ArUco alignment)")
    args = parser.parse_args()

    policy = ACTPolicy.from_pretrained(args.policy_path)
    robot = connect_follower(make_follower_config(with_cameras=True))
    try:
        if not args.skip_align:
            locator = ArucoLocator()
            # Live tracking: enable temporal smoothing for jitter rejection.
            locator.enable_temporal_smoothing(True)
            # Reuse the robot's un-flipped "fixed" camera stream for detection.
            grab_frame = robot.cameras["fixed"].read
            align_base(robot, grab_frame, locator=locator)
            move_arm_to_start_pose(robot)
            time.sleep(0.5)  # let the scene settle before handing over to ACT
        run_policy(robot, policy, args.duration)

        # Return arm joints (ID 2-6) to the calibrated start pose so the
        # robot is ready for the next run without manual repositioning.
        print("[pipeline] returning to start pose...")
        move_arm_to_start_pose(robot)
        print("[pipeline] done — arm at start pose")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        try:
            robot.disconnect()
        except RuntimeError as exc:
            print(f"[pipeline] robot disconnect error (non-fatal): {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
