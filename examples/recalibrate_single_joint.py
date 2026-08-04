#!/usr/bin/env python

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
Recalibrate a single joint of an SO101 arm without redoing the full calibration.

The other joints keep their existing calibration values from the calibration file.

Examples:
    python examples/recalibrate_single_joint.py --arm leader --port COM22 --id my_awesome_leader_arm --motor wrist_roll
    python examples/recalibrate_single_joint.py --arm follower --port COM24 --id my_awesome_follower_arm --motor elbow_flex
"""

import argparse

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    parser = argparse.ArgumentParser(description="Recalibrate a single joint of an SO101 arm.")
    parser.add_argument("--arm", choices=["leader", "follower"], required=True)
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM22")
    parser.add_argument("--id", required=True, help="Arm id used for the calibration file")
    parser.add_argument("--motor", choices=MOTOR_NAMES, required=True, help="Joint to recalibrate")
    args = parser.parse_args()

    if args.arm == "leader":
        device = SO101Leader(SO101LeaderConfig(port=args.port, id=args.id))
    else:
        device = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))

    if not device.calibration:
        raise RuntimeError(
            f"No existing calibration file found at {device.calibration_fpath}. "
            "Run the full `lerobot-calibrate` first."
        )

    motor = args.motor
    device.connect(calibrate=False)
    try:
        bus = device.bus
        # Make sure the motors hold the values from the calibration file before touching one of them
        bus.write_calibration(device.calibration)

        bus.disable_torque()
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move '{motor}' to the middle of its range of motion and press ENTER...")
        homing_offsets = bus.set_half_turn_homings(motor)

        print(f"Move '{motor}' through its entire range of motion (both directions).")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = bus.record_ranges_of_motion([motor])

        device.calibration[motor] = MotorCalibration(
            id=bus.motors[motor].id,
            drive_mode=0,
            homing_offset=homing_offsets[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )

        bus.write_calibration(device.calibration)
        device._save_calibration()
        print(f"New values for '{motor}': {device.calibration[motor]}")
        print(f"Calibration saved to {device.calibration_fpath}")
    finally:
        device.disconnect()


if __name__ == "__main__":
    main()
