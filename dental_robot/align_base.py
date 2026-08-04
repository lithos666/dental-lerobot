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
    START_POSE_FILE,
    connect_follower,
    make_follower_config,
    make_scene_camera_config,
)
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

STEP_LIMIT = 5.0  # max pan increment per command (stay below max_relative_target)
ARM_V_MAX = 8.0   # peak velocity for arm joints (units per step, far from target)
ARM_V_MIN = 0.3   # creep velocity near target (prevents dead-zone oscillation)
ARM_D_REF = 20.0  # sigmoid scale: distance at which speed is ~76% of max
SETTLE_S = 0.05   # inter-step delay (shorter = smoother apparent motion)
TOLERANCE = 0.5  # acceptable |target - present| in normalized pan units
ARM_TOLERANCE = 0.8  # arm joints: stop threshold (tighter to avoid PID micro-jitter)
MAX_STEPS = 60
ARM_MAX_STEPS = 200  # more steps allowed because each step is smaller
ARM_SETTLE_STEPS = 6  # corrective micro-steps after main slew to absorb residual error
# Pre-lift before the return slew: the full slew moves all joints at once,
# which can sweep the end effector across the dental model. Raising the wrist
# first pulls the drill away from the model before anything else moves.
LIFT_JOINT = "wrist_flex"  # ID 4
LIFT_OFFSET = -8.0  # negative = raise the end effector (validated on hardware)
# Shorter than SETTLE_S so the pre-lift flows continuously into the main slew
# instead of pausing between steps.
LIFT_SETTLE_S = 0.02


def load_mapping() -> dict:
    """Load the calibrated pan-angle -> base-yaw mapping from disk."""
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

    # Settling phase: the servo often stalls just outside tolerance because of
    # static friction/backlash.  Each attempt re-commands the target with a
    # long-enough wait for the PID loop to actually reach it; if the joint is
    # stuck, alternate +-nudge commands to break stiction.
    nudge = max(TOLERANCE, 1.0)
    for i in range(8):
        present = robot.bus.sync_read("Present_Position")["shoulder_pan"]
        error = target_pan - present
        if abs(error) <= TOLERANCE:
            break
        cmd = target_pan if i % 2 == 0 else target_pan + np.sign(error) * nudge
        robot.send_action({"shoulder_pan.pos": cmd})
        time.sleep(0.3)  # long enough for the servo PID to settle

    present = robot.bus.sync_read("Present_Position")["shoulder_pan"]
    error = target_pan - present
    if abs(error) > TOLERANCE * 6:
        # Truly blocked: cable snag, joint limit or a stalled motor.
        raise RuntimeError(
            f"Base alignment did not converge within {MAX_STEPS} steps "
            f"(target {target_pan:+.2f}, last error {error:+.2f}). "
            "Check for obstructions or joint range limits."
        )
    if abs(error) > TOLERANCE:
        # Marginally outside tolerance but close enough for a consistent start.
        print(f"[align] WARNING: residual error {error:+.2f} (tolerance {TOLERANCE}), continuing.")

    final = robot.bus.sync_read("Present_Position")["shoulder_pan"]
    print(f"[align] done, shoulder_pan={final:+.2f} (error {target_pan - final:+.2f})")
    return final


def _sigmoid_speed(distance: float, v_max: float, v_min: float, d_ref: float) -> float:
    """Sigmoid velocity profile: v(d) = v_min + (v_max - v_min) * tanh(d / d_ref).

    Returns a positive scalar.  Multiply by sign(err) outside.

    Reference: Craig, *Introduction to Robotics*, 3rd ed., §7.2
    (smooth velocity profiles for point-to-point motion).
    """
    return v_min + (v_max - v_min) * np.tanh(distance / d_ref)


def pre_lift_wrist(robot) -> None:
    """Raise wrist_flex (ID 4) by LIFT_OFFSET to pull the drill off the model.

    MUST run before any base rotation (align_base): rotating shoulder_pan
    (ID 1) while the drill is still down sweeps it across the dental model.
    """
    present = robot.bus.sync_read("Present_Position")[LIFT_JOINT]
    lift_target = float(np.clip(present + LIFT_OFFSET, -100.0, 100.0))
    print(f"[arm_reset] pre-lift {LIFT_JOINT}: {present:+.1f} -> {lift_target:+.1f}")
    for _ in range(MAX_STEPS):
        present = robot.bus.sync_read("Present_Position")[LIFT_JOINT]
        error = lift_target - present
        if abs(error) <= ARM_TOLERANCE:
            break
        step = np.clip(error, -STEP_LIMIT, STEP_LIMIT)
        robot.send_action({f"{LIFT_JOINT}.pos": present + float(step)})
        time.sleep(LIFT_SETTLE_S)


def move_arm_to_start_pose(robot, do_pre_lift: bool = True) -> None:
    """Move arm joints (ID 2-6) to the saved canonical start pose.

    Uses a sigmoid velocity profile: full speed when far from the target,
    smooth deceleration as joints approach their goals.  This eliminates the
    setpoint-jumping that causes oscillation with constant-step control.

    After the main slew, a short corrective phase sends micro-steps to
    absorb any residual PID tracking error.

    Set do_pre_lift=False when pre_lift_wrist() was already called earlier
    in the sequence (e.g. before align_base), to avoid lifting twice.

    Reference: Craig, *Introduction to Robotics*, 3rd ed., ch. 7
    (trajectory planning with smooth velocity profiles).
    """
    if not START_POSE_FILE.exists():
        raise FileNotFoundError(
            f"Start pose not found: {START_POSE_FILE}\n"
            "Run `python -m dental_robot.calibrate_start_pose` first."
        )
    target = json.loads(START_POSE_FILE.read_text(encoding="utf-8"))
    joints = list(target.keys())

    # --- Phase 0: pre-lift the wrist (ID 4) to clear the dental model ---
    if do_pre_lift and LIFT_JOINT in joints:
        pre_lift_wrist(robot)

    # --- Phase 1: main slew with sigmoid velocity profile ---
    for _ in range(ARM_MAX_STEPS):
        present = robot.bus.sync_read("Present_Position")
        errors = {j: target[j] - present[j] for j in joints}
        max_err = max(abs(v) for v in errors.values())

        if max_err <= ARM_TOLERANCE:
            break

        # Per-joint speed scaled by the sigmoid of its own distance.
        action = {}
        for j in joints:
            err = errors[j]
            speed = _sigmoid_speed(abs(err), ARM_V_MAX, ARM_V_MIN, ARM_D_REF)
            # Clamp to the global velocity ceiling.
            speed = min(speed, ARM_V_MAX)
            step = np.clip(np.sign(err) * speed, -abs(err), abs(err))
            action[f"{j}.pos"] = present[j] + float(step)

        robot.send_action(action)
        time.sleep(SETTLE_S)
    else:
        print(f"[arm_reset] WARNING: main slew did not converge in {ARM_MAX_STEPS} steps")

    # --- Phase 2: corrective micro-steps to absorb PID residual ---
    # After the sigmoid slew the joints are within ARM_TOLERANCE but the
    # servo PID may still be chasing.  A few small zero-velocity commands
    # (re-sending the target) give the PID loop time to settle without
    # injecting new setpoint jumps.
    for _ in range(ARM_SETTLE_STEPS):
        action = {f"{j}.pos": target[j] for j in joints}
        robot.send_action(action)
        time.sleep(0.08)

    final = robot.bus.sync_read("Present_Position")
    vals = "  ".join(f"{j}={final[j]:+.1f}" for j in joints)
    residual = max(abs(target[j] - final[j]) for j in joints)
    print(f"[arm_reset] done: {vals}  (max residual {residual:+.2f})")


def main():
    """CLI entry point: align the robot base to the ArUco scene marker."""
    camera = OpenCVCamera(make_scene_camera_config())
    camera.connect()
    robot = connect_follower(make_follower_config(with_cameras=False))
    try:
        align_base(robot, camera.read)
    finally:
        camera.disconnect()
        try:
            robot.disconnect()
        except RuntimeError as exc:
            print(f"[align] robot disconnect error (non-fatal): {exc}")


if __name__ == "__main__":
    main()
