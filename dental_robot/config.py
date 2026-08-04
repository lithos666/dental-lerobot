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

"""Shared configuration for the dental implant robot prototype.

Plan C architecture:
  Phase 1 (geometry): the fixed scene camera detects ArUco markers, solves the
    tooth-model azimuth and rotates the base (shoulder_pan) to a canonical pose.
  Phase 2 (learning): from the canonical pose, a single-task ACT policy performs
    the fine insertion motion with visual closed-loop feedback.
"""

from pathlib import Path

import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

# --- Hardware ---
FOLLOWER_PORT = "COM24"
LEADER_PORT = "COM22"
ROBOT_ID = "my_awesome_follower_arm"
LEADER_ID = "my_awesome_leader_arm"

FPS = 30
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Camera 0: fixed scene camera used for ArUco localization.
# IMPORTANT: never flip this one. Mirroring breaks ArUco corner ordering and
# would silently corrupt the PnP pose estimate.
SCENE_CAM_INDEX = 0
# Camera 1: wrist ("handeye") camera. Its sensor is mirrored -> flip enabled.
HANDEYE_CAM_INDEX = 1

# --- ArUco ---
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_100
MARKER_SIZE_M = 0.03  # printed marker side length in meters (30mm), match your print
TOOTH_MARKER_ID = 0  # marker glued on / next to the dental model
# Reference marker fixed on the table at the robot base. Mount it with its
# X axis (right edge direction) pointing along the arm's "forward" direction.
BASE_MARKER_ID = 1


def make_aruco_detector() -> cv2.aruco.ArucoDetector:
    """Create an ArUco detector with relaxed parameters for small markers.

    Default parameters reject markers that are small in the image or viewed
    at an angle. These tuned settings increase the bit-error tolerance and
    lower the minimum perimeter so 30mm markers at ~50cm distance are accepted.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    # Allow correcting up to 2 bit errors (4x4 has 16 bits, default allows ~10%)
    params.maxErroneousBitsInBorderRate = 0.45
    params.errorCorrectionRate = 0.5
    # Accept smaller markers (default minPerimeterRate=0.03 can miss distant ones)
    params.minMarkerPerimeterRate = 0.01
    params.minMarkerDistanceRate = 0.02
    # Adaptive threshold tuning for uneven lighting
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 25
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    # Enable corner refinement for better PnP accuracy
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    return cv2.aruco.ArucoDetector(dictionary, params)

# --- Files produced by the calibration scripts ---
_HERE = Path(__file__).parent
INTRINSICS_FILE = _HERE / "scene_camera_intrinsics.npz"
PAN_MAPPING_FILE = _HERE / "pan_mapping.json"
# Fixed scene geometry: base marker pose in the robot-base frame and the
# calibrated canonical tooth pose. Produced by calibrate_scene_geometry.py
# and consumed by align_base / run_pipeline to guarantee that the ACT
# policy always starts from the same spatial configuration.
SCENE_GEOMETRY_FILE = _HERE / "scene_geometry.json"


def make_scene_camera_config() -> OpenCVCameraConfig:
    return OpenCVCameraConfig(
        index_or_path=SCENE_CAM_INDEX,
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        fps=FPS,
        flip_horizontal=False,
    )


def make_follower_config(with_cameras: bool = True, max_relative_target: float = 10.0) -> SO101FollowerConfig:
    cameras = {}
    if with_cameras:
        cameras = {
            "handeye": OpenCVCameraConfig(
                index_or_path=HANDEYE_CAM_INDEX,
                width=CAM_WIDTH,
                height=CAM_HEIGHT,
                fps=FPS,
                flip_horizontal=True,
            ),
            "fixed": OpenCVCameraConfig(
                index_or_path=SCENE_CAM_INDEX,
                width=CAM_WIDTH,
                height=CAM_HEIGHT,
                fps=FPS,
                flip_horizontal=False,
            ),
        }
    return SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=ROBOT_ID,
        cameras=cameras,
        # Safety clamp per command, must be a float (see ensure_safe_goal_position)
        max_relative_target=max_relative_target,
    )


def connect_follower(config: SO101FollowerConfig) -> SO101Follower:
    """Connect bypassing the flaky per-motor ping handshake (CH343 adapter)."""
    robot = SO101Follower(config)
    robot.bus.connect(handshake=False)
    robot.bus.write_calibration(robot.calibration)
    for cam in robot.cameras.values():
        cam.connect()
    robot.configure()
    return robot


def connect_leader() -> SO101Leader:
    """Connect the leader arm, same handshake bypass as the follower.

    `configure()` leaves torque disabled so the arm can be moved by hand.
    """
    teleop = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    teleop.bus.connect(handshake=False)
    teleop.bus.write_calibration(teleop.calibration)
    teleop.configure()
    return teleop
