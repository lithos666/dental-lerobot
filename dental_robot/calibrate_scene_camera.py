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

"""Calibrate the fixed scene camera intrinsics with a chessboard.

Usage:
    python -m dental_robot.calibrate_scene_camera

Print a 9x6 chessboard (e.g. OpenCV's pattern, squares of 28.5 mm), tape it to
something rigid. Show it to the scene camera at varied angles/distances.
Keys: SPACE = capture a view (need >= 15), c = calibrate & save, q = quit.
"""

import cv2
import numpy as np

from dental_robot.config import INTRINSICS_FILE, make_scene_camera_config
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

CHESSBOARD = (9, 6)  # inner corners per row, column
SQUARE_SIZE_M = 0.0285  # printed square side length in meters (28.5 mm)
MIN_VIEWS = 15


def main():
    """CLI entry point: chessboard intrinsics calibration of the scene camera."""
    camera = OpenCVCamera(make_scene_camera_config())
    camera.connect()

    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHESSBOARD[0], 0 : CHESSBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M

    obj_points, img_points = [], []
    image_size = None
    print(f"SPACE = capture (need >= {MIN_VIEWS}), c = calibrate & save, q = quit")

    try:
        while True:
            frame_rgb = camera.read()
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            image_size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)

            display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if found:
                cv2.drawChessboardCorners(display, CHESSBOARD, corners, found)
            cv2.putText(
                display,
                f"views: {len(obj_points)}/{MIN_VIEWS}  board: {'OK' if found else '--'}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 0, 255),
                2,
            )
            cv2.imshow("scene camera calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") and found:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                obj_points.append(objp)
                img_points.append(corners)
                print(f"Captured view {len(obj_points)}")
            elif key == ord("c"):
                if len(obj_points) < MIN_VIEWS:
                    print(f"Need at least {MIN_VIEWS} views, have {len(obj_points)}")
                    continue
                rms, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
                np.savez(INTRINSICS_FILE, camera_matrix=mtx, dist_coeffs=dist)
                print(f"RMS reprojection error: {rms:.3f} px (aim for < 0.5)")
                print(f"Saved intrinsics to {INTRINSICS_FILE}")
                break
            elif key == ord("q"):
                break
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
