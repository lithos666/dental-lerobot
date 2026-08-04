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

"""Generate printable ArUco markers (PNG) for the dental robot setup.

Usage:
    python -m dental_robot.generate_markers

Print at exactly MARKER_SIZE_M side length (default 40 mm), matte paper,
then measure with a ruler and update MARKER_SIZE_M in config.py if it differs.
Marker 0 -> dental model, marker 1 -> table reference next to the robot base.
"""

from pathlib import Path

import cv2

from dental_robot.config import ARUCO_DICT_ID, BASE_MARKER_ID, TOOTH_MARKER_ID

OUT_DIR = Path(__file__).parent / "markers"
PIXELS = 600  # print resolution of the marker image
BORDER_PX = 60  # white quiet zone around the marker (required for detection)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    for marker_id, name in [(TOOTH_MARKER_ID, "tooth"), (BASE_MARKER_ID, "base")]:
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, PIXELS)
        img = cv2.copyMakeBorder(
            img, BORDER_PX, BORDER_PX, BORDER_PX, BORDER_PX, cv2.BORDER_CONSTANT, value=255
        )
        out = OUT_DIR / f"marker_{marker_id}_{name}.png"
        cv2.imwrite(str(out), img)
        print(f"Saved {out}")
    print("Print each marker so the BLACK square (not the white border) matches MARKER_SIZE_M.")


if __name__ == "__main__":
    main()
