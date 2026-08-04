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

"""Dental implant robot prototype on the SO-101 platform (Plan C).

Phase 1 (geometry): ArUco localization -> base rotation normalization.
Phase 2 (learning): single-task ACT policy for the fine insertion motion.

Module map:
    config.py                 shared hardware / ArUco / file-path configuration
    generate_markers.py       printable ArUco marker PNGs
    calibrate_scene_camera.py scene camera intrinsics (chessboard)
    calibrate_pan_mapping.py  azimuth -> shoulder_pan linear mapping
    aruco_locator.py          marker detection + azimuth solving
    align_base.py             phase 1 executable / reusable align_base()
    record_episodes.py        teleop data recording with per-episode alignment
    train_act.py              offline ACT training with validation split
    run_pipeline.py           phase 1 + phase 2 end-to-end
"""
