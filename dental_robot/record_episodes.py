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

"""Record ACT training episodes with the Plan-C protocol baked in.

Replaces `lerobot-record` for this project because the CH343 serial adapter
needs the handshake-bypass connection (see config.connect_follower). On top of
the official recording loop it enforces the canonical-pose protocol: before
every episode the base is re-aligned with ArUco (`align_base`), so all episodes
start from the same normalized scene the deployed pipeline will produce.

Usage:
    python -m dental_robot.record_episodes --dataset_repo_id ${HF_USER}/dental_implant
    python -m dental_robot.record_episodes --dataset_repo_id ... --resume   # append episodes

Keyboard controls during recording (official lerobot bindings):
    right arrow -> end current episode early
    left arrow  -> re-record current episode
    escape      -> stop recording session

Per-episode flow:
    1. move the dental model to a new position
    2. the script runs align_base (base rotates to the canonical pose)
    3. put the leader arm at the start pose, press ENTER, teleoperate the task
    4. reset phase: episode is saved, move the model for the next one
"""

import argparse

from dental_robot.align_base import align_base
from dental_robot.aruco_locator import ArucoLocator
from dental_robot.config import FPS, connect_follower, connect_leader, make_follower_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.utils import init_logging, log_say


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_repo_id", required=True, help="e.g. your_hf_username/dental_implant")
    parser.add_argument("--dataset_root", default=None, help="Local dataset dir (default: HF cache)")
    parser.add_argument("--task", default="Insert the implant into the dental model", help="Task description")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episode_time", type=float, default=30.0, help="Max seconds per episode")
    parser.add_argument("--resume", action="store_true", help="Append to an existing dataset")
    parser.add_argument("--skip_align", action="store_true", help="Skip ArUco alignment (debug only)")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset when done")
    args = parser.parse_args()

    init_logging()

    robot = connect_follower(make_follower_config(with_cameras=True))
    teleop = connect_leader()
    locator = None if args.skip_align else ArucoLocator()

    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
        dataset.start_image_writer(num_processes=0, num_threads=4 * len(robot.cameras))
    else:
        dataset = LeRobotDataset.create(
            args.dataset_repo_id,
            FPS,
            root=args.dataset_root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_threads=4 * len(robot.cameras),
        )

    listener, events = init_keyboard_listener()

    try:
        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < args.num_episodes and not events["stop_recording"]:
                # Plan-C protocol: normalize the scene before every episode so
                # training and deployment share the same starting distribution.
                if not args.skip_align:
                    input(
                        f"\n[{recorded + 1}/{args.num_episodes}] Place the model, then press ENTER to align..."
                    )
                    align_base(robot, robot.cameras["fixed"].read, locator=locator)

                input("Hold the leader at the start pose, press ENTER to record...")
                log_say(f"Recording episode {dataset.num_episodes}", play_sounds=False)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=args.episode_time,
                    single_task=args.task,
                )

                if events["rerecord_episode"]:
                    log_say("Re-recording episode", play_sounds=False)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1
                print(f"Episode saved ({recorded}/{args.num_episodes})")
    finally:
        robot.disconnect()
        teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()

    if args.push_to_hub:
        dataset.push_to_hub(tags=["dental", "act", "so101"], private=False)

    print(f"\nDone. {recorded} episode(s) recorded in dataset '{args.dataset_repo_id}'.")
    print(f"Train with: python -m dental_robot.train_act --dataset_repo_id {args.dataset_repo_id}")


if __name__ == "__main__":
    main()
