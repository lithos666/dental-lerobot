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
    space       -> end current episode early (save)
    right arrow -> end current episode early (same as space)
    left arrow  -> re-record current episode
    escape      -> stop recording session

Per-episode flow:
    1. move the dental model to a new position
    2. press ENTER in the camera window to trigger ArUco alignment
    3. the script auto-aligns the base (ID 1) and auto-resets arm joints (ID 2-6)
    4. press ENTER to start recording, teleoperate the task
    5. press SPACE when done; episode is saved automatically
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from dental_robot.align_base import align_base, move_arm_to_start_pose, pre_lift_wrist
from dental_robot.aruco_locator import ArucoLocator
from dental_robot.config import FPS, connect_follower, connect_leader, make_follower_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import _init_rerun

try:
    import rerun as rr
except ImportError:
    rr = None  # rerun is optional; only needed with --display_data

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def export_joint_trajectories(dataset_root: Path, episode_index: int) -> None:
    """Read the saved parquet and export joint curves to CSV.

    Output: <dataset_root>/joint_trajectories/episode_NNNNNN.csv
    Columns: timestamp, frame_index, obs_<joint>, act_<joint> for each joint.
    """
    chunk = episode_index // 1000
    parquet_path = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        print(f"[export] parquet not found: {parquet_path}, skipping trajectory export")
        return

    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    out_dir = dataset_root / "joint_trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episode_{episode_index:06d}.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        header = ["timestamp", "frame_index"]
        for j in JOINT_NAMES:
            header.append(f"obs_{j}")
        for j in JOINT_NAMES:
            header.append(f"act_{j}")
        writer.writerow(header)

        # Rows
        for idx in range(len(df)):
            row = [f"{df['timestamp'].iloc[idx]:.4f}", int(df["frame_index"].iloc[idx])]
            obs_state = df["observation.state"].iloc[idx]
            action = df["action"].iloc[idx]
            for v in obs_state:
                row.append(f"{float(v):.2f}")
            for v in action:
                row.append(f"{float(v):.2f}")
            writer.writerow(row)

    print(f"[export] joint trajectories -> {out_path}")


def main():
    """CLI entry point: teleoperate the arm and record LeRobot episodes."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_repo_id", required=True, help="e.g. your_hf_username/dental_implant")
    parser.add_argument("--dataset_root", default="./data/dental", help="Local dataset dir (default: ./data/dental)")
    parser.add_argument("--task", default="Insert the implant into the dental model", help="Task description")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--episode_time", type=float, default=30.0, help="Max seconds per episode")
    parser.add_argument("--resume", action="store_true", help="Append to an existing dataset")
    parser.add_argument("--skip_align", action="store_true", help="Skip ArUco alignment (debug only)")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset when done")
    parser.add_argument("--display_data", action="store_true", help="Show live camera preview (rerun) during recording")
    args = parser.parse_args()

    init_logging()

    if args.display_data:
        if rr is None:
            raise ImportError("rerun is required for --display_data. Install: pip install rerun-sdk")
        _init_rerun(session_name="dental_recording")

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

    # Extend keyboard bindings: SPACE ends the current episode (same as right arrow).
    if listener is not None:
        from pynput import keyboard as _kb

        _orig_on_press = listener.on_press  # type: ignore[attr-defined]

        def _on_press_ext(key):
            if key == _kb.Key.space:
                print("[record] SPACE pressed — ending episode.")
                events["exit_early"] = True
            else:
                _orig_on_press(key)

        listener.stop()
        listener = _kb.Listener(on_press=_on_press_ext)
        listener.start()

    try:
        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < args.num_episodes and not events["stop_recording"]:
                # Plan-C protocol: normalize the scene before every episode so
                # training and deployment share the same starting distribution.
                if not args.skip_align:
                    # Use cv2.waitKey instead of input(): the keyboard listener
                    # spawned by init_keyboard_listener interferes with the
                    # console stdin on Windows, causing input() to raise EOFError
                    # which disconnects all hardware via the finally block.
                    fixed_cam = robot.cameras["fixed"]
                    print(f"\n[{recorded + 1}/{args.num_episodes}] Place the model, then press ENTER in the camera window to align...")
                    while True:
                        # Show live camera feed so the user can verify the
                        # dental model is within the scene camera's FOV.
                        if not fixed_cam.is_connected:
                            fixed_cam.connect()
                        try:
                            frame_rgb = fixed_cam.read()
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            # Draw ArUco detection overlay for positioning feedback.
                            if locator is not None:
                                frame_bgr = locator.draw_debug(frame_rgb)
                        except Exception as exc:
                            print(f"[record] camera read failed: {exc}")
                            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame_bgr, "Camera unavailable", (150, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        cv2.imshow("[ENTER=align] Scene Camera", frame_bgr)
                        key = cv2.waitKey(50) & 0xFF
                        if key in (13, 10):  # Enter
                            break
                        if key == 27:  # Esc -> stop session
                            events["stop_recording"] = True
                            break
                    cv2.destroyWindow("[ENTER=align] Scene Camera")
                    if events["stop_recording"]:
                        break

                    # USB cameras on Windows may be suspended by the OS during
                    # the wait. Re-check and reconnect before aligning.
                    if not fixed_cam.is_connected:
                        print("[record] scene camera dropped, reconnecting...")
                        fixed_cam.connect()

                    # Lift the wrist FIRST: after the previous episode the arm is
                    # still down near the model, and align_base rotates the base
                    # (ID 1) — doing that with the drill down would sweep it
                    # across the dental model.
                    pre_lift_wrist(robot)
                    align_base(robot, fixed_cam.read, locator=locator)

                    # Auto-reset arm joints (ID 2-6) to the calibrated start pose.
                    # shoulder_pan (ID 1) was just set by ArUco — leave it untouched.
                    # Pre-lift already ran above, so skip it to avoid lifting twice.
                    move_arm_to_start_pose(robot, do_pre_lift=False)

                # Same check before recording: the second wait can also
                # trigger a USB suspend.
                for cam in robot.cameras.values():
                    if not cam.is_connected:
                        print(f"[record] {cam} dropped, reconnecting...")
                        cam.connect()

                print("Arm at start pose. Press ENTER in the camera window to record (or ESC to stop)...")
                while True:
                    # Show both camera feeds during the pre-record wait.
                    for cam_name, cam in robot.cameras.items():
                        try:
                            frame_rgb = cam.read()
                            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                            cv2.imshow(f"[ENTER=record] {cam_name}", frame_bgr)
                        except Exception as exc:
                            print(f"[record] {cam_name} read failed: {exc}")
                    key = cv2.waitKey(50) & 0xFF
                    if key in (13, 10):
                        break
                    if key == 27:
                        events["stop_recording"] = True
                        break
                for cam_name in robot.cameras:
                    cv2.destroyWindow(f"[ENTER=record] {cam_name}")
                if events["stop_recording"]:
                    break
                # Clear stale episode events: a SPACE pressed during any of the
                # wait windows above would leave exit_early=True and make
                # record_loop break before recording a single frame.
                events["exit_early"] = False
                events["rerecord_episode"] = False
                log_say(f"Recording episode {dataset.num_episodes}", play_sounds=False)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=args.episode_time,
                    single_task=args.task,
                    display_data=args.display_data,
                )

                if events["rerecord_episode"]:
                    log_say("Re-recording episode", play_sounds=False)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                if dataset.episode_buffer is None or dataset.episode_buffer["size"] == 0:
                    # record_loop exited without any frame (e.g. stale SPACE
                    # event); skip saving instead of crashing the session.
                    print("[record] episode buffer empty, skipping save")
                    if dataset.episode_buffer is not None:
                        dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1
                print(f"Episode saved ({recorded}/{args.num_episodes})")

                # Export joint trajectories to CSV for offline analysis.
                if args.dataset_root:
                    ep_idx = dataset.meta.total_episodes - 1
                    export_joint_trajectories(Path(args.dataset_root), ep_idx)
    finally:
        if args.display_data and rr is not None:
            rr.rerun_shutdown()
        # Disconnect is wrapped in try/except because the CH343 serial bus
        # can throw Overload errors when disabling torque on a stalled motor.
        # These are non-fatal: the motors will power off when the bus closes.
        try:
            robot.disconnect()
        except RuntimeError as exc:
            print(f"[record] robot disconnect error (non-fatal): {exc}")
        try:
            teleop.disconnect()
        except RuntimeError as exc:
            print(f"[record] leader disconnect error (non-fatal): {exc}")
        if not is_headless() and listener is not None:
            listener.stop()

    if args.push_to_hub:
        dataset.push_to_hub(tags=["dental", "act", "so101"], private=False)

    print(f"\nDone. {recorded} episode(s) recorded in dataset '{args.dataset_repo_id}'.")
    print(f"Train with: python -m dental_robot.train_act --dataset_repo_id {args.dataset_repo_id}")


if __name__ == "__main__":
    main()
