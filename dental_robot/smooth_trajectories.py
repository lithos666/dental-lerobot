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

"""Smooth servo joint trajectories before training (MANDATORY preprocessing).

Recorded episodes contain high-frequency jitter in the joint curves (servo
PID chasing / imperfect teleoperation). Training directly on jittery data
makes the learned policy reproduce the jitter, so the servos oscillate
during deployment. This script removes that jitter offline, in place.

Method: Savitzky-Golay filter applied independently to each joint column of
`action` and `observation.state`, per episode (Savitzky & Golay, 1964 —
least-squares polynomial fit in a sliding window; preserves the shape and
extrema of the trajectory better than a plain moving average).

What it updates:
    - data/chunk-*/episode_*.parquet   (smoothed action + observation.state)
    - meta/episodes_stats.jsonl        (recomputed for the smoothed features)
    - joint_trajectories/*.csv         (re-exported, with --export_csv)

Videos and everything else are left untouched.

Usage:
    # Inspect first (no writes): prints per-episode jitter reduction.
    python -m dental_robot.smooth_trajectories --dry_run

    # Apply (keeps a copy of the original parquet files first).
    python -m dental_robot.smooth_trajectories --dataset_root ./data/dental --backup

Then train as usual:
    python -m dental_robot.train_act --dataset_repo_id local/dental_drilling \
        --dataset_root ./data/dental
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter

from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.utils import (
    load_episodes_stats,
    load_info,
    serialize_dict,
    write_jsonlines,
)

DEFAULT_KEYS = ["action", "observation.state"]
EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")


def roughness(curve: np.ndarray) -> float:
    """Mean |second discrete derivative| — a proxy for high-frequency jitter."""
    if len(curve) < 3:
        return 0.0
    return float(np.mean(np.abs(np.diff(curve, n=2, axis=0))))


def smooth_episode_arrays(
    arrays: dict[str, np.ndarray], window: int, polyorder: int
) -> dict[str, np.ndarray]:
    """Savitzky-Golay each joint column along time; clamp window to episode length."""
    smoothed = {}
    for key, arr in arrays.items():
        n = len(arr)
        # window must be odd and > polyorder; shrink for very short episodes.
        w = min(window, n if n % 2 == 1 else n - 1)
        if w < polyorder + 2:
            print(f"  [skip] {key}: episode too short ({n} frames)")
            smoothed[key] = arr
            continue
        smoothed[key] = savgol_filter(arr, window_length=w, polyorder=polyorder, axis=0).astype(arr.dtype)
    return smoothed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_root", default="./data/dental", help="Local dataset dir (default: ./data/dental)")
    parser.add_argument("--window", type=int, default=15, help="Savitzky-Golay window in frames (odd; 15 = 0.5s at 30fps)")
    parser.add_argument("--polyorder", type=int, default=2, help="Polynomial order for the Savitzky-Golay fit")
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="Features to smooth")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Only smooth these episode indices")
    parser.add_argument("--dry_run", action="store_true", help="Report jitter reduction without writing anything")
    parser.add_argument("--backup", action="store_true", help="Copy original parquet files to <dataset_root>/../backups/<dataset_name>_pre_smooth (outside the dataset root, so it doesn't break save_episode's parquet count assertion)")
    parser.add_argument("--export_csv", action="store_true", help="Re-export joint_trajectories/*.csv from smoothed data")
    args = parser.parse_args()

    if args.window % 2 == 0:
        raise ValueError("--window must be odd")

    root = Path(args.dataset_root).resolve()
    info = load_info(root)
    features = info["features"]
    for key in args.keys:
        if key not in features:
            raise KeyError(f"Feature '{key}' not in dataset features: {list(features)}")

    # Locate episode parquet files from the meta (robust to chunking).
    from lerobot.datasets.utils import load_episodes

    episodes_meta = load_episodes(root)
    episode_indices = sorted(episodes_meta.keys())
    if args.episodes is not None:
        episode_indices = [e for e in episode_indices if e in args.episodes]

    chunk_size = info["chunks_size"]
    data_path_tpl = info["data_path"]

    print(f"Smoothing {len(episode_indices)} episode(s) under {root} "
          f"(window={args.window}, polyorder={args.polyorder}, keys={args.keys})")
    if args.dry_run:
        print("[dry_run] no files will be written")

    episodes_stats = load_episodes_stats(root)
    total_before, total_after = 0.0, 0.0

    for ep_idx in episode_indices:
        chunk = ep_idx // chunk_size
        parquet_path = root / data_path_tpl.format(episode_chunk=chunk, episode_index=ep_idx)
        if not parquet_path.is_file():
            print(f"  [skip] episode {ep_idx}: parquet not found ({parquet_path})")
            continue

        table = pq.read_table(parquet_path)
        arrays = {key: np.array(table.column(key).to_pylist(), dtype=np.float32) for key in args.keys}

        before = sum(roughness(a) for a in arrays.values())
        smoothed = smooth_episode_arrays(arrays, args.window, args.polyorder)
        after = sum(roughness(a) for a in smoothed.values())
        total_before += before
        total_after += after
        print(f"  episode {ep_idx}: jitter {before:.4f} -> {after:.4f} "
              f"({(1 - after / before) * 100 if before else 0:.1f}% reduced)")

        if args.dry_run:
            continue

        # Write the smoothed columns back, keeping every other column untouched.
        for key, arr in smoothed.items():
            col_idx = table.schema.get_field_index(key)
            table = table.set_column(col_idx, key, pa.array(arr.tolist(), type=table.schema.field(key).type))
        if args.backup:
            # Backups must live OUTSIDE dataset_root: save_episode() verifies
            # `rglob("*.parquet")` count == num_episodes, and a backup inside
            # the root would make the assertion fail.
            backup_root = root.parent / "backups" / f"{root.name}_pre_smooth"
            backup_path = backup_root / parquet_path.relative_to(root)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parquet_path, backup_path)
        pq.write_table(table, parquet_path)

        # Refresh the per-episode stats of the smoothed features (normalization
        # used by ACT training); video/image stats stay as they were.
        subset_features = {key: features[key] for key in args.keys}
        new_stats = compute_episode_stats(smoothed, subset_features)
        episodes_stats[ep_idx].update(new_stats)

    if not args.dry_run:
        # Rewrite episodes_stats.jsonl (serialize_dict converts numpy -> lists).
        stats_records = [
            {"episode_index": ep_idx, "stats": serialize_dict(stats)}
            for ep_idx, stats in sorted(episodes_stats.items())
        ]
        write_jsonlines(stats_records, root / EPISODES_STATS_PATH)
        print(f"Updated {root / EPISODES_STATS_PATH}")

        if args.export_csv:
            from dental_robot.record_episodes import export_joint_trajectories

            for ep_idx in episode_indices:
                export_joint_trajectories(root, ep_idx)

    print(f"\nTotal jitter: {total_before:.4f} -> {total_after:.4f} "
          f"({(1 - total_after / total_before) * 100 if total_before else 0:.1f}% reduced)")
    if args.dry_run:
        print("Dry run finished. Re-run without --dry_run to apply.")


if __name__ == "__main__":
    main()
