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

"""Plot raw vs smoothed joint trajectories for one episode.

Reads the original parquet (kept by smooth_trajectories --backup) and the
smoothed parquet, then renders a per-joint overlay figure.

Usage:
    python -m dental_robot.plot_smoothing_comparison --episode 5
    python -m dental_robot.plot_smoothing_comparison --episode 5 --key action
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from dental_robot.smooth_trajectories import roughness

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
FPS = 30


def main():
    """CLI entry point: plot raw vs smoothed joint curves for one episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default="./data/dental")
    parser.add_argument("--episode", type=int, default=5, help="Episode index to compare")
    parser.add_argument("--key", default="observation.state", choices=["observation.state", "action"])
    parser.add_argument("--output", default=None, help="Output png path (default: outputs/...)")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    chunk = args.episode // 1000
    fname = f"episode_{args.episode:06d}.parquet"
    raw_path = root.parent / "backups" / f"{root.name}_pre_smooth" / "data" / f"chunk-{chunk:03d}" / fname
    smooth_path = root / "data" / f"chunk-{chunk:03d}" / fname
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Original data not found: {raw_path}\n"
            "Re-run smooth_trajectories with --backup to keep the raw copies."
        )

    raw = np.array(pq.read_table(raw_path).column(args.key).to_pylist(), dtype=np.float32)
    smooth = np.array(pq.read_table(smooth_path).column(args.key).to_pylist(), dtype=np.float32)
    assert len(raw) == len(smooth), "raw and smoothed episodes have different lengths"

    t = np.arange(len(raw)) / FPS
    n_joints = raw.shape[1]

    fig, axes = plt.subplots(n_joints, 1, figsize=(13, 2.1 * n_joints), sharex=True)
    fig.suptitle(
        f"Episode {args.episode} — {args.key}: raw vs Savitzky-Golay smoothed",
        fontsize=13, fontweight="bold",
    )

    for j in range(n_joints):
        ax = axes[j]
        r_before = roughness(raw[:, j : j + 1])
        r_after = roughness(smooth[:, j : j + 1])
        ax.plot(t, raw[:, j], color="tab:red", lw=0.7, alpha=0.75, label="raw")
        ax.plot(t, smooth[:, j], color="tab:blue", lw=1.6, label="smoothed")
        ax.set_ylabel(JOINT_NAMES[j], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
        ax.set_title(f"jitter {r_before:.4f} → {r_after:.4f}  ({(1 - r_after / r_before) * 100:.1f}% reduced)",
                     fontsize=8, loc="left")

    axes[-1].set_xlabel("time (s)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.output) if args.output else Path("outputs") / f"smoothing_comparison_episode_{args.episode}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
