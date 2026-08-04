# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Generate portfolio-ready figures from the dental drilling dataset.

Outputs (all under outputs/figures/):
  dataset_overview.png      episode lengths, joint-angle histograms, summary
  joint_trajectories.png    all episodes overlaid, one subplot per joint
  smoothing_summary.png     per-episode jitter before/after Savitzky-Golay
  task_keyframes.png        fixed + handeye camera frames (start/mid/end)
  training_curves.png       train/val loss curves (needs train_metrics.csv)

Usage:
    python -m dental_robot.make_figures [--dataset_root ./data/dental] [--episode 49]
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from dental_robot.smooth_trajectories import roughness

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
FPS = 30
FIG_DPI = 150


def load_episode(root: Path, ep: int) -> np.ndarray:
    path = root / "data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet"
    return np.array(pq.read_table(path).column("action").to_pylist(), dtype=np.float32)


def fig_dataset_overview(root: Path, episodes: dict, out: Path) -> None:
    lengths = {ep: len(traj) / FPS for ep, traj in episodes.items()}
    all_actions = np.concatenate(list(episodes.values()), axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Dental Drilling Dataset Overview (LeRobot v3.0 format)",
                 fontsize=14, fontweight="bold")

    # (a) episode lengths
    ax = axes[0, 0]
    ax.bar(lengths.keys(), lengths.values(), color="#4878cf", width=0.8)
    ax.set_xlabel("episode index")
    ax.set_ylabel("duration (s)")
    ax.set_title(f"Episode lengths — {len(lengths)} episodes, "
                 f"{sum(lengths.values()) / 60:.1f} min total")
    ax.grid(alpha=0.25, axis="y")

    # (b) joint angle distribution
    ax = axes[0, 1]
    for j, name in enumerate(JOINT_NAMES):
        ax.hist(all_actions[:, j], bins=60, alpha=0.55, label=name, color=JOINT_COLORS[j])
    ax.set_xlabel("joint position (deg)")
    ax.set_ylabel("frame count")
    ax.set_title("Joint angle distribution (all frames)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    # (c) per-frame joint velocity distribution (proxy for smoothness)
    ax = axes[1, 0]
    deltas = []
    for traj in episodes.values():
        deltas.append(np.abs(np.diff(traj, axis=0)) * FPS)  # deg/s
    deltas = np.concatenate(deltas, axis=0)
    for j, name in enumerate(JOINT_NAMES):
        ax.hist(deltas[:, j], bins=np.linspace(0, 60, 61), alpha=0.55,
                label=name, color=JOINT_COLORS[j])
    ax.set_xlabel("|joint velocity| (deg/s)")
    ax.set_ylabel("frame count")
    ax.set_title("Joint velocity distribution (smoothed data)")
    ax.set_xlim(0, 60)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    # (d) text summary
    ax = axes[1, 1]
    ax.axis("off")
    total_frames = sum(len(t) for t in episodes.values())
    summary = (
        f"Task: dental implant drilling (SO-101 arm)\n"
        f"Episodes:          {len(episodes)}\n"
        f"Total frames:      {total_frames:,}  (@ {FPS} fps)\n"
        f"Total duration:    {total_frames / FPS / 60:.1f} min\n"
        f"Joints:            6 (Feetech STS3215)\n"
        f"Cameras:           2 (scene + hand-eye, 640x480)\n"
        f"Pre-processing:    Savitzky-Golay trajectory smoothing\n"
        f"Policy:            ACT (Action Chunking Transformer)\n"
        f"Training:          50K steps, RTX 3060, AMP"
    )
    ax.text(0.05, 0.95, summary, fontsize=11, family="monospace",
            va="top", bbox=dict(boxstyle="round", fc="#f0f4f8", ec="#4878cf"))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[fig] {out}")


def fig_joint_trajectories(episodes: dict, out: Path) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Demonstrated Joint Trajectories — all episodes overlaid",
                 fontsize=14, fontweight="bold")
    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j]
        for ep, traj in episodes.items():
            t = np.arange(len(traj)) / FPS
            ax.plot(t, traj[:, j], color=JOINT_COLORS[j], alpha=0.28, lw=0.8)
        ax.set_ylabel(f"{name}\n(deg)", fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[fig] {out}")


def fig_smoothing_summary(root: Path, episodes: dict, out: Path) -> None:
    """Per-episode jitter (roughness) before vs after smoothing."""
    backup_root = root.parent / "backups" / f"{root.name}_pre_smooth" / "data" / "chunk-000"
    pairs = []
    for ep, smooth_traj in episodes.items():
        raw_path = backup_root / f"episode_{ep:06d}.parquet"
        if not raw_path.is_file():
            continue
        raw = np.array(pq.read_table(raw_path).column("action").to_pylist(), dtype=np.float32)
        if len(raw) != len(smooth_traj):
            continue
        pairs.append((ep, roughness(raw), roughness(smooth_traj)))
    if not pairs:
        print("[fig] smoothing_summary skipped (no raw backups found)")
        return

    eps = [p[0] for p in pairs]
    before = np.array([p[1] for p in pairs])
    after = np.array([p[2] for p in pairs])

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(eps))
    ax.bar(x - 0.2, before, width=0.4, color="#d62728", alpha=0.8, label="raw (teleop)")
    ax.bar(x + 0.2, after, width=0.4, color="#2ca02c", alpha=0.8, label="smoothed (Savitzky-Golay)")
    ax.set_xticks(x, [str(e) for e in eps], fontsize=7)
    ax.set_xlabel("episode index")
    ax.set_ylabel("trajectory roughness")
    reduction = (1 - after.sum() / before.sum()) * 100
    ax.set_title(f"Per-episode jitter before/after smoothing — {reduction:.1f}% total reduction "
                 f"({len(eps)} episodes)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[fig] {out}")


def _read_frames(video_path: Path, ratios=(0.1, 0.5, 0.9)) -> list:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for r in ratios:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * r))
        ok, frame = cap.read()
        frames.append(frame if ok else np.zeros((480, 640, 3), np.uint8))
    cap.release()
    return frames


def fig_task_keyframes(root: Path, episode: int, out: Path) -> None:
    cams = {"Scene camera (fixed)": "observation.images.fixed",
            "Hand-eye camera (wrist)": "observation.images.handeye"}
    ratios = (0.1, 0.5, 0.9)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
    fig.suptitle(f"Task execution key frames — episode {episode} "
                 "(start / mid / end of demonstration)", fontsize=14, fontweight="bold")
    for i, (title, key) in enumerate(cams.items()):
        video = root / "videos" / "chunk-000" / key / f"episode_{episode:06d}.mp4"
        if not video.is_file():
            print(f"[fig] keyframes: missing {video}, skipped")
            return
        for j, frame in enumerate(_read_frames(video, ratios)):
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t = {ratios[j] * 100:.0f}% of episode", fontsize=10)
            if j == 0:
                ax.set_ylabel(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[fig] {out}")


def fig_training_curves(output_dir: Path, out: Path) -> None:
    csv_path = output_dir / "train_metrics.csv"
    if not csv_path.is_file():
        print("[fig] training_curves skipped: outputs/train/act_dental/train_metrics.csv "
              "not found. Save your training console output to that CSV (or retrain) "
              "and re-run to plot loss curves.")
        return
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    train = [r for r in rows if r["kind"] == "train"]
    val = [r for r in rows if r["kind"] == "val"]
    if not train:
        print("[fig] training_curves skipped: no train rows in CSV")
        return

    ts = np.array([float(r["step"]) for r in train])
    tloss = np.array([float(r["loss"]) for r in train])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    ax.plot(ts, tloss, color="#4878cf", lw=1.2, label="train loss")
    if val:
        vs = np.array([float(r["step"]) for r in val])
        vloss = np.array([float(r["loss"]) for r in val])
        ax.plot(vs, vloss, "o-", color="#d62728", lw=1.5, ms=4, label="val loss")
        best_i = int(np.argmin(vloss))
        ax.axvline(vs[best_i], color="gray", ls="--", lw=0.8)
        ax.annotate(f"best val {vloss[best_i]:.4f}\n(step {int(vs[best_i])})",
                    (vs[best_i], vloss[best_i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training & validation loss", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.semilogy(ts, tloss, color="#4878cf", lw=1.2, label="train loss (log)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Convergence (log scale)", fontweight="bold")
    ax.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"[fig] {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", default="./data/dental")
    parser.add_argument("--episode", type=int, default=49,
                        help="Episode used for the key-frame figure")
    parser.add_argument("--output_dir", default="./outputs/figures")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_dir = root / "data" / "chunk-000"
    eps = sorted(int(p.stem.split("_")[1]) for p in parquet_dir.glob("episode_*.parquet"))
    print(f"[data] {len(eps)} episodes: {eps[0]}..{eps[-1]}")
    episodes = {ep: load_episode(root, ep) for ep in eps}

    fig_dataset_overview(root, episodes, out_dir / "dataset_overview.png")
    fig_joint_trajectories(episodes, out_dir / "joint_trajectories.png")
    fig_smoothing_summary(root, episodes, out_dir / "smoothing_summary.png")
    fig_task_keyframes(root, args.episode, out_dir / "task_keyframes.png")
    fig_training_curves(Path("outputs/train/act_dental"), out_dir / "training_curves.png")
    print("[done] all figures in", out_dir)


if __name__ == "__main__":
    main()
