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

"""Train the single-task ACT policy on a recorded dental-implant dataset.

Built on top of the official training stack (see src/lerobot/scripts/train.py and
examples/3_train_policy.py): it reuses lerobot's `update_policy` (AMP + grad
clipping), `resolve_delta_timestamps`, `MetricsTracker` and the ACT optimizer
preset, and adds what the official script lacks for real-robot data:
  - an episode-held-out validation split with a tracked best checkpoint,
  - symlink-free checkpoint layout (Windows friendly),
  - optional Weights & Biases logging.

Usage:
    python -m dental_robot.train_act --dataset_repo_id ${HF_USER}/dental_implant
    python -m dental_robot.train_act --dataset_repo_id local/dental_implant \
        --dataset_root data/dental_implant --steps 50000 --batch_size 8 --wandb

    # resume an interrupted run
    python -m dental_robot.train_act --dataset_repo_id ... --resume

After training, deploy the checkpoint on the robot with:
    python -m dental_robot.run_pipeline \
        --policy_path outputs/train/act_dental/checkpoints/best/pretrained_model

IMPORTANT: every training episode must start from the canonical pose produced
by `align_base` (run phase 1 before each recording), otherwise the deployment
distribution will not match the training distribution.
"""

import argparse
import logging
import shutil
import time
from pathlib import Path

import torch
from torch.amp import GradScaler

from lerobot.configs.types import FeatureType
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import format_big_number, get_safe_torch_device, init_logging

DEFAULT_OUTPUT_DIR = Path("outputs/train/act_dental")


def make_policy_config(metadata: LeRobotDatasetMetadata, args: argparse.Namespace) -> ACTConfig:
    """Size the ACT policy from the dataset features (see examples/3_train_policy.py)."""
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        optimizer_lr=args.lr,
        device=args.device,
        use_amp=args.use_amp,
    )


def split_episodes(total_episodes: int, num_val: int) -> tuple[list[int], list[int]]:
    """Hold out the last `num_val` episodes for validation.

    The tail is used (rather than a random subset) so the split is reproducible
    and later re-recordings appended to the dataset stay in the val pool.
    """
    if num_val >= total_episodes:
        raise ValueError(f"--val_episodes {num_val} must be < total episodes {total_episodes}")
    episodes = list(range(total_episodes))
    if num_val == 0:
        return episodes, []
    return episodes[:-num_val], episodes[-num_val:]


def make_dataset(
    repo_id: str,
    root: Path | None,
    episodes: list[int],
    cfg: ACTConfig,
    metadata: LeRobotDatasetMetadata,
) -> LeRobotDataset:
    """Instantiate a LeRobotDataset (parquet + video) slice with ACT delta_timestamps."""
    delta_timestamps = resolve_delta_timestamps(cfg, metadata)
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
    )
    # Same normalization trick as the official factory: ImageNet stats for cameras.
    for key in dataset.meta.camera_keys:
        for stats_type, stats in IMAGENET_STATS.items():
            dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    return dataset


def save_checkpoint(
    checkpoint_dir: Path, policy: ACTPolicy, optimizer: torch.optim.Optimizer, step: int
) -> None:
    """Save policy + training state under `checkpoint_dir` (no symlinks, Windows safe)."""
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    policy.save_pretrained(checkpoint_dir / "pretrained_model")
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, checkpoint_dir / "training_state.pt")


def load_checkpoint(
    checkpoint_dir: Path, policy: ACTPolicy, optimizer: torch.optim.Optimizer, device: torch.device
) -> int:
    """Restore weights + optimizer state in-place, return the step to resume from."""
    weights = ACTPolicy.from_pretrained(checkpoint_dir / "pretrained_model").state_dict()
    policy.load_state_dict(weights)
    # weights_only=True: the state dict only holds tensors and plain Python types.
    state = torch.load(checkpoint_dir / "training_state.pt", map_location=device, weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


@torch.no_grad()
def validate(policy: ACTPolicy, dataloader: torch.utils.data.DataLoader, device: torch.device) -> dict:
    """Average forward losses over the held-out episodes.

    float64 accumulators: thousands of float32 losses summed naively would lose
    precision; the batch count is exact so the mean is stable.
    """
    policy.eval()
    sums: dict[str, float] = {}
    num_batches = 0
    for batch in dataloader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, loss_dict = policy.forward(batch)
        sums["loss"] = sums.get("loss", 0.0) + float(loss.item())
        for key, value in (loss_dict or {}).items():
            sums[key] = sums.get(key, 0.0) + float(value)
        num_batches += 1
    policy.train()
    if num_batches == 0:
        raise RuntimeError("Validation dataloader is empty, lower --batch_size or --val_episodes")
    return {key: value / num_batches for key, value in sums.items()}


def train(args: argparse.Namespace) -> None:
    init_logging()
    set_seed(args.seed)
    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, job_type="train", config=vars(args), dir=str(output_dir)
        )

    logging.info("Creating dataset")
    root = Path(args.dataset_root) if args.dataset_root else None
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=root)
    cfg = make_policy_config(metadata, args)
    train_episodes, val_episodes = split_episodes(metadata.total_episodes, args.val_episodes)
    train_dataset = make_dataset(args.dataset_repo_id, root, train_episodes, cfg, metadata)
    val_loader = None
    if val_episodes:
        val_dataset = make_dataset(args.dataset_repo_id, root, val_episodes, cfg, metadata)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

    logging.info("Creating policy")
    # Normalization stats come from the full-dataset metadata (train and val
    # share them, matching what `lerobot-record` will feed at deployment).
    policy = ACTPolicy(cfg, dataset_stats=train_dataset.meta.stats)
    policy.train()
    policy.to(device)

    # Official ACT preset: AdamW with a separate backbone param group.
    optimizer_cfg = cfg.get_optimizer_preset()
    optimizer = optimizer_cfg.build(policy.get_optim_params())
    grad_scaler = GradScaler(device.type, enabled=cfg.use_amp)

    step = 0
    if args.resume:
        step = load_checkpoint(checkpoints_dir / "last", policy, optimizer, device)
        logging.info(f"Resumed from {checkpoints_dir / 'last'} at step {step}")

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"{args.steps=} ({format_big_number(args.steps)})")
    logging.info(f"{train_dataset.num_frames=} ({format_big_number(train_dataset.num_frames)})")
    logging.info(f"{train_dataset.num_episodes=} (validation episodes: {val_episodes or 'none'})")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    dl_iter = cycle(train_loader)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        args.batch_size,
        train_dataset.num_frames,
        train_dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    best_val_loss = float("inf")

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, args.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=device.type == "cuda")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            optimizer_cfg.grad_clip_norm,
            grad_scaler=grad_scaler,
            use_amp=cfg.use_amp,
        )

        # Same convention as the official script: eval / checkpoint counters run
        # after the `step`th update has completed.
        step += 1
        train_tracker.step()
        is_log_step = args.log_freq > 0 and step % args.log_freq == 0
        is_saving_step = step % args.save_freq == 0 or step == args.steps
        is_eval_step = val_loader is not None and args.eval_freq > 0 and step % args.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_run:
                wandb_log_dict = {f"train/{k}": v for k, v in train_tracker.to_dict().items()}
                if output_dict:
                    wandb_log_dict.update({f"train/{k}": v for k, v in output_dict.items()})
                wandb_run.log(wandb_log_dict, step=step)
            train_tracker.reset_averages()

        if is_eval_step:
            val_losses = validate(policy, val_loader, device)
            val_msg = " ".join(f"{k}:{v:.4f}" for k, v in val_losses.items())
            logging.info(f"Validation at step {step}: {val_msg}")
            if wandb_run:
                wandb_run.log({f"val/{k}": v for k, v in val_losses.items()}, step=step)
            if val_losses["loss"] < best_val_loss:
                best_val_loss = val_losses["loss"]
                logging.info(f"New best validation loss {best_val_loss:.4f}, saving best checkpoint")
                save_checkpoint(checkpoints_dir / "best", policy, optimizer, step)

        if is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            save_checkpoint(checkpoints_dir / f"{step:06d}", policy, optimizer, step)
            save_checkpoint(checkpoints_dir / "last", policy, optimizer, step)

    # Without a val split, "best" falls back to the final model.
    if val_loader is None:
        save_checkpoint(checkpoints_dir / "best", policy, optimizer, step)

    if wandb_run:
        wandb_run.finish()
    logging.info("End of training")
    logging.info(
        f"Deploy with: python -m dental_robot.run_pipeline --policy_path {checkpoints_dir / 'best' / 'pretrained_model'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset_repo_id", required=True, help="Dataset repo id used during recording")
    parser.add_argument("--dataset_root", default=None, help="Local dataset dir (default: HF cache)")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to write checkpoints")
    parser.add_argument("--steps", type=int, default=50_000, help="Number of optimizer updates")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="ACT preset learning rate")
    parser.add_argument("--chunk_size", type=int, default=100, help="ACT action chunk size")
    parser.add_argument("--n_action_steps", type=int, default=100, help="Actions executed per chunk")
    parser.add_argument(
        "--val_episodes", type=int, default=2, help="Episodes held out for validation (0 = off)"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=1_000, help="Validation frequency in steps")
    parser.add_argument("--save_freq", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_amp", action="store_true", help="Mixed-precision training (CUDA only)")
    parser.add_argument("--resume", action="store_true", help="Resume from <output_dir>/checkpoints/last")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="dental_robot", help="WandB project name")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
