"""Minimal single-file Mask2Former trainer for flatbug (prototype).

Uses HuggingFace ``Mask2FormerForUniversalSegmentation`` with the smallest
public instance-seg checkpoint (Swin-Tiny). Not distributed, not multi-scale,
no EMA — but the data pipeline is flatbug's own, so training sees the same crops,
rescaling, colour jitter, flips and inpainting the YOLO models are trained on.

Usage:
    fb_train_m2f -d /path/to/prepared/data -o runs/m2f --epochs 5 --batch 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from flat_bug import logger, set_log_level
from flat_bug.mask2former.data import (
    BackgroundMixDataset,
    FlatBugM2FDataset,
    build_augmented_dataset,
    build_background_dataset,
    collate,
)

DEFAULT_CHECKPOINT = "facebook/mask2former-swin-tiny-coco-instance"


def build_model(num_classes: int = 1, checkpoint: str = DEFAULT_CHECKPOINT):
    """Load pretrained Mask2Former and re-init the classification head for ``num_classes``."""
    from transformers import Mask2FormerForUniversalSegmentation

    id2label = {i: f"class_{i}" for i in range(num_classes)}
    id2label[0] = "insect"
    label2id = {v: k for k, v in id2label.items()}
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move a collated batch (stacked pixels, ragged mask/class lists) onto ``device``."""
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "mask_labels": [m.to(device, non_blocking=True) for m in batch["mask_labels"]],
        "class_labels": [c.to(device, non_blocking=True) for c in batch["class_labels"]],
    }


def autocast_context(device: torch.device, amp: bool):
    """bfloat16 autocast on CUDA when enabled; a no-op otherwise.

    bfloat16 rather than float16: no `GradScaler`, and the set-prediction loss sums
    many small terms, where fp16 underflow is a real risk.
    """
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda")


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    log_every: int = 50,
    scheduler=None,
    clip_grad: float = 0.0,
    amp: bool = False,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total, n = 0.0, 0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with autocast_context(device, amp):
            loss = model(**batch).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
        n += 1
        if step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"epoch {epoch} step {step}/{len(loader)}  loss={loss.item():.4f}  lr={lr:.2e}")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, amp: bool = False) -> float:
    """Return the mean loss over the validation loader."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, amp):
            total += model(**batch).loss.item()
        n += 1
    return total / max(n, 1)


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay, stepped per optimizer step."""

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def build_split(args: argparse.Namespace, split: str):
    """Build one split, with flatbug's augmentation pipeline unless it was turned off."""
    if args.no_augment:
        return FlatBugM2FDataset(args.data_dir, split=split, image_size=args.image_size)
    return build_augmented_dataset(
        args.data_dir,
        split=split,
        image_size=args.image_size,
        max_instances=args.max_instances,
        batch_size=args.batch,
        max_images=args.max_images,
        exclude_datasets=args.exclude_datasets,
    )


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-o", "--output-dir", default="runs/mask2former")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--val", action="store_true", help="Also run validation each epoch")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Use the deterministic center-crop dataset instead of flatbug's augmentation pipeline",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None, help="Cap on the number of instances kept per sample"
    )
    parser.add_argument(
        "--background-dir",
        default=None,
        help="Directory of insect-free images to mix into training as out-of-domain negatives",
    )
    parser.add_argument(
        "--background-fraction",
        type=float,
        default=0.2,
        help="Proportion of each training epoch drawn from --background-dir",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable bfloat16 autocast on CUDA")
    parser.add_argument(
        "--val-samples",
        type=int,
        default=2000,
        help="Cap on validation samples per epoch (the oversampled val split is very large). -1 for all.",
    )
    parser.add_argument("--warmup", type=int, default=500, help="Linear LR warmup steps")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient-norm clip; 0 disables")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume weights and epoch from")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only warn; suppress per-step progress")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Keep only the first N images per split"
    )
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=None,
        help="Sub-dataset name prefixes to hold out of training, as in fb_train",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # A trainer that runs for hours should say what it is doing; flatbug's logger
    # defaults to WARNING, which would make the whole run silent.
    set_log_level("WARNING" if args.quiet else "INFO")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = build_split(args, "train")
    if args.background_dir:
        # Out-of-domain negatives: in-domain empty crops cannot teach the model to
        # ignore printed text, packaging or debris it has never seen.
        background_set = build_background_dataset(
            args.background_dir, image_size=args.image_size, batch_size=args.batch
        )
        train_set = BackgroundMixDataset(train_set, background_set, background_fraction=args.background_fraction)
        logger.info(
            f"background: {len(background_set)} samples from {args.background_dir} "
            f"mixed at {args.background_fraction:.0%}"
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )
    pipeline = "center-crop (no augmentation)" if args.no_augment else "flatbug augmentations"
    logger.info(f"train: {len(train_set)} samples [{pipeline}]")

    val_loader = None
    if args.val:
        val_set = build_split(args, "val")
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=True,
        )
        if 0 < args.val_samples < len(val_set):
            # The validation split is oversampled and resampled 5x; a fixed subset keeps
            # per-epoch validation affordable and comparable across epochs.
            generator = torch.Generator().manual_seed(0)
            indices = torch.randperm(len(val_set), generator=generator)[: args.val_samples].tolist()
            val_set = Subset(val_set, indices)
        logger.info(f"val:   {len(val_set)} samples")

    device = torch.device(args.device)
    model = build_model(num_classes=1, checkpoint=args.checkpoint).to(device)
    # Stored alongside the weights so that `fb_predict_m2f` can rebuild the architecture
    # without being told the base checkpoint, class count and tile size again.
    meta = {"base_checkpoint": args.checkpoint, "num_classes": 1, "image_size": args.image_size}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        # Restoring Adam's moments matters: without them a resume restarts the
        # optimizer cold, which shows up as a loss spike for the first few hundred
        # steps. Older checkpoints predate this, hence the guard.
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"resumed {args.resume} at epoch {start_epoch} (with optimizer state)")
        else:
            logger.warning(f"resumed {args.resume} at epoch {start_epoch}; no optimizer state in checkpoint")

    amp = not args.no_amp
    total_steps = max(1, len(train_loader) * (args.epochs - start_epoch))
    scheduler = cosine_schedule_with_warmup(optimizer, min(args.warmup, total_steps // 10), total_steps)
    logger.info(f"{total_steps} optimizer steps, bf16={amp and device.type == 'cuda'}, clip={args.clip_grad}")

    best = float("inf")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            scheduler=scheduler, clip_grad=args.clip_grad, amp=amp,
        )
        msg = f"epoch {epoch} train_loss={train_loss:.4f}"
        score = train_loss
        if val_loader is not None:
            val_loss = validate(model, val_loader, device, amp=amp)
            msg += f" val_loss={val_loss:.4f}"
            score = val_loss
        logger.info(msg)

        if score < best:
            best = score
            ckpt = out_dir / "best.pt"
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "score": score, **meta},
                ckpt,
            )
            logger.info(f"saved {ckpt}")

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "epoch": args.epochs - 1, **meta},
        out_dir / "last.pt",
    )


if __name__ == "__main__":
    main()
