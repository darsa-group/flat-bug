"""Minimal single-file Mask2Former trainer for flatbug (prototype).

Uses HuggingFace ``Mask2FormerForUniversalSegmentation`` with the smallest
public instance-seg checkpoint (Swin-Tiny). Not distributed, not multi-scale,
no EMA, no augmentation — just enough to prove that data → model → loss →
checkpoint works end-to-end.

Usage:
    fb_train_m2f -d /path/to/prepared/data -o runs/m2f --epochs 5 --batch 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flat_bug import logger
from flat_bug.mask2former.data import FlatBugM2FDataset, collate

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
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "mask_labels": [m.to(device, non_blocking=True) for m in batch["mask_labels"]],
        "class_labels": [c.to(device, non_blocking=True) for c in batch["class_labels"]],
    }


def train_one_epoch(model, loader, optimizer, device, epoch: int, log_every: int = 10) -> float:
    model.train()
    total, n = 0.0, 0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
        if step % log_every == 0:
            logger.info(f"epoch {epoch} step {step}/{len(loader)}  loss={loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(**batch)
        total += outputs.loss.item()
        n += 1
    return total / max(n, 1)


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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = FlatBugM2FDataset(args.data_dir, split="train", image_size=args.image_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )
    logger.info(f"train: {len(train_set)} images")

    val_loader = None
    if args.val:
        val_set = FlatBugM2FDataset(args.data_dir, split="val", image_size=args.image_size)
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate,
            pin_memory=True,
        )
        logger.info(f"val:   {len(val_set)} images")

    device = torch.device(args.device)
    model = build_model(num_classes=1, checkpoint=args.checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        msg = f"epoch {epoch} train_loss={train_loss:.4f}"
        score = train_loss
        if val_loader is not None:
            val_loss = validate(model, val_loader, device)
            msg += f" val_loss={val_loss:.4f}"
            score = val_loss
        logger.info(msg)

        if score < best:
            best = score
            ckpt = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, ckpt)
            logger.info(f"saved {ckpt}")

    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1}, out_dir / "last.pt")


if __name__ == "__main__":
    main()
