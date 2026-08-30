#!/usr/bin/env python3
"""PROTOTYPE - not part of the mainstream flat-bug pipeline.

Train a 1024px U-Net predicting arthropod foreground and instance outline.

Two channels, both binary: foreground (any arthropod) and outline (the contour ring of
each instance). Subtracting the outline from the foreground leaves one connected core per
animal, which seeds a watershed - so the outline channel is what separates touching
instances without any detection head, NMS, or scale pyramid.

Full input resolution throughout, unlike the YOLO mask head whose prototypes live at
imgsz/4 and destroy 32% of appendage pixels before the loss ever sees them.

Usage:
    train.py -d DATA_ROOT [-e EPOCHS] [-b BATCH] [--encoder tu-convnext_tiny] [-o OUT]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import TileSegDataset, WindowSampler  # noqa: E402


def dice_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor,
             valid: torch.Tensor) -> torch.Tensor:
    """Soft-Dice plus weighted BCE, summed over channels.

    Dice keeps the loss meaningful when foreground is ~12% of pixels and outline ~1%;
    BCE alone would be dominated by background.

    Args:
        logits: Raw model output, (B, 2, H, W).
        target: Binary targets, (B, 2, H, W).
        pos_weight: Per-channel positive weight for the BCE term, (2,).
        valid: (B, 1, H, W) mask; 0 where the crop was reflect-padded and carries no
            annotation, so those pixels must not be scored as background.

    Returns:
        Scalar loss.
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight.view(1, -1, 1, 1), reduction="none"
    )
    bce = (bce * valid).sum() / valid.expand_as(bce).sum().clamp_min(1.0)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    dims = (0, 2, 3)
    dice = 1 - (2 * (p * t).sum(dims) + 1.0) / (p.sum(dims) + t.sum(dims) + 1.0)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Compute per-channel IoU and F1 at threshold 0.5.

    Args:
        model: The network.
        loader: Validation dataloader.
        device: Torch device.

    Returns:
        Dict of metric name to value.
    """
    model.eval()
    tp = torch.zeros(2, device=device)
    fp = torch.zeros(2, device=device)
    fn = torch.zeros(2, device=device)
    for x, y, v in loader:
        x, y, v = x.to(device, non_blocking=True), y.to(device, non_blocking=True), v.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = (torch.sigmoid(model(x)) > 0.5).float() * v
        y = y * v
        tp += (p * y).sum((0, 2, 3))
        fp += (p * (1 - y)).sum((0, 2, 3))
        fn += ((1 - p) * y).sum((0, 2, 3))
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    model.train()
    return {"fg_iou": iou[0].item(), "ol_iou": iou[1].item(), "fg_f1": f1[0].item(), "ol_f1": f1[1].item()}


def main() -> None:
    """Train the model and write checkpoints."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--data", required=True, help="root with images/{train,val} and labels/{train,val}")
    ap.add_argument("-o", "--out", default="runs/semseg", help="output directory")
    ap.add_argument("-e", "--epochs", type=int, default=20)
    ap.add_argument("-b", "--batch", type=int, default=4)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--encoder", default="tu-convnext_tiny")
    ap.add_argument("--steps", type=int, default=4000,
                    help="crops per EPOCH (a rotating window; full image coverage accumulates "
                         "over len(coverage)/steps epochs)")
    ap.add_argument("--val-steps", type=int, default=800, help="crops per validation pass")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = smp.Unet(a.encoder, encoder_weights="imagenet", in_channels=3, classes=2).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net / {a.encoder}: {n_par:.1f}M params, tile {a.tile}, batch {a.batch}", flush=True)

    tr_ds = TileSegDataset(a.data, "train", a.tile)
    va_ds = TileSegDataset(a.data, "val", a.tile, seed=1234)
    tr_sampler = WindowSampler(len(tr_ds), a.steps)
    va_sampler = WindowSampler(len(va_ds), a.val_steps)
    cycle = max(1, round(len(tr_ds) / a.steps))
    print(f"coverage list = {len(tr_ds)} crops over {len(tr_ds.images)} images; "
          f"epoch = {a.steps} crops, so every image is seen once per {cycle} epochs. "
          f"val = {a.val_steps} crops (fixed window).", flush=True)
    tr = DataLoader(tr_ds, batch_size=a.batch, sampler=tr_sampler, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    va = DataLoader(va_ds, batch_size=a.batch, sampler=va_sampler,
                    num_workers=max(2, a.workers // 2), pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                               total_steps=a.epochs * (len(tr) + 1) + 1)
    # Outline is ~1% of pixels against ~12% foreground, so it needs the heavier positive weight.
    pw = torch.tensor([2.0, 8.0], device=dev)

    best = -1.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_sampler.set_epoch(ep - 1)
        tot = n = 0
        for x, y, v in tr:
            x, y, v = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), v.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = dice_bce(model(x), y, pw, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        m = evaluate(model, va, dev)
        score = m["fg_iou"] + m["ol_iou"]
        torch.save({"model": model.state_dict(), "encoder": a.encoder, "tile": a.tile, "epoch": ep, "metrics": m},
                   os.path.join(a.out, "last.pt"))
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "encoder": a.encoder, "tile": a.tile, "epoch": ep, "metrics": m},
                       os.path.join(a.out, "best.pt"))
        lr_now = opt.param_groups[0]["lr"]
        print(f"ep {ep:3d}/{a.epochs}  lr {lr_now:.2e}  loss {tot / max(n, 1):.4f}  "
              f"fg IoU {m['fg_iou']:.4f}  outline IoU {m['ol_iou']:.4f}  "
              f"fg F1 {m['fg_f1']:.4f}  outline F1 {m['ol_f1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done. best fg+outline IoU {best:.4f} -> {a.out}/best.pt")


if __name__ == "__main__":
    main()
