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
from seam_weight import weight_from_seam  # noqa: E402


def dice_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor,
             valid: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Soft-Dice plus weighted BCE, summed over channels.

    Dice keeps the loss meaningful when foreground is ~12% of pixels and outline ~1%;
    BCE alone would be dominated by background.

    Args:
        logits: Raw model output, (B, 2, H, W).
        target: Binary targets, (B, 2, H, W).
        pos_weight: Per-channel positive weight for the BCE term, (2,).
        valid: (B, 1, H, W) BINARY mask; 0 where the crop was reflect-padded and carries
            no annotation, so those pixels must not be scored as background. Must stay
            binary: it also masks the Dice term, where a non-binary multiplier would take
            the predictions outside [0, 1] and drive Dice negative.
        weight: Optional (B, 1, H, W) per-pixel weight applied to the BCE term only, e.g.
            emphasis around inter-instance seams. Defaults to ``valid``.

    Returns:
        Scalar loss.
    """
    w = valid if weight is None else weight
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight.view(1, -1, 1, 1), reduction="none"
    )
    bce = (bce * w).sum() / w.expand_as(bce).sum().clamp_min(1.0)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    dims = (0, 2, 3)
    dice = 1 - (2 * (p * t).sum(dims) + 1.0) / (p.sum(dims) + t.sum(dims) + 1.0)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, loader, device, pos_weight: torch.Tensor | None = None) -> dict:
    """Compute per-channel IoU and F1 at threshold 0.5, plus the validation loss.

    The validation loss is deliberately UNWEIGHTED - no seam, instance-area or background
    weighting - so it is comparable across arms that train with different weightings, and so
    it measures generalisation rather than how well each arm optimised its own objective.
    IoU and F1 are thresholded at 0.5 and are therefore blind to calibration drift, which is
    exactly what a rising validation loss against a falling training loss would reveal.

    Args:
        model: The network.
        loader: Validation dataloader.
        device: Torch device.
        pos_weight: Per-channel positive weight for the BCE term, matching training.

    Returns:
        Dict of metric name to value.
    """
    model.eval()
    vl = 0.0
    nb = 0
    tp = torch.zeros(2, device=device)
    fp = torch.zeros(2, device=device)
    fn = torch.zeros(2, device=device)
    for x, y, v in loader:
        x, y, v = x.to(device, non_blocking=True), y.to(device, non_blocking=True), v.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)[:, :2]
            if pos_weight is not None:
                vl += float(dice_bce(out, y[:, :2], pos_weight, v))
                nb += 1
            p = (torch.sigmoid(out) > 0.5).float() * v
        y = y[:, :2] * v
        tp += (p * y).sum((0, 2, 3))
        fp += (p * (1 - y)).sum((0, 2, 3))
        fn += ((1 - p) * y).sum((0, 2, 3))
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    model.train()
    return {"fg_iou": iou[0].item(), "ol_iou": iou[1].item(), "fg_f1": f1[0].item(), "ol_f1": f1[1].item(),
            "val_loss": vl / max(nb, 1)}


def main() -> None:
    """Train the model and write checkpoints."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--data", required=True, help="root with images/{train,val} and labels/{train,val}")
    ap.add_argument("-o", "--out", default="runs/semseg", help="output directory")
    ap.add_argument("-e", "--epochs", type=int, default=20)
    ap.add_argument("-b", "--batch", type=int, default=4)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--encoder", default="tu-convnext_tiny",
                    help="tu-convnext_tiny (local context) or mit_b2 (global self-attention)")
    ap.add_argument("--dist-channel", action="store_true",
                    help="predict a per-instance distance map as a third channel, for watershed markers")
    ap.add_argument("--inst-weight", action="store_true",
                    help="weight the loss by 1/sqrt(instance area), so small animals are not drowned")
    ap.add_argument("--dist-gain", type=float, default=1.0, help="weight of the distance-map L1 term")
    ap.add_argument("--bg-gamma", type=float, default=0.0,
                    help="penalise predicted foreground by distance from any real instance; 0 disables")
    ap.add_argument("--bg-saturate", type=float, default=50.0,
                    help="distance in px at which the background penalty saturates")
    ap.add_argument("--steps", type=int, default=4000,
                    help="crops per EPOCH (a rotating window; full image coverage accumulates "
                         "over len(coverage)/steps epochs)")
    ap.add_argument("--val-steps", type=int, default=800, help="crops per validation pass")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seam-weight", type=float, default=0.0,
                    help="w0 for Ronneberger-style weighting around inter-instance seams. "
                         "0 disables it and reproduces the unweighted run exactly. "
                         "Measured signal share in the seam neighbourhood: w0=30 -> 1.2%%, 100 -> 3.5%%.")
    ap.add_argument("--seam-sigma", type=float, default=6.0, help="spatial spread of the seam weight, px")
    ap.add_argument("--synth-bank", default=None, help="crop-bank dir; enables synthetic scenes")
    ap.add_argument("--synth-cache", default=None, help="background cache dir for synthetic scenes")
    ap.add_argument("--synth-prob", type=float, default=0.0, help="fraction of TRAIN crops that are composed scenes")
    ap.add_argument("--synth-touch-prob", type=float, default=0.92)
    ap.add_argument("--synth-coverage", type=float, default=0.30)
    ap.add_argument("--blur", type=float, default=0.0, help="probability of Gaussian blur, sigma 0.4-2.0")
    ap.add_argument("--noise", type=float, default=0.0, help="probability of Gaussian noise, 1-6%% of crop contrast")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="probability of arbitrary-angle rotation; target is re-rasterised from rotated polygons")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_pred = 3 if a.dist_channel else 2
    model = smp.Unet(a.encoder, encoder_weights="imagenet", in_channels=3, classes=n_pred).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net / {a.encoder}: {n_par:.1f}M params, tile {a.tile}, batch {a.batch}, "
          f"{n_pred} predicted channels", flush=True)

    import dataset as _ds  # module-level knobs, so workers inherit them through the fork
    _ds.P_BLUR, _ds.P_NOISE, _ds.P_ROTATE = a.blur, a.noise, a.rotate
    tr_ds = TileSegDataset(a.data, "train", a.tile, seam_channel=a.seam_weight > 0,
                           dist_channel=a.dist_channel, inst_weight=a.inst_weight,
                           bg_gamma=a.bg_gamma, bg_saturate=a.bg_saturate,
                           synth_bank=a.synth_bank, synth_cache=a.synth_cache,
                           synth_prob=a.synth_prob, synth_touch_prob=a.synth_touch_prob,
                           synth_coverage=a.synth_coverage)
    va_ds = TileSegDataset(a.data, "val", a.tile, seed=1234, dist_channel=a.dist_channel)
    tr_sampler = WindowSampler(len(tr_ds), a.steps)
    va_sampler = WindowSampler(len(va_ds), a.val_steps)
    cycle = max(1, round(len(tr_ds) / a.steps))
    print(f"seam weight w0={a.seam_weight} sigma={a.seam_sigma}; synthetic scenes p={tr_ds.synth_prob}; "
          f"blur p={a.blur} noise p={a.noise} rotate p={a.rotate}; "
          f"dist_channel={a.dist_channel} inst_weight={a.inst_weight} "
          f"bg_gamma={a.bg_gamma} (saturating at {a.bg_saturate}px)", flush=True)
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
            # NOTE: weights are kept separate from `v`. Folding them into the validity mask
            # breaks the Dice term and the metrics, both of which need a binary mask.
            w = None
            if tr_ds.idx_seam is not None:
                w = v * weight_from_seam(y[:, tr_ds.idx_seam:tr_ds.idx_seam + 1],
                                         w0=a.seam_weight, sigma=a.seam_sigma)
            if tr_ds.idx_instw is not None:
                iw = y[:, tr_ds.idx_instw:tr_ds.idx_instw + 1]
                w = iw * v if w is None else w * iw
            if tr_ds.idx_bgw is not None:
                bw = y[:, tr_ds.idx_bgw:tr_ds.idx_bgw + 1]
                w = bw * v if w is None else w * bw
            yp = y[:, :n_pred]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = dice_bce(out[:, :2], yp[:, :2], pw, v, w)
                if a.dist_channel:  # regression, not classification
                    d = torch.sigmoid(out[:, 2:3])
                    loss = loss + a.dist_gain * ((d - yp[:, 2:3]).abs() * v).sum() / v.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
            n += 1
        m = evaluate(model, va, dev, pw)
        score = m["fg_iou"] + m["ol_iou"]
        torch.save({"model": model.state_dict(), "encoder": a.encoder, "tile": a.tile, "epoch": ep, "metrics": m},
                   os.path.join(a.out, "last.pt"))
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "encoder": a.encoder, "tile": a.tile, "epoch": ep, "metrics": m},
                       os.path.join(a.out, "best.pt"))
        lr_now = opt.param_groups[0]["lr"]
        print(f"ep {ep:3d}/{a.epochs}  lr {lr_now:.2e}  loss {tot / max(n, 1):.4f}  "
              f"val_loss {m['val_loss']:.4f}  "
              f"fg IoU {m['fg_iou']:.4f}  outline IoU {m['ol_iou']:.4f}  "
              f"fg F1 {m['fg_f1']:.4f}  outline F1 {m['ol_f1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done. best fg+outline IoU {best:.4f} -> {a.out}/best.pt")


if __name__ == "__main__":
    main()
