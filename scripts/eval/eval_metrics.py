#!/usr/bin/env python3
"""Summarise `fb_evaluate` output without R.

A port of the headline statistics from ``eval-metrics.R``, which needs six R
packages that are absent from both our workstation and GHPC. The metric
definitions are taken verbatim from that script, which remains canonical:

    in_gt     = idx_1 != -1          # the row corresponds to a ground-truth object
    in_im     = idx_2 != -1          # ... and/or to a prediction
    precision = sum(in_gt & in_im) / sum(in_im)
    recall    = sum(in_gt & in_im) / sum(in_gt)
    dataset   = leading "[^_]+" of the per-image CSV filename
    area      = contourArea_1 if > 0 else contourArea_2

Adds a size-stratified recall breakdown, which is what tells you whether misses
are concentrated in objects too large for a single tile.

Usage:
    eval_metrics.py <eval-dir> [--output results.csv] [--bootstrap 200]
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

SIZE_BINS = [0, 32, 64, 128, 256, 512, np.inf]


def load(eval_dir: str) -> pd.DataFrame:
    """Read the per-image CSVs, skipping the combined one, as the R script does."""
    files = [f for f in sorted(glob.glob(os.path.join(eval_dir, "*.csv"))) if "combined_results" not in f]
    if not files:
        raise FileNotFoundError(f"No per-image result CSVs in {eval_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, sep=";", usecols=lambda c: not c.startswith("contour_"))
        frame["filename"] = os.path.basename(path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["in_gt"] = data.idx_1 != -1
    data["in_im"] = data.idx_2 != -1
    data["dataset"] = data.filename.map(lambda f: (re.match(r"^[^_]+", f) or [""])[0])
    data["area"] = np.where(data.contourArea_1 > 0, data.contourArea_1, data.contourArea_2)
    return data


def bootstrap_ci(values: np.ndarray, n: int = 200, conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI of a mean, mirroring the R script's `binomial_mean_qci`."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    tail = (1 - conf) / 2
    return tuple(np.quantile(means, [tail, 1 - tail]))


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Per-sub-dataset precision/recall, as `eval-metrics.R` writes to results.csv.

    Plain aggregation rather than `groupby.apply`: the `include_groups` argument
    that quiets its deprecation warning is pandas>=2.2 only, and GHPC runs 3.x.
    """
    grouped = data.assign(matched=data.in_gt & data.in_im).groupby("dataset")
    agg = grouped.agg(
        tp=("matched", "sum"),
        n_gt=("in_gt", "sum"),
        n_im=("in_im", "sum"),
        n_instances=("matched", "size"),
    ).reset_index()
    agg["precision"] = agg.tp / agg.n_im.clip(lower=1)
    agg["recall"] = agg.tp / agg.n_gt.clip(lower=1)
    return agg[["dataset", "precision", "recall", "n_instances"]]


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", help="Directory of per-image CSVs written by fb_evaluate")
    parser.add_argument("-o", "--output", default=None, help="Where to write results.csv")
    parser.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples for the CIs")
    args = parser.parse_args()

    data = load(args.eval_dir)
    matched = data.in_gt & data.in_im
    tp, fn, fp = matched.sum(), (data.in_gt & ~data.in_im).sum(), (~data.in_gt & data.in_im).sum()
    lo, hi = bootstrap_ci(matched[data.in_gt].to_numpy(), args.bootstrap)

    print(f"images: {data.filename.nunique()}   sub-datasets: {data.dataset.nunique()}")
    print(f"TP {tp}  FN {fn}  FP {fp}")
    print(f"precision {tp / max(tp + fp, 1):.3f}   recall {tp / max(tp + fn, 1):.3f} "
          f"[{lo:.3f}, {hi:.3f}]   mean IoU {data.loc[matched, 'IoU'].mean():.3f}")

    print("\nrecall by ground-truth size (sqrt area, px):")
    gt = data[data.in_gt].copy()
    gt["side"] = np.sqrt(gt.area.clip(lower=0))
    labels = ["<32", "32-64", "64-128", "128-256", "256-512", ">512"]
    gt["bucket"] = pd.cut(gt.side, bins=SIZE_BINS, labels=labels, right=False)
    for bucket, group in gt.groupby("bucket", observed=True):
        lo_b, hi_b = bootstrap_ci(group.in_im.to_numpy(), args.bootstrap)
        print(f"  {str(bucket):<10} n={len(group):>5}  recall {group.in_im.mean():.3f} [{lo_b:.3f}, {hi_b:.3f}]")

    results = summarize(data)
    print(f"\nper sub-dataset ({len(results)} datasets):")
    print(results.sort_values("recall").to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = args.output or os.path.join(args.eval_dir, "results.csv")
    results.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
