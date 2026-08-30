#!/usr/bin/env python3
"""PROTOTYPE - plot learning curves from a semseg training log.

Reads the ``ep N/M lr ... loss ... fg IoU ...`` lines that ``train.py`` prints and writes a
three-panel figure. Rerunnable at any time; it simply plots whatever epochs exist so far.

Usage:
    plot_curves.py LOGFILE [-o OUT.png]
"""

from __future__ import annotations

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LINE = re.compile(
    r"^ep\s+(\d+)/(\d+)\s+lr\s+([\d.e+-]+)\s+loss\s+([\d.]+)\s+"
    r"fg IoU\s+([\d.]+)\s+outline IoU\s+([\d.]+)\s+fg F1\s+([\d.]+)\s+outline F1\s+([\d.]+)"
)


def parse(path: str) -> dict[str, list[float]]:
    """Extract per-epoch metrics from a training log.

    Args:
        path: Path to the log file.

    Returns:
        Dict of column name to list of values, in epoch order.
    """
    keys = ["ep", "total", "lr", "loss", "fg_iou", "ol_iou", "fg_f1", "ol_f1"]
    out: dict[str, list[float]] = {k: [] for k in keys}
    for line in open(path):
        m = LINE.match(line)
        if not m:
            continue
        for k, v in zip(keys, m.groups()):
            out[k].append(float(v))
    return out


def main() -> None:
    """Parse the log and write the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("-o", "--out", default="semseg_curves.png")
    a = ap.parse_args()
    d = parse(a.log)
    n = len(d["ep"])
    if n == 0:
        print("no epoch lines found yet")
        return
    total = int(d["total"][0])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax[0].plot(d["ep"], d["loss"], "o-", color="#1f77b4", ms=3)
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("train loss (Dice + weighted BCE)")
    ax[0].set_title("Training loss")

    ax[1].plot(d["ep"], d["fg_iou"], "o-", color="#2ca02c", ms=3, label="foreground IoU")
    ax[1].plot(d["ep"], d["fg_f1"], "o--", color="#2ca02c", ms=3, alpha=.5, label="foreground F1")
    ax[1].plot(d["ep"], d["ol_iou"], "s-", color="#d62728", ms=3, label="outline IoU")
    ax[1].plot(d["ep"], d["ol_f1"], "s--", color="#d62728", ms=3, alpha=.5, label="outline F1")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("validation metric")
    ax[1].set_title("Validation — outline IoU is the one that matters")
    ax[1].legend(fontsize=8, loc="upper left")

    ax[2].plot(d["ep"], d["lr"], "o-", color="#9467bd", ms=3)
    ax[2].set_xlabel("epoch"), ax[2].set_ylabel("learning rate")
    ax[2].set_title("OneCycleLR (peaks at 30% of run)")

    for a_ in ax:
        a_.grid(alpha=.3)
        a_.set_xlim(0, max(total, max(d["ep"]) + 1))
    fig.suptitle(f"semseg U-Net — epoch {int(d['ep'][-1])} of {total}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(a.out, dpi=130)
    print(f"saved {a.out}  ({n} epochs plotted)")
    print(f"latest: fg IoU {d['fg_iou'][-1]:.4f}  outline IoU {d['ol_iou'][-1]:.4f}  lr {d['lr'][-1]:.2e}")


if __name__ == "__main__":
    main()
