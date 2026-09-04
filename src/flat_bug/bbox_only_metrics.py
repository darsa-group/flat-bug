"""Mask metrics computed over masked images only; box metrics over everything.

`SegmentMetrics.process` feeds one `target_cls` array to both the box and the mask AP, so
excluding an image from mask metrics is not simply a matter of dropping its `tp_m`: the
ground-truth count is the recall denominator, and leaving it in would count every instance of a
bbox-only image as a mask false negative. That would make mask mAP fall in proportion to how
much bbox-only data is in the validation split, which is meaningless.

This subclass keeps a parallel set of stat lists for the mask branch, appending to them only
for images whose `has_mask` flag is true. With every image masked, it is arithmetically the
stock SegmentMetrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics.utils.metrics import SegmentMetrics, ap_per_class
from ultralytics.utils.metrics import DetMetrics

_MASK_KEYS = ("tp_m", "conf", "pred_cls", "target_cls")


class FlatBugSegmentMetrics(SegmentMetrics):
    """SegmentMetrics that restricts the mask branch to images carrying real masks."""

    def __init__(self, names: dict[int, str] = {}) -> None:  # noqa: B006  (matches upstream signature)
        super().__init__(names=names)
        self._mask_stats: dict[str, list] = {k: [] for k in _MASK_KEYS}

    def update_stats(self, stat: dict[str, Any]) -> None:
        """Append box stats always; append mask stats only for masked images."""
        super().update_stats(stat)
        if stat.get("has_mask", True):
            for k in _MASK_KEYS:
                self._mask_stats[k].append(stat[k])

    def clear_stats(self) -> None:  # noqa: D102
        super().clear_stats()
        for v in self._mask_stats.values():
            v.clear()

    def process(self, save_dir: Path = Path("."), plot: bool = False, on_plot=None) -> dict[str, np.ndarray]:
        """Box AP over all images, mask AP over masked images only."""
        stats = DetMetrics.process(self, save_dir, plot, on_plot=on_plot)
        m = {k: (np.concatenate(v, 0) if len(v) else np.array([])) for k, v in self._mask_stats.items()}
        if len(m["tp_m"]) == 0 or len(m["target_cls"]) == 0:
            # Nothing carries masks: leave the mask metrics at their zero state rather than
            # inventing numbers from an empty set.
            self.seg.nc = len(self.names)
            return stats
        results_mask = ap_per_class(
            m["tp_m"], m["conf"], m["pred_cls"], m["target_cls"],
            plot=plot, on_plot=on_plot, save_dir=save_dir, names=self.names, prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        return stats
