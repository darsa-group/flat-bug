"""Validator that keeps bbox-only images out of the mask metrics but in the box metrics."""

from __future__ import annotations

from typing import Any

from ultralytics.models.yolo.segment import SegmentationValidator

from flat_bug.bbox_only_metrics import FlatBugSegmentMetrics


class FlatBugSegmentationValidator(SegmentationValidator):
    """As upstream, but every stat carries the image's `has_mask` flag.

    The flag rides through `_prepare_batch` -> `_process_batch` because ultralytics builds the
    stat dict from `**self._process_batch(...)` and never passes the batch index down to it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = FlatBugSegmentMetrics()

    def init_metrics(self, model) -> None:  # noqa: D102
        super().init_metrics(model)
        # init_metrics may reset names on the metrics object; keep our subclass in place.
        if not isinstance(self.metrics, FlatBugSegmentMetrics):
            names = getattr(self.metrics, "names", {})
            self.metrics = FlatBugSegmentMetrics(names)

    def update_metrics(self, preds, batch) -> None:  # noqa: D102
        self._has_mask_batch = batch.get("has_mask") if isinstance(batch, dict) else None
        return super().update_metrics(preds, batch)

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:  # noqa: D102
        pb = super()._prepare_batch(si, batch)
        hm = getattr(self, "_has_mask_batch", None)
        try:
            self._cur_has_mask = True if hm is None else bool(hm[si])
        except (IndexError, TypeError):
            self._cur_has_mask = True
        return pb

    def _process_batch(self, preds, batch) -> dict[str, Any]:  # noqa: D102
        tp = super()._process_batch(preds, batch)
        tp["has_mask"] = getattr(self, "_cur_has_mask", True)
        return tp
