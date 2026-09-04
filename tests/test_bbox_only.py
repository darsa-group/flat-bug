"""Bbox-only datasets: polygons become boxes, and those images leave the mask loss alone.

The critical property is the first test: with every image masked, the patched loss must equal
the stock loss exactly. If it does not, enabling this feature silently changes the objective
for data that was never meant to be affected.
"""

import numpy as np
import pytest
import torch

from flat_bug.bbox_only import compile_bbox_only, downgrade_labels, is_bbox_only
from flat_bug.bbox_only_loss import bbox_only_segmentation_loss

ultralytics_loss = pytest.importorskip("ultralytics.utils.loss")


@pytest.fixture(autouse=True)
def _vectorised_crop_mask(monkeypatch):
    """crop_mask branches on `n < 50 and not is_cuda` with non-equivalent branches; pin one."""
    def crop_mask(masks, boxes):
        _, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    monkeypatch.setattr(ultralytics_loss, "crop_mask", crop_mask)


class _Criterion:
    def __init__(self, overlap=True):
        self.overlap = overlap
        self.single_mask_loss = ultralytics_loss.v8SegmentationLoss.single_mask_loss


def _inputs(bs=3, n_anchors=200, n_gt=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    nm, mh, mw = 32, 32, 32
    fg = torch.rand(bs, n_anchors, generator=g) > 0.4
    for i in range(bs):
        fg[i, 0] = True
    tgi = torch.randint(0, n_gt, (bs, n_anchors), generator=g)
    xy1 = torch.rand(bs, n_anchors, 2, generator=g) * 60
    tb = torch.cat([xy1, xy1 + 20 + torch.rand(bs, n_anchors, 2, generator=g) * 40], 2)
    proto = torch.randn(bs, nm, mh, mw, generator=g)
    pm = torch.randn(bs, n_anchors, nm, generator=g)
    masks = torch.randint(0, n_gt + 1, (bs, mh, mw), generator=g).float()
    bi = torch.zeros(bs * n_gt, 1)
    return fg, masks, tgi, tb, bi, proto, pm, torch.tensor([128.0, 128.0])


def test_all_masked_is_identical_to_stock_loss():
    c, args = _Criterion(), _inputs()
    ref = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(c, *args)
    import flat_bug.bbox_only_loss as M
    with bbox_only_segmentation_loss():
        M._state.has_mask = [True, True, True]
        got = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(c, *args)
        M._state.has_mask = None
    assert torch.allclose(ref, got, rtol=1e-6, atol=1e-7), f"{ref} != {got}"


def test_absent_flag_is_identical_to_stock_loss():
    """No has_mask in the batch must behave exactly as before the feature existed."""
    c, args = _Criterion(), _inputs(seed=3)
    ref = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(c, *args)
    with bbox_only_segmentation_loss():
        got = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(c, *args)
    assert torch.allclose(ref, got, rtol=1e-6, atol=1e-7)


def test_masking_an_image_changes_the_loss():
    c, args = _Criterion(), _inputs(seed=1)
    import flat_bug.bbox_only_loss as M
    with bbox_only_segmentation_loss():
        M._state.has_mask = [True, True, True]
        full = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(c, *args)
        M._state.has_mask = [True, False, True]
        part = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(c, *args)
        M._state.has_mask = None
    assert not torch.allclose(full, part), "excluding an image must change the loss"


def test_excluded_image_contributes_nothing_and_is_not_counted():
    """Excluding image 1 must equal computing the loss over images 0 and 2 alone."""
    import flat_bug.bbox_only_loss as M
    c = _Criterion()
    fg, masks, tgi, tb, bi, proto, pm, imgsz = _inputs(seed=2)
    keep = [0, 2]
    sub = (fg[keep], masks[keep], tgi[keep], tb[keep], bi, proto[keep], pm[keep], imgsz)
    ref = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(c, *sub)
    with bbox_only_segmentation_loss():
        M._state.has_mask = [True, False, True]
        got = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(
            c, fg, masks, tgi, tb, bi, proto, pm, imgsz)
        M._state.has_mask = None
    assert torch.allclose(ref, got, rtol=1e-5, atol=1e-6), f"{ref} != {got}"


# ---- label downgrade ----

def test_polygons_become_their_bounding_rectangle():
    tri = np.array([[10.0, 10.0], [30.0, 12.0], [20.0, 40.0]], dtype=np.float32)
    labels = [{"segments": [tri]}, {"segments": [tri.copy()]}]
    files = ["bugbox-bulk_a.jpg", "ArTaxOr_b.jpg"]
    pat = compile_bbox_only(["bugbox-bulk"])
    downgrade_labels(labels, files, pat)
    r = labels[0]["segments"][0]
    assert labels[0]["has_mask"] is False and labels[1]["has_mask"] is True
    assert r.shape == (4, 2)
    assert r[:, 0].min() == 10.0 and r[:, 0].max() == 30.0
    assert r[:, 1].min() == 10.0 and r[:, 1].max() == 40.0
    assert np.allclose(labels[1]["segments"][0], tri), "untouched dataset must keep its polygon"


def test_no_config_means_no_change():
    tri = np.array([[1.0, 1.0], [5.0, 1.0], [3.0, 6.0]], dtype=np.float32)
    labels = [{"segments": [tri.copy()]}]
    downgrade_labels(labels, ["bugbox-bulk_a.jpg"], compile_bbox_only(None))
    assert labels[0]["has_mask"] is True
    assert np.allclose(labels[0]["segments"][0], tri)


@pytest.mark.parametrize("f,expect", [
    ("bugbox-bulk_4053.jpg", True), ("bugbox_2020.jpg", False),
    ("/a/b/bugbox-bulk_x.jpg", True), ("ArTaxOr_1.jpg", False),
])
def test_prefix_matching_is_anchored(f, expect):
    """`bugbox` must not match `bugbox-bulk` or vice versa - they are different datasets."""
    assert is_bbox_only(f, compile_bbox_only(["bugbox-bulk"])) is expect


def test_empty_segments_still_get_flagged():
    labels = [{"segments": []}]
    downgrade_labels(labels, ["bugbox-bulk_a.jpg"], compile_bbox_only(["bugbox-bulk"]))
    assert labels[0]["has_mask"] is False


# ---- metrics: mask AP must ignore bbox-only images, box AP must not ----

def test_mask_metrics_exclude_bbox_only_images():
    """An unmasked image must not count its instances as mask false negatives."""
    from flat_bug.bbox_only_metrics import FlatBugSegmentMetrics
    from ultralytics.utils.metrics import SegmentMetrics

    def stat(seed, has_mask, n_pred=4, n_gt=3):
        g = np.random.default_rng(seed)
        return {
            "tp": g.random((n_pred, 10)) > 0.5,
            "tp_m": g.random((n_pred, 10)) > 0.5,
            "conf": g.random(n_pred),
            "pred_cls": np.zeros(n_pred),
            "target_cls": np.zeros(n_gt),
            "target_img": np.zeros(1),
            "im_name": f"im{seed}.jpg",
            "has_mask": has_mask,
        }

    names = {0: "insect"}
    # All masked: our subclass must agree with upstream.
    a, b = FlatBugSegmentMetrics(names), SegmentMetrics(names)
    for s in (stat(1, True), stat(2, True)):
        a.update_stats(dict(s)); b.update_stats(dict(s))
    a.process(); b.process()
    assert np.allclose(a.seg.mean_results(), b.seg.mean_results()), "must match upstream when all masked"
    assert np.allclose(a.box.mean_results(), b.box.mean_results())

    # One unmasked: mask metrics must equal computing over the masked image alone,
    # while box metrics still see both.
    c = FlatBugSegmentMetrics(names)
    c.update_stats(stat(1, True)); c.update_stats(stat(2, False))
    c.process()
    d = FlatBugSegmentMetrics(names)
    d.update_stats(stat(1, True))
    d.process()
    assert np.allclose(c.seg.mean_results(), d.seg.mean_results()), "mask AP must ignore the unmasked image"
    e = FlatBugSegmentMetrics(names)
    e.update_stats(stat(1, True)); e.update_stats(stat(2, True))
    e.process()
    assert np.allclose(c.box.mean_results(), e.box.mean_results()), "box AP must still use both images"
