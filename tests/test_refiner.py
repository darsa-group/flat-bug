"""Tests for `Predictor.refine_instances`, the zoomed-in second segmentation pass.

The refiner is tested against a stub model rather than a checkpoint, because what needs
guarding is not the mask quality - that is an empirical question settled by evaluation - but
the three contracts it makes with the rest of the pipeline:

  * it never changes how many instances there are,
  * a refinement that disagrees with the original is rejected,
  * a crop that yields nothing leaves the original untouched.

Each of those is a silent failure if it breaks: predictions would still come out, just wrong.
"""

import pytest
import torch

import flat_bug.predictor as predictor_module
from flat_bug.predictor import Predictor, TensorPredictions

TILE = 1024
OCC = 0.67
IMG_SIZE = 2000
# a 200px square at (600, 600) - comfortably inside the refinable size range
SQUARE = torch.tensor([[600.0, 600.0], [800.0, 600.0], [800.0, 800.0], [600.0, 800.0]])


class StubModel(torch.nn.Module):
    """Stands in for the YOLO model; `postprocess` is stubbed too, so the output is unused."""

    def forward(self, x):  # noqa: D102
        return x

    def to(self, *args, **kwargs):  # noqa: D102
        return self


@pytest.fixture
def predictor():  # noqa: D103
    p = Predictor(StubModel(), device="cpu", dtype=torch.float32)
    p.set_hyperparameters(
        REFINE=True, REFINE_MIN_PX=96, REFINE_OCCUPANCY=OCC, REFINE_MIN_AGREEMENT=0.5,
        TILE_SIZE=TILE, BATCH_SIZE=8, PREFER_POLYGONS=True,
    )
    return p


def make_preds(polygon=SQUARE):
    """A single-instance `TensorPredictions` in original-image coordinates."""
    image = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)
    tp = TensorPredictions(image=image, image_path="stub.jpg")
    tp.polygons = [polygon.clone()]
    # `offset_scale_pad` leaves boxes as integral xyxy, which is what the refiner reads
    tp.boxes = torch.tensor([[
        polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max(),
    ]]).long()
    tp.confs = torch.tensor([0.9])
    tp.classes = torch.tensor([0.0])
    tp.scales = [1.0]
    return tp


def crop_mask(cover, size=256):
    """A proto-resolution mask covering the central `cover` fraction of the crop.

    The refiner magnifies an instance to `REFINE_OCCUPANCY` of the tile, so `cover=OCC`
    reproduces the original instance and smaller values are progressively worse disagreements.
    """
    m = torch.zeros((1, size, size))
    lo, hi = int(size * (1 - cover) / 2), int(size * (1 + cover) / 2)
    m[0, lo:hi, lo:hi] = 1.0
    return m


def stub_postprocess(masks):
    """Replace `postprocess` with one that always returns `masks` for every crop in the batch."""
    def _f(preds, imgs, **kwargs):
        return [{"masks": masks} for _ in range(len(imgs))]
    return _f


class TestRefineInstances:  # noqa: D101
    def test_agreeing_refinement_is_accepted(self, predictor, monkeypatch):
        """A mask matching the original is taken, and the box is rebuilt from it."""
        monkeypatch.setattr(predictor_module, "postprocess", stub_postprocess(crop_mask(OCC)))
        preds = make_preds()
        before = preds.polygons[0].clone()
        out = predictor.refine_instances(preds)

        assert len(out) == 1
        assert not torch.allclose(out.polygons[0], before), "the refinement was not applied"
        # the stub reproduces the original square, so it must land back on it
        assert out.polygons[0].amin(dim=0).max() == pytest.approx(600, abs=8)
        assert out.polygons[0].amax(dim=0).min() == pytest.approx(800, abs=8)
        # the box is rebuilt from the new polygon, padded outwards like the pyramid's
        box = out.boxes[0, :4]
        assert box[0] <= out.polygons[0][:, 0].min() and box[2] >= out.polygons[0][:, 0].max()

    def test_divergent_refinement_is_rejected(self, predictor, monkeypatch):
        """A mask that overlaps the original too little is discarded, not applied.

        This is the neighbouring-animal case: the crop contains more than the target instance
        and the model segmented the wrong one.
        """
        # a quarter-size central mask overlaps the original at IoU ~0.09, far below 0.5
        monkeypatch.setattr(predictor_module, "postprocess", stub_postprocess(crop_mask(OCC / 4)))
        preds = make_preds()
        before = preds.polygons[0].clone()
        out = predictor.refine_instances(preds)

        assert len(out) == 1
        assert torch.equal(out.polygons[0], before), "a divergent refinement was accepted"

    def test_empty_crop_keeps_the_original(self, predictor, monkeypatch):
        """Finding nothing in the crop must never delete or alter the detection."""
        monkeypatch.setattr(
            predictor_module, "postprocess", stub_postprocess(torch.zeros((0, 256, 256)))
        )
        preds = make_preds()
        before = preds.polygons[0].clone()
        out = predictor.refine_instances(preds)

        assert len(out) == 1, "the refiner deleted a detection"
        assert torch.equal(out.polygons[0], before)

    def test_disabled_is_a_no_op(self, predictor, monkeypatch):
        """With `REFINE` off the model is never called."""
        def boom(*a, **k):
            raise AssertionError("the refiner ran with REFINE=False")

        monkeypatch.setattr(predictor_module, "postprocess", boom)
        predictor.set_hyperparameters(REFINE=False)
        preds = make_preds()
        before = preds.polygons[0].clone()
        assert torch.equal(predictor.refine_instances(preds).polygons[0], before)

    @pytest.mark.parametrize(
        "side, refined",
        [
            (40, False),    # below REFINE_MIN_PX: magnifying only interpolates
            (200, True),
            (900, False),   # above TILE * REFINE_OCCUPANCY: would have to be shrunk
        ],
    )
    def test_size_range(self, predictor, monkeypatch, side, refined):
        """Only instances inside the refinable size range are touched."""
        monkeypatch.setattr(predictor_module, "postprocess", stub_postprocess(crop_mask(OCC)))
        x0 = y0 = 500.0
        poly = torch.tensor([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]])
        preds = make_preds(poly)
        before = preds.polygons[0].clone()
        out = predictor.refine_instances(preds)
        assert (not torch.equal(out.polygons[0], before)) == refined

    def test_confidence_is_carried_over(self, predictor, monkeypatch):
        """The crop's own confidence must not replace the first pass's."""
        monkeypatch.setattr(predictor_module, "postprocess", stub_postprocess(crop_mask(OCC)))
        preds = make_preds()
        out = predictor.refine_instances(preds)
        assert float(out.confs[0]) == pytest.approx(0.9)

    def test_batches_cover_every_instance(self, predictor, monkeypatch):
        """More instances than `BATCH_SIZE` must all be refined, not just the first batch."""
        monkeypatch.setattr(predictor_module, "postprocess", stub_postprocess(crop_mask(OCC)))
        predictor.set_hyperparameters(BATCH_SIZE=2)
        n = 5
        polys = [SQUARE + torch.tensor([[float(i) * 250, 0.0]]) for i in range(n)]
        preds = make_preds()
        preds.polygons = polys
        preds.boxes = torch.stack([
            torch.tensor([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()]).long()
            for p in polys
        ])
        preds.confs = torch.full((n,), 0.9)
        preds.classes = torch.zeros(n)
        preds.scales = [1.0] * n
        before = [p.clone() for p in polys]
        out = predictor.refine_instances(preds)
        assert all(not torch.equal(a, b) for a, b in zip(out.polygons, before))


class TestPolygonsRequired:  # noqa: D101
    def test_mask_mode_is_skipped(self, predictor, monkeypatch):
        """Refinement is polygon-only; in mask mode it must decline rather than corrupt."""
        monkeypatch.setattr(predictor_module, "postprocess", stub_postprocess(crop_mask(OCC)))
        preds = make_preds()
        preds.PREFER_POLYGONS = False
        before = preds.polygons[0].clone()
        assert torch.equal(predictor.refine_instances(preds).polygons[0], before)
