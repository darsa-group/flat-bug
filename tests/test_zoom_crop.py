"""Does ZoomCrop magnify, and does it respect the size floor?"""
import numpy as np, pytest
from ultralytics.utils.instance import Instances
from flat_bug.augmentations import ZoomCrop, MaybeZoomCrop, RandomCrop

def _labels(boxes, imsz=2000):
    """boxes: list of (cx, cy, w, h) in pixels -> a labels dict ZoomCrop accepts."""
    img = np.random.randint(0, 255, (imsz, imsz, 3), np.uint8)
    segs, bb = [], []
    for cx, cy, w, h in boxes:
        segs.append(np.array([[cx-w/2,cy-h/2],[cx+w/2,cy-h/2],[cx+w/2,cy+h/2],[cx-w/2,cy+h/2]], np.float32))
        bb.append([cx, cy, w, h])
    inst = Instances(np.asarray(bb, np.float32), np.stack(segs), bbox_format="xywh", normalized=False)
    return {"img": img, "instances": inst, "cls": np.zeros((len(boxes),1), np.float32)}

def test_zoom_magnifies_a_large_instance():
    z = ZoomCrop(imsize=1536, min_px=100, drop_below=32)
    out = z(_labels([(1000, 1000, 300, 300)]))
    assert out is not None
    inst = out["instances"]
    if inst.normalized: inst.denormalize(*out["img"].shape[:2][::-1])
    if inst._bboxes.format != "xywh": inst.convert_bbox(format="xywh")
    size = float(np.maximum(inst._bboxes.bboxes[:,2], inst._bboxes.bboxes[:,3]).max())
    assert size > 300, f"instance should be magnified beyond its 300px original, got {size:.0f}"
    assert out["img"].shape[0] == 1536

def test_returns_none_when_nothing_is_large_enough():
    z = ZoomCrop(imsize=1536, min_px=100, drop_below=32)
    assert z(_labels([(1000, 1000, 40, 40)])) is None   # 40px < min_px 100

def test_maybe_falls_back_when_zoom_declines():
    m = MaybeZoomCrop(RandomCrop(imsize=1536), ZoomCrop(imsize=1536, min_px=100, drop_below=32), p=1.0)
    out = m(_labels([(1000, 1000, 40, 40)]))          # zoom returns None -> RandomCrop runs
    assert out is not None and out["img"].shape[0] == 1536

def test_p_zero_never_zooms():
    m = MaybeZoomCrop(RandomCrop(imsize=1536), ZoomCrop(imsize=1536, min_px=100, drop_below=32), p=0.0)
    sizes = []
    for _ in range(12):
        o = m(_labels([(1000, 1000, 300, 300)]))
        inst = o["instances"]
        if inst.normalized: inst.denormalize(*o["img"].shape[:2][::-1])
        if inst._bboxes.format != "xywh": inst.convert_bbox(format="xywh")
        if len(inst._bboxes.bboxes):
            sizes.append(float(np.maximum(inst._bboxes.bboxes[:,2], inst._bboxes.bboxes[:,3]).max()))
    assert sizes and max(sizes) < 900, f"p=0 should never magnify 4x+, saw {max(sizes):.0f}"


def test_min_scale_below_one_lets_a_large_instance_be_scaled_down():
    """Without this, an instance already bigger than xsize*occupancy is left untouched."""
    big = _labels([(1000, 1000, 1200, 1200)], imsz=3000)
    z_hi = ZoomCrop(imsize=1536, min_px=100, drop_below=32, occupancy=(0.45, 0.45), min_scale=1.0)
    z_lo = ZoomCrop(imsize=1536, min_px=100, drop_below=32, occupancy=(0.45, 0.45), min_scale=0.15)
    def longest(o):
        i = o["instances"]
        if i.normalized: i.denormalize(o["img"].shape[1], o["img"].shape[0])
        if i._bboxes.format != "xywh": i.convert_bbox(format="xywh")
        b = i._bboxes.bboxes
        return float(np.maximum(b[:, 2], b[:, 3]).max()) if len(b) else 0.0
    kept = longest(z_hi(_labels([(1000, 1000, 1200, 1200)], imsz=3000)))
    shrunk = longest(z_lo(_labels([(1000, 1000, 1200, 1200)], imsz=3000)))
    assert abs(kept - 1200) < 60, f"min_scale=1.0 should leave it at 1200px, got {kept:.0f}"
    assert abs(shrunk - 1536*0.45) < 80, f"min_scale=0.15 should bring it to ~691px, got {shrunk:.0f}"


def test_occupancy_sets_the_output_size_regardless_of_original_size():
    z = ZoomCrop(imsize=1536, min_px=100, drop_below=32, occupancy=(0.45, 0.45), min_scale=0.15)
    out = []
    for orig in (120, 340, 700, 1500):
        o = z(_labels([(1500, 1500, orig, orig)], imsz=4000))
        i = o["instances"]
        if i.normalized: i.denormalize(o["img"].shape[1], o["img"].shape[0])
        if i._bboxes.format != "xywh": i.convert_bbox(format="xywh")
        b = i._bboxes.bboxes
        out.append(float(np.maximum(b[:, 2], b[:, 3]).max()))
    assert max(out) - min(out) < 120, f"all should land near 1536*0.45=691px, got {[round(v) for v in out]}"
