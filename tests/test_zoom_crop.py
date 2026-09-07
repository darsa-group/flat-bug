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
