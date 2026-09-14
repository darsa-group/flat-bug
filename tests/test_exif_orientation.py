"""Prepared images must carry no EXIF rotation, or training and inference disagree.

Annotators work on the EXIF-upright image: CVAT applies the tag, records the ROTATED
width/height, and the exported COCO - hence the YOLO labels - lives in that frame. The two
consumers then disagree. Training reads with cv2.imread, which ignores EXIF. Inference reads
with decode_image(apply_exif_orientation=True), which does not. On a tagged image the frames
differ by a transpose, so labels land in the wrong place during training with no error raised.

Found in PeMaToEuroPep: 13 of 44 validation images tagged orientation 6, PIL reporting
5184x3456 where CVAT recorded 3456x5184.
"""

import numpy as np
import pytest
from PIL import Image

from flat_bug.cli.fb_prepare_data import (
    _EXIF_ORIENTATION,
    assert_no_exif_rotation,
    copy_image_upright,
)

ROTATE_90 = 6


def _write(path, w, h, orientation=None):
    rng = np.random.default_rng(0)
    im = Image.fromarray((rng.random((h, w, 3)) * 255).astype("uint8"), "RGB")
    kw = {}
    if orientation is not None:
        ex = Image.Exif()
        ex[_EXIF_ORIENTATION] = orientation
        kw["exif"] = ex
    im.save(path, **kw)
    return im


def test_rotation_is_baked_in(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    _write(src, 40, 20, ROTATE_90)
    assert copy_image_upright(str(src), str(dst)) is True
    with Image.open(dst) as out:
        assert out.size == (20, 40), "dimensions must match the upright frame CVAT annotated"
        assert out.getexif().get(_EXIF_ORIENTATION, 1) in (1, None), "tag must be gone"


def test_untagged_images_are_copied_byte_for_byte(tmp_path):
    """Only rotated images may be re-encoded; the rest must take no generation loss."""
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    _write(src, 40, 20)
    assert copy_image_upright(str(src), str(dst)) is False
    assert src.read_bytes() == dst.read_bytes()


def test_orientation_1_is_not_re_encoded(tmp_path):
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    _write(src, 40, 20, 1)
    assert copy_image_upright(str(src), str(dst)) is False


def test_assertion_catches_a_rotated_image(tmp_path):
    _write(tmp_path / "fine.jpg", 40, 20)
    _write(tmp_path / "rotated.jpg", 40, 20, ROTATE_90)
    with pytest.raises(AssertionError, match="EXIF orientation"):
        assert_no_exif_rotation(str(tmp_path))


def test_assertion_passes_once_copied_upright(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    _write(tmp_path / "rotated.jpg", 40, 20, ROTATE_90)
    copy_image_upright(str(tmp_path / "rotated.jpg"), str(out / "rotated.jpg"))
    assert_no_exif_rotation(str(out))


def test_label_frame_survives_preparation(tmp_path):
    """The regression itself: a normalised label must land on the same pixels either way.

    A YOLO label is a fraction of width/height. Annotated at 20x40 (upright) but read back
    against a 40x20 file, a box at (0.25, 0.5) moves from x=5,y=20 to x=10,y=10 - a different
    insect, or empty background.
    """
    src, dst = tmp_path / "src.jpg", tmp_path / "dst.jpg"
    _write(src, 40, 20, ROTATE_90)
    fx, fy = 0.25, 0.5
    with Image.open(src) as im:
        raw_w, raw_h = im.size
    copy_image_upright(str(src), str(dst))
    with Image.open(dst) as im:
        up_w, up_h = im.size
    assert (fx * raw_w, fy * raw_h) != (fx * up_w, fy * up_h), "test must exercise a real transpose"
    assert (fx * up_w, fy * up_h) == (fx * 20, fy * 40), "prepared frame must be the annotated one"
