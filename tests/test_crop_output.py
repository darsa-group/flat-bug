"""Crops are written as high-quality WebP and carry the source's physical resolution.

Two things were previously lost on every crop:

  * format. `save_crops` took the container from the SOURCE image's extension, so a run over
    JPEGs wrote JPEG crops and a run over WebP wrote WebP at Pillow's default quality 80 -
    lossy, and nobody chose it. Masked crops were forced to PNG because JPEG has no alpha.
  * physical resolution. The crop is rebuilt from a tensor with `Image.fromarray`, whose
    `.info` is empty, and `.save()` was called with neither `dpi=` nor `exif=`. Anything
    downstream reasoning about real-world size found nothing and silently used its own default.

The WebP encoder accepts `dpi=` and discards it, which is why DPI is written as EXIF for every
container rather than relying on `dpi=` alone.
"""

import numpy as np
import pytest
from PIL import Image

from flat_bug.predictor import (
    WEBP_MAX_DIM,
    TensorPredictions,
    crop_save_kwargs,
    source_dpi,
)

X_RES, Y_RES, RES_UNIT = 0x011A, 0x011B, 0x0128


@pytest.fixture
def rgb():
    rng = np.random.default_rng(0)
    return (rng.random((40, 30, 3)) * 255).astype("uint8")


def _read_dpi(path):
    """DPI as any reasonable consumer would look for it: native field first, then EXIF."""
    with Image.open(path) as im:
        if im.info.get("dpi"):
            return tuple(round(float(v)) for v in im.info["dpi"])
        ex = im.getexif()
        if ex.get(X_RES) is None:
            return None
        assert ex.get(RES_UNIT) == 2, "resolution unit must say inches"
        return (round(float(ex[X_RES])), round(float(ex[Y_RES])))


# --------------------------------------------------------------------- source_dpi


@pytest.mark.parametrize("fmt,ext", [("PNG", ".png"), ("JPEG", ".jpg"), ("TIFF", ".tif")])
def test_source_dpi_is_read_back(tmp_path, rgb, fmt, ext):
    p = tmp_path / f"src{ext}"
    Image.fromarray(rgb, "RGB").save(p, format=fmt, dpi=(1200, 1200))
    # PNG records pixels-per-metre as an integer, so 1200 dpi returns as 1199.9976.
    assert source_dpi(str(p)) == pytest.approx((1200.0, 1200.0), rel=1e-4)


def test_source_dpi_none_when_absent(tmp_path, rgb):
    p = tmp_path / "plain.png"
    Image.fromarray(rgb, "RGB").save(p)
    assert source_dpi(str(p)) is None


def test_source_dpi_ignores_the_tiff_unit_placeholder(tmp_path, rgb):
    """TIFF writes dpi=(1, 1) when it has nothing to say; that is not a resolution."""
    p = tmp_path / "placeholder.tif"
    Image.fromarray(rgb, "RGB").save(p, format="TIFF")
    assert source_dpi(str(p)) is None


def test_source_dpi_converts_centimetres(tmp_path, rgb):
    from PIL.TiffImagePlugin import IFDRational

    p = tmp_path / "cm.png"
    ex = Image.Exif()
    ex[X_RES], ex[Y_RES], ex[RES_UNIT] = IFDRational(100), IFDRational(100), 3  # cm
    Image.fromarray(rgb, "RGB").save(p, exif=ex)
    got = source_dpi(str(p))
    assert got is not None and abs(got[0] - 254.0) < 0.01


def test_source_dpi_handles_missing_and_unreadable(tmp_path):
    assert source_dpi(None) is None
    assert source_dpi(str(tmp_path / "nope.png")) is None
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert source_dpi(str(bad)) is None


# --------------------------------------------------------------------- encoder kwargs


def test_webp_gets_lossless_not_compress_level():
    """compress_level is a PNG argument; the WebP encoder ignores it, so quality was Pillow's 80.

    In lossless mode `quality` is encoder effort rather than fidelity, so it is still passed -
    0 writes an exact crop in 5.0 ms against 60.2 ms at the default, for 12% more bytes.
    """
    kw = crop_save_kwargs("WEBP", has_alpha=False, lossless=True, quality=0, dpi=None)
    assert kw == {"lossless": True, "quality": 0}
    assert "compress_level" not in kw


def test_lossless_effort_setting_still_round_trips_exactly(tmp_path, rgb):
    """The speed knob must not quietly become a fidelity knob."""
    import torch

    crop = torch.from_numpy(rgb).permute(2, 0, 1)
    for q in (0, 50, 100):
        out = TensorPredictions._save_1_crop(
            crop, None, str(tmp_path / f"q{q}.webp"), None, True, q
        )
        with Image.open(out) as im:
            assert np.array_equal(np.asarray(im.convert("RGB")), rgb), f"effort {q} was not exact"


def test_webp_lossy_uses_quality():
    kw = crop_save_kwargs("WEBP", has_alpha=False, lossless=False, quality=95, dpi=None)
    assert kw == {"quality": 95}


def test_dpi_goes_in_as_exif_for_webp_and_not_as_dpi():
    kw = crop_save_kwargs("WEBP", has_alpha=False, lossless=True, quality=95, dpi=(600, 600))
    assert "exif" in kw
    assert "dpi" not in kw, "WebP accepts dpi= and silently drops it"


def test_other_formats_get_both_dpi_and_exif():
    kw = crop_save_kwargs("PNG", has_alpha=False, lossless=True, quality=95, dpi=(600, 600))
    assert kw["dpi"] == (600, 600) and "exif" in kw


def test_jpeg_refuses_alpha():
    with pytest.raises(ValueError, match="alpha"):
        crop_save_kwargs("JPEG", has_alpha=True, lossless=False, quality=95, dpi=None)


# --------------------------------------------------------------------- writing crops


def _save(tmp_path, arr, name, mask=None, **kw):
    import torch

    crop = torch.from_numpy(arr).permute(2, 0, 1)
    # chw2hwc_uint8 concatenates the mask onto the crop, so it needs a channel axis.
    m = None if mask is None else torch.from_numpy(mask)[None]
    return TensorPredictions._save_1_crop(crop, m, str(tmp_path / name), **kw)


def test_webp_crop_is_pixel_exact_and_carries_dpi(tmp_path, rgb):
    out = _save(tmp_path, rgb, "c.webp", dpi=(1200, 1200), lossless=True)
    with Image.open(out) as im:
        assert im.format == "WEBP"
        assert np.array_equal(np.asarray(im.convert("RGB")), rgb), "lossless must round-trip exactly"
    assert _read_dpi(out) == (1200, 1200)


@pytest.mark.parametrize("name", ["c.webp", "c.png", "c.jpg"])
def test_every_format_carries_dpi(tmp_path, rgb, name):
    out = _save(tmp_path, rgb, name, dpi=(600, 600), lossless=True, quality=95)
    assert _read_dpi(out) == (600, 600), f"{name} lost its DPI"


def test_masked_webp_keeps_alpha_and_dpi(tmp_path, rgb):
    mask = (np.random.default_rng(1).random((40, 30)) > 0.5)
    out = _save(tmp_path, rgb, "m.webp", mask=mask, dpi=(300, 300), lossless=True)
    with Image.open(out) as im:
        assert im.mode == "RGBA", "masked crops need the alpha channel PNG used to be required for"
    assert _read_dpi(out) == (300, 300)


def test_crop_without_dpi_still_writes(tmp_path, rgb):
    out = _save(tmp_path, rgb, "c.webp", dpi=None, lossless=True)
    assert _read_dpi(out) is None
    with Image.open(out) as im:
        assert im.size == (30, 40)


def test_oversized_crop_falls_back_to_png(tmp_path):
    """WebP refuses >16383 px on an axis; the crop must survive, not be lost."""
    tall = np.zeros((WEBP_MAX_DIM + 2, 4, 3), dtype="uint8")
    out = _save(tmp_path, tall, "big.webp", dpi=(600, 600), lossless=True)
    assert out.endswith(".png")
    with Image.open(out) as im:
        assert im.format == "PNG" and im.size == (4, WEBP_MAX_DIM + 2)
    assert _read_dpi(out) == (600, 600), "the fallback must not drop DPI either"
