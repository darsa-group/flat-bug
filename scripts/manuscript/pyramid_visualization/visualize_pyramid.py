import math
import os
from typing import List, Tuple

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from flat_bug import download_from_repository
from flat_bug.geometric import equal_allocate_overlaps
from flat_bug.predictor import Predictor


def get_scales(model : Predictor, image : str, scale_increment : float=1/2, scale_before : float=1) -> List[float]:
    if isinstance(image, str):
        path : str = image
        image : torch.Tensor = read_image(
            path=image, 
            mode=ImageReadMode.RGB, 
            apply_exif_orientation=True
        ).to(model._device)
    elif isinstance(image, torch.Tensor):
        logger.info("Input image source file not specified for prediction, saving the prediction will require specifying the source file basename.")
    else:
        raise TypeError(f"Unknown type for image: {type(image)}, expected str or torch.Tensor")

    c, h, w = image.shape
    transform_list = []
    # Check if the image has an integer data type
    if image.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
        transform_list.append(transforms.ConvertImageDtype(model._dtype))

    if scale_before != 1:
        w, h = int(w * scale_before), int(h * scale_before)
        resize = transforms.Resize((h, w), antialias=True)
        transform_list.append(resize)

    # A border is always added now, to avoid edge-cases on the actual edge of the image. I.e. only detections on internal edges of tiles should be removed, not detections on the edge of the image.
    edge_case_margin_padding_multiplier = 2
    padding_offset = torch.tensor((model.EDGE_CASE_MARGIN, model.EDGE_CASE_MARGIN), dtype=model._dtype, device=model._device) * edge_case_margin_padding_multiplier
    if padding_offset.sum() > 0:
        padding_for_edge_cases = transforms.Pad(
            padding=model.EDGE_CASE_MARGIN * edge_case_margin_padding_multiplier, 
            fill=0,
            padding_mode='constant'
        )
        # padding_for_edge_cases = InpaintPad(padding=model.EDGE_CASE_MARGIN * edge_case_margin_padding_multiplier)
        transform_list.append(padding_for_edge_cases)
    else:
        padding_offset[:] = 0
    if transform_list:
        transforms_composed = transforms.Compose(transform_list)

    transformed_image = transforms_composed(image) if transform_list else image

    # Check correct dimensions
    assert len(transformed_image.shape) == 3, RuntimeError(f"transformed_image.shape {transformed_image.shape} != 3") 
    # Check correct number of channels
    assert transformed_image.shape[0] == 3, RuntimeError(f"transformed_image.shape[0] {transformed_image.shape[0]} != 3. The image is probably supplied in WxHxC instead of CxWxH, try image.permute(2, 1, 0) before passing it.")

    max_dim = max(transformed_image.shape[1:])
    min_dim = min(transformed_image.shape[1:])

    # fixme, what to do if the image is too small? - RE: Fixed by adding padding in _detect_instances
    scales = []

    s = model.TILE_SIZE / max_dim

    if s >= 1:
        scales.append(s)
    else:
        while s <= 0.9:  # Cut off at 90%, to avoid having s~1 and s=1.
            scales.append(s)
            s /= scale_increment
        if s != 1:
            scales.append(1.0)
    
    return scales, transformed_image

def get_tile_params(model : Predictor, image : torch.Tensor, scale : float) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], torch.Tensor]:
    orig_h, orig_w = image.shape[1:]
    w, h = orig_w, orig_h
    h_pad, w_pad = 0, 0
    pad_lrtb = 0, 0, 0, 0

    # Check dimensions and channels
    assert image.device == model._device, RuntimeError(f"image.device {image.device} != model._device {model._device}")
    assert image.dtype == model._dtype, RuntimeError(f"image.dtype {image.dtype} != model._dtype {model._dtype}")

    # Resize if scale is not 1
    if scale != 1:
        h, w = round(orig_h * scale / 4) * 4, round(orig_w * scale / 4) * 4
        resize = transforms.Resize((h, w), antialias=True) 
        image = resize(image)
        h, w = image.shape[1:]
    
    # If any of the sides are smaller than the TILE_SIZE, pad to TILE_SIZE
    if w < model.TILE_SIZE or h < model.TILE_SIZE:
        w_pad = max(0, model.TILE_SIZE - w) // 2
        h_pad = max(0, model.TILE_SIZE - h) // 2
        pad_lrtb = w_pad, w_pad + (w % 2 == 1), h_pad, h_pad + (h % 2 == 1)
        image = torch.nn.functional.pad(image, pad_lrtb, mode="constant", value=0) # Pad with black
        h, w = image.shape[1:]

    # Tile calculation
    x_n_tiles = math.ceil(w / (model.TILE_SIZE - model.MINIMUM_TILE_OVERLAP)) if w != model.TILE_SIZE else 1
    y_n_tiles = math.ceil(h / (model.TILE_SIZE - model.MINIMUM_TILE_OVERLAP)) if h != model.TILE_SIZE else 1

    x_range = equal_allocate_overlaps(w, x_n_tiles, model.TILE_SIZE)
    y_range = equal_allocate_overlaps(h, y_n_tiles, model.TILE_SIZE)

    assert max(x_range) + model.TILE_SIZE == w, f'{w} : {x_range}'
    assert max(y_range) + model.TILE_SIZE == h, f'{h} : {y_range}'

    offsets = [((m, n), (j, i)) for n, j in enumerate(y_range) for m, i in enumerate(x_range)]

    return offsets, image

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    image = "test_image.jpg"
    if not (os.path.exists(image) and os.path.isfile(image)):
        download_from_repository(image)

    dtype, device = "float16", "cuda:0"
    model = Predictor(dtype=dtype, device=device)

    TILE_SIZE = model.TILE_SIZE

    scales, prep_image = get_scales(model, image)

    pyramid_params = {
        s : get_tile_params(model, prep_image, scale=s) 
        for s in reversed(scales)
    }

    for scale in pyramid_params:
        offsets, scaled_image = pyramid_params[scale]
        _, dy, dx = scaled_image.shape

        xs = set()
        ys = set()

        fig, axs = plt.subplots(1, 2, figsize=(17,10), dpi=300)
        ax : Axes = axs[0]
        ax.imshow(scaled_image.cpu().permute(1, 2, 0).float())
        ax.set_xlim(-5, dx + 5)
        ax.set_ylim(-5, dy + 5)
        for tile in offsets:
            _, (y, x) = tile
            xs.add(x)
            ys.add(y)
            ax.add_artist(Rectangle((x, y), TILE_SIZE, TILE_SIZE, edgecolor="blue", fill=None))
            ax.add_artist(Rectangle((x, y), TILE_SIZE, TILE_SIZE, alpha = .25, edgecolor=None))
        tile_bbox = Rectangle((0, 0), dx, dy, color="red", fill=None)
        ax.add_artist(tile_bbox)
        ax.axis("off")
        ax.set_aspect(1)

        ax : Axes = axs[1]
        pad = 100
        ax.set_xlim(0, (TILE_SIZE + pad) * len(xs) - pad)
        ax.set_ylim(0, (TILE_SIZE + pad) * len(ys) - pad)
        for ix, iy, x, y in [(ix, iy, x, y) for iy, y in enumerate(sorted(ys, reverse=True)) for ix, x in enumerate(sorted(xs))]:
            t = scaled_image[:, y:(y + TILE_SIZE), x:(x + TILE_SIZE)] #.permute(1, 2, 0).float().cpu()
            ax.imshow(
                model(t, single_scale=True).plot(), 
                extent=(
                    ix * (TILE_SIZE + pad), 
                    ix * (TILE_SIZE + pad) + TILE_SIZE,
                    iy * (TILE_SIZE + pad), 
                    iy * (TILE_SIZE + pad) + TILE_SIZE,
                )
            )
        ax.axis("off")
        ax.set_aspect(1)

        plt.tight_layout()

        fig.savefig(f"tile_and_pred_{scale:.2f}.png", transparent=True, bbox_inches="tight")
        plt.close()

    model(image).plot(outpath="full_prediction.png")

