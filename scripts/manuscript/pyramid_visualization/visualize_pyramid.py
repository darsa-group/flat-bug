import math
import os
from typing import List, Tuple

import cv2
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm as TQDM

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
    edge_case_margin_padding_multiplier = 0 # We don't want to do this for the visualization...
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
    TILE_PAD = round(TILE_SIZE / 100 * 5)
    TILE_PAD += TILE_PAD % 2
    DPI = 400
    # TILE_PAD = model.EDGE_CASE_MARGIN * 2
    vertical_spacing = 0.0175
    tile_row_colors = ["blue", "orange", "green", "magenta"]

    scales, prep_image = get_scales(model, image)
    
    nrow = len(scales)
    ncol = 2

    pyramid_params = {
        s : get_tile_params(model, prep_image, scale=s) 
        for s in reversed(scales)
    }

    # === Step 1. Compute each row’s height (in data units) from the left image panel ===
    # For the left panel, we show the scaled image with x‑limits (-5, dx+5) and y‑limits (-5, dy+5)
    # so that its data width is (dx+10) and data height is (dy+10).
    row_heights = []
    scales_order = list(pyramid_params.keys())
    for scale in scales_order:
        offsets, scaled_image = pyramid_params[scale]
        # scaled_image is (channels, height, width)
        _, dy, dx = scaled_image.shape
        row_heights.append(dy + 10)  # using the vertical extent as the row’s data height

    total_data_height = sum(row_heights)

    _, h, w = pyramid_params[scales_order[0]][1].shape
    asp = h / w
    
    conv_y = 1 / total_data_height  # conversion factor from data units to normalized figure units (vertical)
    conv_x = 1 / (2 * w)

    # Compute normalized row heights and their bottom coordinates (figure coordinates run 0=bottom, 1=top)
    norm_row_heights = [rh * conv_y / (1 + (nrow + 1) * vertical_spacing) for rh in row_heights]
    row_bottoms = []
    current_top = 1.0 - vertical_spacing
    for nrh in norm_row_heights:
        bottom = current_top - nrh - vertical_spacing
        row_bottoms.append(bottom)
        current_top = bottom

    # Create the figure. (figsize can be adjusted as needed.)
    canvas_height, canvas_width = (1 + (nrow + 1) * vertical_spacing) / conv_y, 1 / conv_x
    fig = plt.figure(figsize=(canvas_width / DPI, canvas_height / DPI), dpi=DPI)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # === Step 2. For each row, add two axes with custom positions so that the midline is at x=0.5
    for i, scale in enumerate(TQDM(scales_order)):
        offsets, scaled_image = pyramid_params[scale]
        ys, xs = [set(cs) for cs in zip(*[o[1] for o in offsets])]
        _, dy, dx = scaled_image.shape  # dy = height, dx = width
        bottom = row_bottoms[i]

        ## Calculate data ranges for each column
        # Left:
        left_data_width = dx + 10    # from -5 to dx+5
        left_data_height = dy + 10   # from -5 to dy+5
        # Right:
        right_data_width = (TILE_SIZE + TILE_PAD) * len(xs) - TILE_PAD
        right_data_height = (TILE_SIZE + TILE_PAD) * len(ys) - TILE_PAD
        right_data_asp = right_data_width / right_data_height
        # Use the left column as a reference for the row height
        row_norm_height = left_data_height * conv_y

        # --- Right panel ---
        # We force the right panel’s axes to have the same normalized width (and height) as the left panel.
        axr = fig.add_axes([0.5, bottom, (row_norm_height * right_data_asp / conv_y + 25) * conv_x, row_norm_height])
        # Set the full data range so that the mosaic is drawn, but the axes is fixed in size.
        axr.set_xlim(-30, right_data_width + 10)
        axr.set_ylim(-10, right_data_height + 10)
        # Do NOT force equal aspect here, so that the larger data range is squeezed into the same axes.
        # axr.set_aspect("auto")
        axr.axis("off")

        # Draw the predicted tiles.
        # (Tiles are placed according to the grid defined by sorted xs and ys.)
        for ix, iy, x, y in [
            (ix, iy, x, y)
            for iy, y in enumerate(sorted(ys))
            for ix, x in enumerate(sorted(xs))
        ]:
            this_tile = scaled_image[:, y:(y + TILE_SIZE), x:(x + TILE_SIZE)]
            this_tile_pred = model(this_tile, single_scale=True).plot()
            tile_x = ix * (TILE_SIZE + TILE_PAD)
            tile_y = iy * (TILE_SIZE + TILE_PAD)
            # contract_x = 1 if ix == 0 else -1 if ix == len(xs) else 0
            # contract_y = 1 if iy == 0 else -1 if iy == len(ys) else 0
            correct_x = 1 if ix == 0 else 0
            correct_y = 1 if iy == 0 else 0
            axr.imshow(
                this_tile_pred,
                extent=(
                    tile_x,
                    tile_x + TILE_SIZE,
                    tile_y,
                    tile_y + TILE_SIZE,
                ),
            )
            corr_xy = TILE_PAD/TILE_SIZE
            tile_border = Rectangle(
                xy          = (tile_x + math.floor(corr_xy), tile_y + math.floor(corr_xy)), 
                width       = TILE_SIZE - 2 * math.floor(corr_xy),
                height      = TILE_SIZE - 2 * math.floor(corr_xy),
                edgecolor   = tile_row_colors[iy % len(tile_row_colors)], 
                fill        = None,
                linewidth   = 1.5
            )
            axr.add_artist(tile_border)

        # --- Left panel ---
        # Place its axes so that its right edge is at x=0.5.
        axl = fig.add_axes([0.5 - (left_data_width + 25) * conv_x, bottom, (left_data_width + 25) * conv_x, row_norm_height])
        axl.imshow(scaled_image.cpu().permute(1, 2, 0).float())
        axl.set_xlim(-5, dx + 30)
        axl.set_ylim(-5, dy + 5)
        axl.set_xlabel(f'{dx}px')
        axl.set_ylabel(f'{dy}px')
        axl.set_aspect(1)  # show the image at native 1:1 scale
        axl.set_xticks([])
        axl.set_yticks([])
        axl.set_frame_on(False)
        
        # Draw tile boxes on the left panel.
        y_last = ri = 0
        for tile in offsets:
            _, (y, x) = tile
            ri += int(y != y_last)
            y_last = y
            tile_color = tile_row_colors[ri % len(tile_row_colors)]
            axl.add_artist(Rectangle((x, y), TILE_SIZE, TILE_SIZE, linewidth=1.5, edgecolor=tile_color, fill=None))
            axl.add_artist(Rectangle((x, y), TILE_SIZE, TILE_SIZE, alpha=0.25, edgecolor=None, facecolor=tile_color))
        # tile_bbox = Rectangle((0, 0), dx, dy, color="red", fill=None)
        # axl.add_artist(tile_bbox)

    plt.savefig("pyramid_tiling_visualization.png", transparent=True, bbox_inches="tight")
    plt.close()

    # model(image).plot(outpath="full_prediction.png")

