from math import ceil, floor, log
from typing import Tuple, List, Generator

import torch
from torchvision import transforms

def adjust_image_for_tile(
        image_size : Tuple[int, int],
        tile_size : int,
        scale : float    
    ) -> Tuple[int, int, int, int, int, int]:
    """
    Adjusts the image size to be divisible by 4 and to fit the tile size.
    
    Args:
        image_size (`Tuple[int, int]`): The size of the image to adjust.
        tile_size (`int`): The size of the tiles.
        scale (`float`): The scaling factor of the layer.

    Returns:
        `Tuple[int, int, int, int, int, int]`: The adjusted image size and padding (h, w, pad_l, pad_t, pad_r, pad_b).
    """
    # First we ensure that the image size is divisible by 4
    h, w = image_size
    if scale != 1:
        h, w = [round(s * scale / 4) * 4 for s in (h, w)]
    # And that the tile can fit in the image by padded
    pad = [max(0, tile_size - s) for s in (h, w)]
    ltrb_pad = floor(pad[1] / 2), ceil(pad[0] / 2), ceil(pad[1] / 2), floor(pad[0] / 2)
    return h, w, *ltrb_pad

def calculate_tile_parameters(
        h : int,
        w : int,
        tile_size : int,
        min_overlap : int    
    ) -> Tuple[int, int, int, int]:
    """
    Calculates the number of tiles in the x and y directions, the stride of the tiles in the x and y directions, 
    given the size of the image, the size of the tiles and the minimum overlap between tiles.

    Args:
        h (`int`): The height of the image.
        w (`int`): The width of the image.
        tile_size (`int`): The size of the tiles.
        min_overlap (`int`): The minimum overlap between tiles.

    Returns:
        `Tuple[int, int, int, int]`: The number of tiles in the x and y directions, and the stride of the tiles in the x and y directions.
    """
    # Calculate the number of tiles in the x and y directions
    x_n_tiles = ceil(w / (tile_size - min_overlap)) if w != tile_size else 1
    y_n_tiles = ceil(h / (tile_size - min_overlap)) if h != tile_size else 1

    # Calculate the actual overlap of the tiles in the x and y directions
    x_overlap = floor((tile_size * x_n_tiles - w) / x_n_tiles) if x_n_tiles > 1 else 0
    y_overlap = floor((tile_size * y_n_tiles - h) / y_n_tiles) if y_n_tiles > 1 else 0
    # Ensure that the overlap is divisible by 4
    x_overlap += (4 - x_overlap % 4) if x_overlap % 4 != 0 else 0 
    y_overlap += (4 - y_overlap % 4) if y_overlap % 4 != 0 else 0

    # Calculate the stride of the tiles in the x and y directions 
    x_stride = tile_size - x_overlap
    y_stride = tile_size - y_overlap

    return x_n_tiles, y_n_tiles, x_stride, y_stride

def count_tiles(
        image_size : Tuple[int, int],
        tile_size : int,
        min_overlap : int,
        scale : float
    ) -> int:
    """
    Calculates the number of tiles needed to tile a single layer of an image pyramid.

    Args:
        image_size (`Tuple[int, int]`): The size of the image to tile.
        tile_size (`int`): The size of the tiles.
        min_overlap (`int`): The minimum overlap between tiles.
        scale (`float`): The scaling factor of the layer.

    Returns:
        `int`: The number of tiles needed to tile the layer.
    """
    h, w, l_pad, t_pad, r_pad, b_pad = adjust_image_for_tile(image_size, tile_size, scale)
    h, w = h + t_pad + b_pad, h + t_pad + b_pad
    x_n_tiles, y_n_tiles, _, _ = calculate_tile_parameters(h, w, tile_size, min_overlap)

    return x_n_tiles * y_n_tiles

def tile_layer(
        image_size : Tuple[int, int],
        tile_size : int,
        min_overlap : int,
        scale : float
    ) -> Tuple[Tuple[int, int, int, int], List[Tuple[int, int]]]:
    """
    Calculates the offsets of the tiles in a single layer of an image pyramid.

    Args:
        image_size (`Tuple[int, int]`): The size of the image to tile.
        tile_size (`int`): The size of the tiles.
        min_overlap (`int`): The minimum overlap between tiles.
        scale (`float`): The scaling factor of the layer.

    Returns:
        `Tuple[Tuple[int, int, int, int], List[Tuple[int, int]]]`: The target image size and padding, and the offsets of the tiles.
    """
    h, w, l_pad, t_pad, r_pad, b_pad = adjust_image_for_tile(image_size, tile_size, scale)
    h, w = h + t_pad + b_pad, h + t_pad + b_pad
    x_n_tiles, y_n_tiles, x_stride, y_stride = calculate_tile_parameters(h, w, tile_size, min_overlap)
    
    # Calculate the offsets of the tiles
    x_offsets = [i if (i + tile_size) < w else (w - tile_size - w % 4) for i in range(0, x_stride * x_n_tiles, x_stride)]
    y_offsets = [i if (i + tile_size) < h else (h - tile_size - h % 4) for i in range(0, y_stride * y_n_tiles, y_stride)]

    # Return the offsets of the tiles
    return (h, w, l_pad, t_pad, r_pad, b_pad), [
        ((y_tile_index, x_tile_index), (y_offset, x_offset)) 
        for y_tile_index, y_offset in enumerate(y_offsets) 
        for x_tile_index, x_offset in enumerate(x_offsets)
    ]

def fit_pyramid_scales(
        image_size : Tuple[int, int],
        tile_size : int,
        min_overlap : int,
        minimum_instance_size : int
    ) -> List[float]:
    """
    Creates the optimal set of scales for pyramid tiling, given the image size, tile size, overlap and minimum instance size.

    Based on the following constraints:
        - Lower bound: scaled `minimum_instance_size` = `minimum_instance_size` * (scale / scale_0)
        - Upper bound: scaled `min_overlap` = `min_overlap` * scale / scale_0
    
    Then, the next scale is calculated as:
        - LowerBound_{i + 1}  = UpperBound_i
        - `minimum_instance_size` * (scale_{i + 1} / scale_0) = `min_overlap` * scale_i / scale_0
        - scale_{i + 1} = (`min_overlap` * scale_i / scale_0) * scale_0 / `minimum_instance_size`
        - scale_{i + 1} = (`min_overlap` / `minimum_instance_size`) * scale_i
        
    This can also be simplified to:
        - scale_{i} = (`min_overlap` / `minimum_instance_size`)^i * scale_0
    where the number of scales is given by:
        - n_scales = ceil(log(max_dimension / tile_size) / log(`min_overlap` / `minimum_instance_size`)) + 1 if max_dimension > tile_size else 1
    But since the speed of this calculation is not a concern, we use the easier to understand iterative approach.


    Args:
        image_size (`Tuple[int, int]`): The size of the image to tile.
        tile_size (`int`): The size of the tiles.
        min_overlap (`int`): The minimum overlap between tiles.
        minimum_instance_size (`int`): The minimum size of that any instance between `minimum_instance_size` and max(image_size) is guaranteed to be seen at in a single tile.

    Returns:
        `List[float]`: The optimal set of scales for pyramid tiling.
    """
    max_dimension = max(image_size)
    scale_0 = tile_size / max_dimension
    
    scales = [scale_0]
    while scales[-1] < 1:
        upper_bound = min_overlap * scales[-1] / scale_0
        next_scale = (upper_bound + 1) * scale_0 / minimum_instance_size
        scales.append(min(1, next_scale))
    
    while len(scales) > 2 and scales[-2] > 0.9:
        scales.pop(-2)
    
    return scales

def count_pyramid_tiles(
        image_size : Tuple[int, int],
        tile_size : int,
        min_overlap : int,
        minimum_instance_size : int
    ) -> int:
    """
    Calculates the number of tiles needed to tile an image pyramid.

    Args:
        image_size (`Tuple[int, int]`): The size of the image to tile.
        tile_size (`int`): The size of the tiles.
        min_overlap (`int`): The minimum overlap between tiles.
        minimum_instance_size (`int`): The minimum size of that any instance between `minimum_instance_size` and max(image_size) is guaranteed to be seen at in a single tile.
    """
    scales = fit_pyramid_scales(
        image_size,
        tile_size,
        min_overlap,
        minimum_instance_size
    )

    return sum(count_tiles(image_size, tile_size, min_overlap, scale) for scale in scales)

def find_best_overlap(
        image_size : Tuple[int, int],
        tile_size : int,
        minimum_instance_size : int
    ) -> int:
    """
    Finds the best overlap for tiling an image pyramid.

    Args:
        image_size (`Tuple[int, int]`): The size of the image to tile.
        tile_size (`int`): The size of the tiles.
        minimum_instance_size (`int`): The minimum size of that any instance between `minimum_instance_size` and max(image_size) is guaranteed to be seen at in a single tile.

    Returns:
        `int`: The best overlap for tiling the image pyramid.
    """
    # The overlap must be at least 1 greater than the minimum instance size to capture the smallest possible instances, 
    # and at most half the tile size (reasonable practical upper bound)
    overlaps = list(range(minimum_instance_size + 1, tile_size // 2))

    # Calculate the cost of tiling the pyramid for each overlap
    cost = [count_pyramid_tiles(image_size, tile_size, o, minimum_instance_size) for o in overlaps]

    # The best cost is given by the largest overlap that gives the minimum cost
    return overlaps[len(cost) - cost[::-1].index(min(cost)) - 1]

def functional_pyramid_tiling(
        image_size : Tuple[int, int],
        tile_size : int,
        minimum_instance_size : int,
        instance_edge_buffer : int = 0
    ) -> Generator[Tuple[Tuple[int, int, int, int, int, int], List[Tuple[int, int]]], None, None]:
    """
    Calculates the number of tiles needed to tile an image pyramid.

    Args:
        image_size (`Tuple[int, int]`): The size of the image to tile.
        tile_size (`int`): The size of the tiles.
        minimum_instance_size (`int`): The minimum size of that any instance between `minimum_instance_size` and max(image_size) is guaranteed to be seen at in a single tile.
        instance_edge_buffer (`int`): The buffer around the instances to avoid splitting them between tiles.

    Returns:
        `Generator[Tuple[Tuple[int, int, int, int, int, int], List[Tuple[int, int]]], None, None]`: The target image size and padding, and the offsets of the tiles.
    """ 
    minimum_instance_size = minimum_instance_size + 2 * instance_edge_buffer
    best_overlap = find_best_overlap(image_size, tile_size, minimum_instance_size)[0]
    for scale in fit_pyramid_scales(image_size, tile_size, best_overlap, minimum_instance_size):
        yield tile_layer(image_size, tile_size, best_overlap, scale)

class PyramidLayer:
    def __init__(self, hw_pad : Tuple[int, int, int, int], offsets : List[Tuple[int, int]]):
        self.hw = hw_pad[:2]
        self.ltrb_pad = hw_pad[2:]

        self.offsets = offsets

        self.resizer = transforms.Resize(self.hw, antialias=True)
        self.padder = transforms.Pad(self.ltrb_pad, fill=0, padding_mode='constant') if any(self.ltrb_pad) else None

    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, index):
        return self.offsets[index]
    
    def __iter__(self):
        return iter(self.offsets)
    
    def __str__(self):
        hw_str = f"{str(self.hw[0]):>5} x {str(self.hw[1]):<5}"
        pad_str = f"(L={str(self.ltrb_pad[0]):<3}, T={str(self.ltrb_pad[1]):<3}, R={str(self.ltrb_pad[2]):<3}, B={str(self.ltrb_pad[3]):<3})"
        return f"PyramidLayer[ Tiles: {str(len(self)):<3} | Size: {hw_str} | Padding: {pad_str} ]"
    
    def adjust_tensor_image(self, tensor_image : torch.Tensor) -> torch.Tensor:
        """
        Adjusts a tensor to the pyramid layer.

        Args:
            tensor_image (`torch.Tensor`): The tensor to adjust. The tensor should have shape (C, H, W).
        """
        tensor_image = self.resizer(tensor_image)
        if self.padder is not None:
            tensor_image = self.padder(tensor_image)
        return tensor_image

class PyramidTilingSolution:
    def __init__(
        self,
        image_size: Tuple[int, int],
        tile_size: int,
        minimum_instance_size: int,
        instance_edge_buffer: int = 0
    ):
        self.image_size = image_size
        self.tile_size = tile_size
        self.minimum_instance_size = minimum_instance_size
        self.instance_edge_buffer = instance_edge_buffer

        adjusted_min_instance_size = minimum_instance_size + 2 * instance_edge_buffer
        self.best_overlap = find_best_overlap(image_size, tile_size, adjusted_min_instance_size)
        self.scales = fit_pyramid_scales(image_size, tile_size, self.best_overlap, adjusted_min_instance_size)
        self.layers = [PyramidLayer(*tile_layer(image_size, tile_size, self.best_overlap, scale)) for scale in self.scales]

    def __len__(self):
        return len(self.scales)

    def __getitem__(self, index):
        return self.layers[index]

    def __iter__(self):
        return iter(self.layers)

    @property
    def n_layers(self) -> int:
        """Returns the number of layers in the pyramid."""
        return len(self.scales)

    @property
    def n_tiles(self) -> int:
        """Returns the total number of tiles across all layers."""
        return sum(len(layer) for layer in self.layers)
    
    def statistics(self) -> dict:
        """Returns statistics about the tiling solution."""
        return {
            "best_overlap": self.best_overlap,
            "n_layers": self.n_layers,
            "n_tiles": self.n_tiles,
            "scales": self.scales,
            "layers": [str(layer) for layer in self.layers]
        }
    
    def __str__(self):
        call = "".join([
            f"PyramidTilingSolution("
            f"image_size={self.image_size}, " 
            f"tile_size={self.tile_size}, "
            f"minimum_instance_size={self.minimum_instance_size}, "
            f"instance_edge_buffer={self.instance_edge_buffer})"
        ])
        statistics = []
        for key, value in self.statistics().items():
            if key == "layers":
                value = "\n\t " + "\n\t ".join([f" {i}) {layer}" for i, layer in enumerate(value)])
            if key == "scales":
                value = "[" + ", ".join([f"{scale:.3f}" for scale in value]) + "]"
            statistics.append(f"{key} : {value}")
        return f"{call}\n" + "\n".join(statistics)

    def plot(self):
        """Plots the tiling solution over the image."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        h, w = self.image_size
        asp = w / h
        nrow = ceil(self.n_layers ** 0.5)
        ncol = ceil(self.n_layers / nrow)
        if nrow * ncol < self.n_layers:
            nrow += 1
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2 * asp, nrow * 2))
        if self.n_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for layer_index, (ax, scale, layer) in enumerate(zip(axes, self.scales, self)):
            ax.set_xlim(0, w)
            ax.set_ylim(0, h)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_title(f"Layer {layer_index + 1}")
            ax.axis('off')
            for (_, _), (y_offset, x_offset) in layer:
                tile_h = self.tile_size / scale
                tile_w = self.tile_size / scale
                rect = patches.Rectangle(
                    (x_offset / scale, y_offset / scale),
                    tile_w,
                    tile_h,
                    linewidth=1,
                    edgecolor="black",
                    facecolor="firebrick",
                    alpha=0.5
                )
                ax.add_patch(rect)
        for ax in axes[self.n_layers:]:
            ax.axis('off')
        plt.show()

class PyramidTiling:
    def __init__(self, tile_size : int, minimum_instance_size : int, instance_edge_buffer : int = 0):
        self.tile_size = tile_size
        self.minimum_instance_size = minimum_instance_size
        self.instance_edge_buffer = instance_edge_buffer
    
    def __call__(self, image_size : Tuple[int, int]):
        return PyramidTilingSolution(image_size, self.tile_size, self.minimum_instance_size, self.instance_edge_buffer)
