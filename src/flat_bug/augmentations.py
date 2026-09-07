"""Augmentations used for flatbug."""

import math
import random
from typing import cast, overload

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from ultralytics.data.augment import RandomPerspective
from ultralytics.utils.instance import Instances

from flat_bug import logger
from flat_bug.config import check_types


### From Ultralytics repository, remove clipping from `RandomPerspective` and add `apply_segments` function
def segment2box(segment: torch.Tensor, width: int = 640, height: int = 640) -> np.ndarray:
    """Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment: the segment label
        width: OBS: Unused. The width of the image. Defaults to 640.
        height: OBS: Unused. The height of the image. Defaults to 640.

    Returns:
        The minimum and maximum x and y values of the segment (xyxy).

    """
    x, y = segment.T  # segment xy
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)  # type: ignore


def apply_segments(segments: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply affine to segments and generate new bboxes from segments.

    Args:
        segments: list of segments, [num_samples, 500, 2].
        M: affine matrix.

    Returns:
        out:
        * new_segments (`np.ndarray`): list of segments after affine, [num_samples, 500, 2].
        * new_bboxes (`np.ndarray`): bboxes after affine, [N, 4].

    """
    n, num = segments.shape[:2]
    if n == 0:
        return np.empty(shape=(0,)), segments

    xy = np.ones((n * num, 3), dtype=segments.dtype)
    segments = segments.reshape(-1, 2)
    xy[:, :2] = segments
    xy = xy @ M.T  # transform
    xy = xy[:, :2] / xy[:, 2:3]
    segments = xy.reshape(n, -1, 2)
    bboxes = np.stack([segment2box(xy) for xy in segments], 0)
    segments[..., 0] = segments[..., 0].clip(bboxes[:, 0:1], bboxes[:, 2:3])
    segments[..., 1] = segments[..., 1].clip(bboxes[:, 1:2], bboxes[:, 3:4])
    return bboxes, segments


def low_res_inpaint(img: np.ndarray, mask: np.ndarray, scale: int = 6) -> np.ndarray:
    """Perform low resolution inpainting in-place.

    In-painting is done on a low-resolution copy of the image,
    and then copies the upsampled inpainted image back into the original image.

    Args:
        img: Image to be inpainted.
        mask: Mask to specify inpainting area.
        scale: Scale of the low-resolution copy used for inpainting.

    Returns:
        The original (modified) instance.

    """
    # Create a low-res version of the image and mask
    lr_img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
    lr_mask = cv2.resize(mask, (mask.shape[1] // scale, mask.shape[0] // scale))

    # Perform inpainting on the low-res image
    lr_inpainted = cv2.inpaint(lr_img, lr_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Copy the upsampled inpainted image back into the original image
    img[mask == 1] = cv2.resize(lr_inpainted, (img.shape[1], img.shape[0]))[mask == 1]

    return img


def telea_inpaint_polys(
    img: np.ndarray,
    polys: list[np.ndarray],
    exclude_polys: list[np.ndarray] | None = None,
    downscale_factor: int | float = 6,
    **kwargs,
) -> np.ndarray:
    """Mutably inpaints the polygons in an image using the Fast Marching method by Alexandru Telea.

    The inpainting algorithm is performed on a downsampled version of the image to speed up the process,
    and the inpainted results are then upsampled and pasted back into the original image.

    Args:
        img: The image to inpaint.
        polys: A list of polygons to inpaint.
        exclude_polys: A list of polygons to exclude from inpainting. Defaults to None.
        downscale_factor: The factor by which to downscale the image before inpainting. Defaults to 6.
        **kwargs: Additional keyword arguments to pass to `cv2.drawContours`.

    Returns:
        The inpainted image.

    """
    # Type checking and sanitizing
    check_types(img, np.ndarray)
    if not ((img.ndim == 3 and img.shape[2] < 5) or img.ndim == 2):
        raise ValueError(f"img must be a 2D or 3D numpy array, of shape (H, W) or (H, W, C), got shape {img.shape}")
    check_types(polys, [list, np.ndarray])
    check_types(exclude_polys, ([list, np.ndarray], None))  # type: ignore
    if exclude_polys is None:
        exclude_polys = []
    check_types(downscale_factor, (int, float))

    # Early return on no-op
    if len(polys) == 0:
        return img

    # Get the original and low-res image shapes
    orig_shape = img.shape[:2][::-1]
    low_res_size = [s // downscale_factor for s in orig_shape]

    # Initialize the inpaint bitmap and create the low-res image
    inpaint_bitmap = np.zeros(low_res_size[::-1], dtype=np.uint8)
    lr_img = cv2.resize(img, low_res_size)

    # Draw both the polygons and the exclusion polygons on the inpaint bitmap
    # This is done so that excluded polygons don't bleed into the inpainted regions
    for p in polys + exclude_polys:
        inpaint_bitmap = cv2.drawContours(
            inpaint_bitmap,
            [p // downscale_factor],
            color=1,
            **kwargs
        )
    
    # Dilate the inpaint bitmap to ensure that the inpainting doesn't bleed
    # from the edges of the instances under the polygons
    cv2.dilate(
        src=inpaint_bitmap,
        dst=inpaint_bitmap,
        kernel=np.ones((3, 3), np.uint8),
        iterations=1
    )

    # Inpaint the low-res image using the Fast Marching algorithm
    cv2.inpaint(
        src=lr_img,
        dst=lr_img,
        inpaintMask=inpaint_bitmap,
        inpaintRadius=5,
        flags=cv2.INPAINT_TELEA
    )

    # Remove the exclude polygons from the inpaint bitmap, so that the original image is not
    # inpainted under them. This must undo the DILATION as well as the polygon: the dilate
    # above grew every contour drawn, excluded ones included, so subtracting only the bare
    # polygon left a dilated ring - 1 low-res px, about `downscale_factor` px at full
    # resolution - still marked for inpainting, painting a smeared halo tight around every
    # KEPT instance.
    #
    # This runs in the validation pipeline as well as the training one, so the marker sat on
    # both sides of the train/val split: a ring hugging every labelled instance is a cue a
    # model can learn instead of the animal, and be rewarded for at validation time. Every
    # metric computed before this fix was measured against crops carrying it.
    exclude_bitmap = np.zeros_like(inpaint_bitmap)
    for p in exclude_polys:
        exclude_bitmap = cv2.drawContours(
            exclude_bitmap,
            [p // downscale_factor],
            color=1,
            **kwargs
        )
    cv2.dilate(
        src=exclude_bitmap,
        dst=exclude_bitmap,
        kernel=np.ones((3, 3), np.uint8),
        iterations=1
    )
    inpaint_bitmap[exclude_bitmap == 1] = 0

    # Upsample the inpainted image and bitmap
    inpaint_bitmap = cv2.resize(inpaint_bitmap, orig_shape)
    lr_img = cv2.resize(lr_img, orig_shape)

    # Copy the inpainted low-res image back into the original image
    img[inpaint_bitmap == 1] = lr_img[inpaint_bitmap == 1]

    # Return the inpainted image (not necessary, as the inpainting is done in-place)
    return img


@overload
def inpaint_pad(array: torch.Tensor, padding: int | tuple[int, int] | tuple[int, int, int, int]) -> torch.Tensor: ...
@overload
def inpaint_pad(array: np.ndarray, padding: int | tuple[int, int] | tuple[int, int, int, int]) -> np.ndarray: ...
def inpaint_pad(  # noqa: D103
    array: torch.Tensor | np.ndarray, padding: int | tuple[int, int] | tuple[int, int, int, int]
) -> torch.Tensor | np.ndarray:
    # Ensure padding is a tuple (pad_top, pad_bottom, pad_left, pad_right)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        lp = list(padding)
        padding = cast(tuple[int, int, int, int], tuple(lp + lp))
    elif isinstance(padding, (tuple, list)) and len(padding) == 4:
        pass
    else:
        raise TypeError(f"padding must be an integer or a tuple of length 2 or 4, got {type(padding)}")
    pad_t, pad_b, pad_l, pad_r = padding
    if pad_t == 0 and pad_b == 0 and pad_l == 0 and pad_r == 0:
        return array

    # Convert to integer whc numpy array
    is_tensor = isinstance(array, torch.Tensor)
    if is_tensor:
        device = array.device
        dtype = array.dtype
        array = array.cpu().numpy()

    # If array is not a integer multiply by 255
    array_is_integer = np.issubdtype(array.dtype, np.integer)
    if not array_is_integer:
        array = (array * 255).astype(np.uint8)

    # Convert array from CWH to HWC format
    is_cwh = len(array.shape) == 3 and array.shape[0] == 3 and not array.shape[2] == 3
    if is_cwh:
        array = array.transpose(1, 2, 0)

    # Get original dimensions
    original_h, original_w, _ = array.shape

    # Create padded image and mask
    padded_h = original_h + pad_t + pad_b
    padded_w = original_w + pad_l + pad_r

    padded_image = np.zeros((padded_h, padded_w, array.shape[2]), dtype=np.uint8)
    mask = np.ones((padded_h, padded_w), dtype=np.uint8)

    # Insert original image into the center of the padded image
    # padded_image[pad_h:pad_h + original_h, pad_w:pad_w + original_w] = array
    # mask[pad_h:pad_h + original_h, pad_w:pad_w + original_w] = 0
    h_slice = slice(pad_t, pad_t + original_h)
    w_slice = slice(pad_l, pad_l + original_w)
    padded_image[h_slice, w_slice] = array  # <-- HERE
    mask[h_slice, w_slice] = 0

    # Perform inpainting
    low_res_inpaint(padded_image, mask)

    # Convert back original format
    if is_cwh:
        padded_image = padded_image.transpose(2, 0, 1)
    if not array_is_integer:
        padded_image = padded_image.astype(np.float32) / 255
    if is_tensor:
        padded_image = torch.tensor(padded_image).to(device, dtype)

    return padded_image


class InpaintPad:  # noqa: D101
    def __init__(self, padding: int | tuple[int, int] | tuple[int, int, int, int]):  # noqa: D107
        self.padding = padding

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return inpaint_pad(tensor, self.padding)


def remove_instances(  # noqa: D103
    labels: dict, area_thr: float | int = 1, max_targets: float | int | None = 1000, min_size: int = 0
) -> dict:
    instances = cast(Instances, labels.pop("instances"))
    assert instances.segments is not None
    imsize = labels["img"].shape[:2][::-1]

    if instances.normalized:
        instances.denormalize(*imsize)
    if instances._bboxes.format != "xywh":
        instances.convert_bbox(format="xywh")

    bboxes = instances._bboxes.bboxes

    if bboxes.shape[0] == 0:
        labels["instances"] = Instances(
            np.empty([0, 4], dtype=np.float32), np.empty([0, 2], dtype=np.float32), normalized=False
        )
        labels["cls"] = np.empty((0), dtype=np.int32)
        return labels

    # Instead of using bounding boxes, we calculate the proportion of the segment area that is within the image
    eps = 1e-9
    image_bbox = box(0, 0, *imsize)
    area_ratios = np.zeros(bboxes.shape[0])
    for i, s in enumerate(instances.segments):
        # Initiate overlap using bounding box
        x, y, w, h = bboxes[i]
        bbox = box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)  # type: ignore
        bbox_iarea = bbox.intersection(image_bbox).area
        area_ratios[i] = (bbox_iarea + eps) / (bbox.area + eps) if bbox.area > 0 and bbox_iarea > 0 else 0
        if area_ratios[i] < area_thr:
            continue
        poly = make_valid(Polygon(s))
        intersection_area = poly.intersection(image_bbox).area
        area_ratios[i] = (intersection_area + eps) / (poly.area + eps) if poly.area > 0 and intersection_area > 0 else 0

    valid = np.all(
        [
            # (b[:, 0] - b[:, 2] / 2) / self._imsize > 0,
            # (b[:, 1] - b[:, 3] / 2) / self._imsize > 0,
            # (b[:, 0] + b[:, 2] / 2) / self._imsize < 1,
            # (b[:, 1] + b[:, 3] / 2) / self._imsize < 1,
            area_ratios >= area_thr,
            bboxes[:, 2] > min_size,
            bboxes[:, 3] > min_size,
        ],
        axis=0,
    )

    if max_targets is not None and np.sum(valid) > max_targets:
        n_remove, n_keep = int(np.sum(valid) - max_targets), int(max_targets)
        valid[valid] &= np.random.permutation(np.array([True] * n_keep + [False] * n_remove, dtype=bool))

    # here, we paint the edge cases (partially outside the image, using telea inpainting),
    # this should help learning. Indeed it would be very confusing if an image if an insect that is
    # 10% outside is flagged as NOT insect!

    invalid = np.bitwise_not(valid)
    invalid_visible = np.bitwise_and(invalid, area_ratios > 0)  # We only need to inpaint polygons within the frame

    invalid_i = np.nonzero(invalid_visible)[0]
    invalid_segments = instances.segments[invalid_i]
    invalid_segments = [np.array(s, dtype=np.int32) for s in invalid_segments]
    valid_segments = [np.array(s, dtype=np.int32) for s in instances.segments[np.nonzero(valid)[0]]]

    if len(invalid_segments):
        # DEBUG: This can be used to visualize the invalid segments, and is also the old way of inpainting
        # cv2.drawContours(
        #     labels["img"],
        #     invalid_segments,
        #     contourIdx=-1,
        #     # color=self.bg_fill,
        #     color=(0, 0, 255),
        #     thickness=-1,
        #     lineType=cv2.LINE_4,
        #     offset=(0,0)
        # )
        # Up-to-date inpainting method
        telea_inpaint_polys(
            img=labels["img"],
            polys=invalid_segments,
            exclude_polys=valid_segments,
            downscale_factor=6,
            contourIdx=-1,
            thickness=-1,
            lineType=cv2.LINE_4,
            offset=(0, 0),
        )

    # cv2.imwrite(f"/tmp/{os.path.basename(labels['im_file'])}", or_img)
    valid_i = np.nonzero(valid)[0]

    if len(valid_i) == 0:
        labels["instances"] = Instances(
            np.empty([0, 4], dtype=np.float32), np.empty([0, 2], dtype=np.float32), normalized=False
        )
        labels["cls"] = np.empty_like(labels["cls"])
        return labels

    # DEBUG: plot boxes on image
    # for bbox in bboxes[valid_i, :]:
    #     x, y, w, h = bbox
    #     x0, y0, x1, y1 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    #     cv2.rectangle(labels["img"], (x0, y0), (x1, y1), (0, 255, 0), 2)
    # for bbox in bboxes[invalid_i, :]:
    #     x, y, w, h = bbox
    #     x0, y0, x1, y1 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
    #     cv2.rectangle(labels["img"], (x0, y0), (x1, y1), (255, 0, 0), 2)
    instances.segments = instances.segments[valid_i]
    instances._bboxes.bboxes = bboxes[valid_i]
    instances.clip(*imsize)
    labels["cls"] = labels["cls"][valid_i]

    labels["instances"] = instances
    # logger.info(labels)
    return labels


def scale_labels(  # noqa: D103
    labels: dict, scale: float
) -> dict:
    orig_shape = labels["img"].shape[:2]
    # Scale the image
    labels["img"] = cv2.resize(labels["img"], (0, 0), fx=scale, fy=scale)
    new_shape = labels["img"].shape[:2]
    labels["resized_shape"] = new_shape
    labels["ori_shape"] = new_shape
    # Scale the instances
    labels["instances"].normalize(*orig_shape[::-1])
    labels["instances"].denormalize(*new_shape[::-1])
    return labels


class FlatBugRandomPerspective(RandomPerspective):  # noqa: D101
    fill_value = (0, 0, 0)
    size: tuple[int, int]

    def __init__(self, imgsz: int, *args, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)
        self.imgsz = imgsz

    def affine_transform(self, img: np.ndarray, border: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
        """Center."""
        self.scale = self.imgsz / max(img.shape), 1  # fime hardcoded
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(self.scale[0], self.scale[1])

        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        img_transform_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=self.fill_value)
                img_transform_mask = cv2.warpPerspective(img_transform_mask, M, dsize=self.size, borderValue=1)
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=self.fill_value)
                img_transform_mask = cv2.warpAffine(img_transform_mask, M[:2], dsize=self.size, borderValue=1)

        low_res_inpaint(img, img_transform_mask, scale=6)

        return img, M, s

    def __call__(self, labels: dict):
        """Affine images and targets.

        Args:
            labels: a dict of `bboxes`, `segments`, `keypoints`.

        """
        # if self.pre_transform and "mosaic_border" not in labels:
        #     labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad
        img = labels["img"]
        cls = labels["cls"]
        instances: Instances = labels.pop("instances")
        # Make sure the coord formats are right
        if instances._bboxes.format != "xyxy":
            instances.convert_bbox(format="xyxy")
        if instances.normalized:
            instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", (0, 0))
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        assert segments is not None
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M, self.size)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)

        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01 if len(segments) else 0.10
        )
        labels["instances"] = new_instances[i]
        labels["instances"].convert_bbox(format="xywh")
        labels["cls"] = cls[i]
        # if len(cls) > 0:
        #     labels["cls"] = cls[i]
        # else:
        #     labels["cls"] = np.empty((0), dtype=np.int32)
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]

        # labels["instances"].normalize(*labels["resized_shape"][::-1])
        return labels


class Crop:
    """Abstact crop-related augmentation."""

    bg_fill = (0, 0, 0)
    min_size = 0  # px

    def __init__(self, imsize: int | tuple[int, int] | list[int] | np.ndarray):  # noqa: D107
        if isinstance(imsize, int):
            self._imsize = (imsize, imsize)
        elif isinstance(imsize, (tuple, list, np.ndarray)):
            if len(imsize) == 1:
                self._imsize = (imsize[0], imsize[0])
            elif len(imsize) != 2:
                raise ValueError("imsize should be a list of length 2")
            self._imsize = imsize
        else:
            raise TypeError(f"`imsize` should be of type `int`, `tuple`, or `list`, got {type(imsize)}")
        self._imsize = tuple([int(i) for i in self._imsize])
        self.xsize, self.ysize = self._imsize

    def crop_image(  # noqa: D102
        self, labels: dict, start_x: int, start_y: int, size_x: int, size_y: int
    ) -> dict:
        img = labels["img"]
        orig_shape = img.shape
        h, w = img.shape[:2]

        px0 = -min(0, start_x)
        n_start_x = max(0, start_x)
        px1 = -min(0, w - (start_x + size_x))
        px = px0 + px1
        n_size_x = size_x - px

        py0 = -min(0, start_y)
        n_start_y = max(0, start_y)
        py1 = -min(0, h - (start_y + size_y))
        py = py0 + py1
        n_size_y = size_y - py

        img = img[n_start_y : n_start_y + n_size_y, n_start_x : n_start_x + n_size_x, :]

        if px > 0 or py > 0:
            img = np.pad(img, pad_width=((py0, py1), (px0, px1), (0, 0)), mode="constant", constant_values=0.0)
            # img = inpaint_pad(img, (py0, py1, px0, px1)) # Fixme: this is very slow for large images

        if img.shape != (size_x, size_y, 3):
            logger.info("shape:", img.shape)
            logger.info("or-shape", orig_shape)
            logger.info("x, y:", start_x, start_y)
            logger.info(labels["im_file"])  # fixme, this is also done during validation?!

        assert img.shape == (size_x, size_y, 3), f"{img.shape}, ({size_x}, {size_y})"

        labels["ori_shape"] = (size_x, size_y)
        labels["resized_shape"] = (size_x, size_y)
        labels["img"] = np.copy(img, order="C")

        # Fix label positions

        instances = cast(Instances, labels.pop("instances"))
        if instances._bboxes.format != "xywh":
            instances.convert_bbox(format="xywh")
        if instances.normalized:
            instances.denormalize(*orig_shape[:2][::-1])

        labels["ratio_pad"] = ((1.0, 1.0), (0.0, 0.0))
        x_offset = -n_start_x + px0
        y_offset = -n_start_y + py0

        # positions in the cropped image
        instances._bboxes.add([x_offset, y_offset, 0, 0])

        assert instances.segments is not None
        for s in instances.segments:
            s[:, 0] += x_offset
            s[:, 1] += y_offset

        labels["instances"] = instances

        return labels

    def __call__(self, x):
        """Abstract function.

        Should be implemented in subclasses.
        """
        raise NotImplementedError("This method should be implemented in a subclass")


class CenterCrop(Crop):  # noqa: D101
    def __call__(self, labels: dict) -> dict:  # noqa: D102
        h, w = labels["img"].shape[:2]

        start_x = (w - self.xsize) // 2
        start_y = (h - self.ysize) // 2

        return self.crop_image(labels, start_x, start_y, self.xsize, self.ysize)


class RandomCrop(Crop):  # noqa: D101
    def __init__(self, *args, **kwargs):  # noqa: D107
        super().__init__(*args, **kwargs)

    def __call__(self, labels: dict) -> dict:  # noqa: D102
        # Get the initial image to target crop size ratio
        h, w = labels["img"].shape[:2]
        target_source_ratio_h = self.ysize / h
        target_source_ratio_w = self.xsize / w
        # We need to select a single scaling factor for both dimensions
        min_target_source_ratio = min(target_source_ratio_h, target_source_ratio_w)

        # If the image is larger than the target size we scale between crop_dim/image_dim, 1
        if min_target_source_ratio < 1:
            scale = np.random.uniform(min_target_source_ratio, 1) ** 2
        # If the image is smaller than the target size we scale between 1, and crop_dim/image_dim
        else:
            scale = np.random.uniform(1, min_target_source_ratio) ** (1 / 2)

        # When we scale up, this is done before cropping
        do_scale_before = scale > 1
        if do_scale_before:
            labels = scale_labels(labels, scale)
            target_xsize, target_ysize = self.xsize, self.ysize
        else:
            target_size = max(int(w * scale), int(h * scale))
            target_xsize, target_ysize = target_size, target_size
            # Reset the scale such that when the labels/image are
            # scaled after cropping the size is self.xsize, self.ysize (assuming these are equal)
            scale = self.xsize / target_xsize

        # Calculate possible crop start positions
        h, w = labels["img"].shape[:2]
        if w <= target_xsize:
            start_x = (w - target_xsize) // 2
        else:
            start_x = np.random.randint(-target_xsize // 2, w - target_xsize // 2, size=1)[0]
        if h <= target_ysize:
            start_y = (h - target_ysize) // 2
        else:
            start_y = np.random.randint(-target_ysize // 2, h - target_ysize // 2, size=1)[0]
        labels = self.crop_image(labels, start_x, start_y, target_xsize, target_ysize)

        # When we scale down we do this after cropping
        if not do_scale_before:
            labels = scale_labels(labels, scale)

        return labels


class ZoomCrop(Crop):
    """Crop centred on one instance and magnified, so appendages are seen at high resolution.

    ``RandomCrop`` places crops uniformly over the image and draws its scale as
    ``uniform(ratio, 1) ** 2``, which biases towards downscaling a whole image into the tile.
    The consequence is that a leg three pixels wide is three pixels wide in most training
    crops, and a network cannot learn a structure it cannot resolve. This crop does the
    opposite: it picks one sufficiently large instance and magnifies it to fill a set
    fraction of the tile.

    It matches inference rather than departing from it - flat-bug predicts over a scale
    pyramid, so magnified views are already what the finest pyramid level presents.

    Two rules keep it honest:

    - Instances below ``drop_below`` px in the ORIGINAL image are removed and inpainted
      before any magnification. Without this, zooming 5x would lift a 20px instance over
      the 32px floor that ``FixInstances`` applies downstream, smuggling in exactly the
      poorly resolved annotations that floor exists to exclude.
    - Every other instance falling in the crop keeps its label. Dropping a label while
      leaving its pixels manufactures a false negative, which is the failure the Telea
      inpainting in ``FixInstances`` exists to prevent.

    The centre is jittered so the target does not always sit dead-centre, which would teach
    "one object, middle of frame" and hurt on the crowded scenes flat-bug is built for.
    """

    def __init__(
        self,
        imsize: int | tuple[int, int] | list[int] | np.ndarray,
        min_px: int = 100,
        drop_below: int = 32,
        occupancy: tuple[float, float] = (0.22, 0.45),
        max_zoom: float = 8.0,
        min_scale: float = 1.0,
        jitter: float = 0.25,
    ) -> None:
        """.

        Args:
            imsize: Output crop size.
            min_px: An instance must be at least this many px (longest box side) in the
                original image to be chosen as the zoom target. Large instances are chosen
                because their annotations are the trustworthy ones - their legs are actually
                drawn - so they are what should teach high-resolution mask appearance.
            drop_below: Instances smaller than this in the original image are removed and
                inpainted before magnification.
            occupancy: Fraction of the tile the target should span, sampled uniformly. Kept
                modest deliberately: magnification shrinks the crop in SOURCE pixels, so at
                6x a 1536px tile sees only ~250px of original image and nearly every
                neighbour is clipped by the frame. Measured on a crowded synthetic plate,
                occupancy (0.30, 0.60) left 1.2 of 40 instances labelled and painted out 5.1%
                of the tile, against 7.6 and 1.1% for a normal crop - i.e. the crop degrades
                into a single-instance scene. A lower occupancy keeps more real neighbours.
            max_zoom: Never magnify beyond this, to avoid training on interpolated pixels.
            min_scale: Lower bound on the scale factor. At the default 1.0 an instance already
                larger than ``xsize * occupancy`` is left at native scale, which on this corpus
                means about half of all picks are not magnified at all. Setting it below 1.0
                lets those be scaled DOWN to the target occupancy instead, so every pick lands
                at the same size in the tile. Do not set it so low that the source crop
                (``xsize / zoom``) dwarfs the image, or the tile is mostly padding.
            jitter: Centre offset as a fraction of the crop side.
        """
        super().__init__(imsize)
        self.min_px = int(min_px)
        self.drop_below = int(drop_below)
        self.occupancy = occupancy
        self.max_zoom = float(max_zoom)
        self.min_scale = float(min_scale)
        self.jitter = float(jitter)

    def __call__(self, labels: dict) -> dict | None:
        """Crop and magnify, or return None if no instance is large enough to zoom on."""
        instances = cast(Instances, labels["instances"])
        if instances.segments is None or len(instances.segments) == 0:
            return None
        h, w = labels["img"].shape[:2]
        if instances.normalized:
            instances.denormalize(w, h)
        if instances._bboxes.format != "xywh":
            instances.convert_bbox(format="xywh")
        bboxes = instances._bboxes.bboxes
        sizes = np.maximum(bboxes[:, 2], bboxes[:, 3])

        eligible = np.nonzero(sizes >= self.min_px)[0]
        if len(eligible) == 0:  # nothing worth zooming on; caller falls back
            return None

        t = int(np.random.choice(eligible))
        cx, cy = float(bboxes[t, 0]), float(bboxes[t, 1])
        occ = float(np.random.uniform(*self.occupancy))
        # min_scale < 1 lets an instance larger than the target occupancy be scaled DOWN to it,
        # so every picked instance lands at the same size in the tile instead of half of them
        # being left untouched at zoom 1.0.
        zoom = float(np.clip(self.xsize * occ / max(sizes[t], 1.0), self.min_scale, self.max_zoom))
        side = max(int(round(self.xsize / zoom)), 8)

        j = self.jitter * side
        cx += float(np.random.uniform(-j, j))
        cy += float(np.random.uniform(-j, j))
        labels = self.crop_image(labels, int(round(cx - side / 2)), int(round(cy - side / 2)), side, side)

        # Drop the too-small instances and inpaint their pixels AFTER cropping but BEFORE the
        # magnifying resample. Sizes are still in source pixels here, so the rule is applied
        # at the original scale exactly as if it had run on the whole image - but the
        # inpainting touches only the ~side^2 crop instead of the entire source image, which
        # on flat-bug's larger images is the difference between a cheap call and a very
        # expensive one.
        inst = cast(Instances, labels["instances"])
        bb = inst._bboxes.bboxes
        sz = np.maximum(bb[:, 2], bb[:, 3])
        vis = np.array([
            s[:, 0].max() > 0 and s[:, 1].max() > 0 and s[:, 0].min() < side and s[:, 1].min() < side
            for s in inst.segments
        ]) if len(inst.segments) else np.zeros(0, bool)
        tiny = np.nonzero((sz < self.drop_below) & vis)[0]
        if len(tiny):
            keep = np.nonzero(~((sz < self.drop_below) & vis))[0]
            telea_inpaint_polys(
                img=labels["img"],
                polys=[np.asarray(s, dtype=np.int32) for s in inst.segments[tiny]],
                exclude_polys=[np.asarray(s, dtype=np.int32) for s in inst.segments[keep]],
                downscale_factor=6,
                contourIdx=-1,
                thickness=-1,
                lineType=cv2.LINE_4,
            )
            inst.segments = inst.segments[keep]
            inst._bboxes.bboxes = bb[keep]
            labels["cls"] = labels["cls"][keep]
            labels["instances"] = inst

        return scale_labels(labels, self.xsize / side)


class MaybeZoomCrop:
    """Take a magnified single-instance crop with probability ``p``, else the usual crop.

    The fallback is not just for the probability draw: ``ZoomCrop`` returns None whenever an
    image holds no instance at least ``min_px`` across, which is common in the small-animal
    sub-datasets, and those images must still contribute normal crops.
    """

    def __init__(self, random_crop: "RandomCrop", zoom_crop: ZoomCrop, p: float) -> None:
        """.

        Args:
            random_crop: The standard crop, used for the remaining ``1 - p``.
            zoom_crop: The magnifying crop.
            p: Probability of attempting a zoomed crop.
        """
        self.random_crop = random_crop
        self.zoom_crop = zoom_crop
        self.p = float(p)

    def __call__(self, labels: dict) -> dict:  # noqa: D102
        if self.p > 0 and np.random.random() < self.p:
            out = self.zoom_crop(labels)
            if out is not None:
                return out
        return self.random_crop(labels)


class FixInstances:
    """Removes instances that are too small or which overlap less than a certain threshold with the image."""

    def __init__(self, area_thr: float | int, max_targets: int | float | None, min_size: int):
        """.

        Args:
            area_thr: The minimum proportion of the instance that must be within the image in order for it to be kept.
            max_targets: The maximum number of instances to keep. If there are more instances than this,
                a random subset of instances will be kept. If `None`, all instances will be kept.
            min_size: The minimum size of the bounding box of the instance.
                Instances with a width or height less than this value will be removed.

        """
        self.area_thr = area_thr
        self.max_targets = max_targets if max_targets is None or max_targets > 0 else None
        self.min_size = min_size

    def __call__(self, labels: dict) -> dict:
        """Fix instances.

        Args:
            labels: Dictionary containing the instances.

        Returns:
            out: A dictionary containing the updated instances.

        """
        return remove_instances(labels, area_thr=self.area_thr, max_targets=self.max_targets, min_size=self.min_size)


class RandomColorInv:  # noqa: D101
    def __init__(self, p: float = 0.5):
        """Invert the colors of an image with a probability p.

        Args:
            p: probability of inverting the colors. Defaults to 0.5

        """
        if p < 0:
            logger.warning("p should be in [0,1], got", p, "setting to 0")
            p = 0
        if p > 1:
            logger.warning("p should be in [0,1], got", p, "setting to 1")
            p = 1
        self.p = 1 - p

    def __call__(self, labels: dict) -> dict:  # noqa: D102
        img = labels["img"]
        if random.uniform(0, 1) > self.p:
            assert img.dtype == np.uint8
            labels["img"] = 255 - img
        return labels
