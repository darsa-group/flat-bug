import math
import random
from typing import Dict, List, Optional, Self, Tuple, Union

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
def segment2box(
        segment : torch.Tensor, 
        width : int=640, 
        height : int=640
    ) -> np.ndarray:
    """
    Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy).

    Args:
        segment (`torch.Tensor`): the segment label
        width (`int`, optional): OBS: Unused. The width of the image. Defaults to 640.
        height (`int`, optional): OBS: Unused. The height of the image. Defaults to 640. 

    Returns:
        out (`np.ndarray`): the minimum and maximum x and y values of the segment.
    """
    x, y = segment.T  # segment xy
    return np.array([x.min(), y.min(), x.max(), y.max()], dtype=segment.dtype)  # xyxy

def apply_segments(
        segments : np.ndarray, 
        M : np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply affine to segments and generate new bboxes from segments.

    Args:
        segments (`np.ndarray`): list of segments, [num_samples, 500, 2].
        M (`np.ndarray`): affine matrix.

    Returns:
        out (`Tuple[np.ndarray, np.ndarray]`):
        * new_segments (`np.ndarray`): list of segments after affine, [num_samples, 500, 2].
        * new_bboxes (`np.ndarray`): bboxes after affine, [N, 4].
    """
    n, num = segments.shape[:2]
    if n == 0:
        return [], segments

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

def low_res_inpaint(
        img : np.ndarray, 
        mask : np.ndarray, 
        scale : int=6
    ) -> np.ndarray:
    """
    Performs inpainting on a low-resolution version of the image, and then copies the upsampled inpainted image back into the original image.
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
        img : np.ndarray, 
        polys : List[np.ndarray], 
        exclude_polys : Optional[List[np.ndarray]]=None, 
        downscale_factor : Union[int, float]=6, 
        **kwargs
    ) -> np.ndarray:
    """
    Mutably inpaints the polygons in an image using the Fast Marching method by Alexandru Telea.

    The inpainting algorithm is performed on a downsampled version of the image to speed up the process, and the inpainted results are then upsampled and pasted back into the original image.
    
    Args:
        img (`np.ndarray`): The image to inpaint.
        polys (`List[np.ndarray]`): A list of polygons to inpaint.
        exclude_polys (`Optional[List[np.ndarray]]`, optional): A list of polygons to exclude from inpainting. Defaults to None.
        downscale_factor (`Union[int, float]`, optional): The factor by which to downscale the image before inpainting. Defaults to 6.
        **kwargs: Additional keyword arguments to pass to `cv2.drawContours`.

    Returns:
        out (`np.ndarray`): The inpainted image.
    """
    # Type checking and sanitizing
    check_types(img, np.ndarray)
    if not ((img.ndim == 3 and img.shape[2] < 5) or img.ndim == 2): 
        raise ValueError(f"img must be a 2D or 3D numpy array, of shape (H, W) or (H, W, C), got shape {img.shape}")
    check_types(polys, [list, np.ndarray])
    check_types(exclude_polys, ([list, np.ndarray], None))
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
    
    # Dilate the inpaint bitmap to ensure that the inpainting doesn't bleed from the edges of the instances under the polygons
    cv2.dilate(src=inpaint_bitmap, dst=inpaint_bitmap, kernel=np.ones((3, 3), np.uint8), iterations=1)
    
    # Inpaint the low-res image using the Fast Marching algorithm
    cv2.inpaint(
        src=lr_img,
        dst=lr_img,
        inpaintMask=inpaint_bitmap,
        inpaintRadius=5,
        flags=cv2.INPAINT_TELEA
    )

    # Remove the exclude polygons from the inpaint bitmap, so that the original image is not inpainted under them
    for p in exclude_polys:
        inpaint_bitmap = cv2.drawContours(
            inpaint_bitmap,
            [p // downscale_factor],
            color=0,
            **kwargs
        )

    # Upsample the inpainted image and bitmap
    inpaint_bitmap = cv2.resize(inpaint_bitmap, orig_shape)
    lr_img = cv2.resize(lr_img, orig_shape)
    
    # Copy the inpainted low-res image back into the original image
    img[inpaint_bitmap == 1] = lr_img[inpaint_bitmap == 1]

    # Return the inpainted image (not necessary, as the inpainting is done in-place)
    return img

def inpaint_pad(
        array : Union[torch.Tensor, np.ndarray], 
        padding : Union[int, Tuple[int, int], Tuple[int, int, int, int]]
    ) -> Union[torch.Tensor, np.ndarray]:
    # Ensure padding is a tuple (pad_top, pad_bottom, pad_left, pad_right)
    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        padding = list(padding)
        padding = tuple(padding + padding)
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
    padded_image[h_slice, w_slice] = array # <-- HERE
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

class InpaintPad:
    def __init__(self, padding : Union[int, Tuple[int, int], Tuple[int, int, int, int]]):
        self.padding = padding

    def __call__(self, tensor : torch.Tensor) -> torch.Tensor:
        return inpaint_pad(tensor, self.padding)
    
def remove_instances(
        labels : Dict, 
        area_thr : Union[float, int]=1, 
        max_targets : Optional[int]=1000, 
        min_size : int=0
    ) -> Dict:
    instances : Instances = labels.pop("instances")
    imsize = labels["img"].shape[:2][::-1]

    if instances.normalized:
        instances.denormalize(*imsize)
    if instances._bboxes.format != "xywh":
        instances.convert_bbox(format="xywh")

    bboxes = instances._bboxes.bboxes

    if bboxes.shape[0] == 0:
        labels["instances"] = Instances(np.empty([0, 4], dtype=np.float32), np.empty([0, 2], dtype=np.float32),
                                        normalized=False)
        labels["cls"] = np.empty((0), dtype=np.int32)
        return labels

    # Instead of using bounding boxes, we calculate the proportion of the segment area that is within the image
    eps = 1e-9
    image_bbox = box(0, 0, *imsize)
    area_ratios = np.zeros(bboxes.shape[0])
    for i, s in enumerate(instances.segments):
        # Initiate overlap using bounding box
        x, y, w, h = bboxes[i]
        bbox = box(x - w/2, y - h/2, x + w/2, y + h/2)
        bbox_iarea = bbox.intersection(image_bbox).area
        area_ratios[i] = (bbox_iarea + eps) / (bbox.area + eps) if bbox.area > 0 and bbox_iarea > 0 else 0
        if area_ratios[i] < area_thr:
            continue
        poly = make_valid(Polygon(s))
        intersection_area = poly.intersection(image_bbox).area
        area_ratios[i] = (intersection_area + eps) / (poly.area + eps) if poly.area > 0 and intersection_area > 0 else 0

    valid = np.all(
        [
            #(b[:, 0] - b[:, 2] / 2) / self._imsize > 0,
            #(b[:, 1] - b[:, 3] / 2) / self._imsize > 0,
            #(b[:, 0] + b[:, 2] / 2) / self._imsize < 1,
            #(b[:, 1] + b[:, 3] / 2) / self._imsize < 1,
            area_ratios >= area_thr,
            bboxes[:, 2] > min_size,
            bboxes[:, 3] > min_size
        ],
       axis=0
    )

    if max_targets is not None and np.sum(valid) > max_targets:
        n_remove, n_keep = np.sum(valid) - max_targets, max_targets
        valid[valid] &= np.random.permutation(np.array([True] * n_keep + [False] * n_remove, dtype=bool))

    # here, we paint the edge cases (partially outside the image, using telea inpainting),
    # this should help learning. Indeed it would be very confusing if an image if an insect that is
    # 10% outside is flagged as NOT insect!

    invalid = np.bitwise_not(valid)
    invalid_visible = np.bitwise_and(invalid, area_ratios > 0) # We only need to inpaint polygons within the frame

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
            offset=(0, 0)
        )

    # cv2.imwrite(f"/tmp/{os.path.basename(labels['im_file'])}", or_img)
    valid_i = np.nonzero(valid)[0]

    if len(valid_i) == 0:
        labels["instances"] = Instances(np.empty([0, 4], dtype=np.float32), np.empty([0, 2], dtype=np.float32),
                                        normalized=False)
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

def scale_labels(
        labels : Dict, 
        scale : float
    ) -> Dict:
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

class FlatBugRandomPerspective(RandomPerspective):
    fill_value = (0, 0, 0)

    def __init__(self, imgsz : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imgsz = imgsz

    def affine_transform(
            self : Self, 
            img : np.ndarray, 
            border : Tuple[int, int]
        ) -> Tuple[np.ndarray, np.ndarray, float]:
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
    
    def __call__(self, labels : dict):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        # labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls = labels["cls"]
        instances : Instances = labels.pop("instances")
        # Make sure the coord formats are right
        if instances._bboxes.format != "xyxy":
            instances.convert_bbox(format="xyxy")
        if instances.normalized:
            instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
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
    bg_fill = (0, 0, 0)
    min_size = 0  # px

    def __init__(self, imsize : Union[int, Tuple[int, int], List[int], np.ndarray]):
        if isinstance(imsize, int):
            self._imsize = (imsize, imsize)
        elif isinstance(imsize, (tuple, list, np.ndarray)):
            if len(imsize) == 1:
                self._imsize = (imsize[0], imsize[0])
            elif len(imsize) != 2:
                raise ValueError("imsize should be a list of length 2")
            self._imsize = imsize
        else:
            raise TypeError(f'`imsize` should be of type `int`, `tuple`, or `list`, got {type(imsize)}')
        self._imsize = tuple([int(i) for i in self._imsize])
        self.xsize, self.ysize = self._imsize

    def crop_image(
            self : Self, 
            labels : Dict, 
            start_x : int, 
            start_y : int, 
            size_x : int, 
            size_y : int
        ) -> Dict:
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

        img = img[n_start_y: n_start_y + n_size_y, n_start_x: n_start_x + n_size_x, :]

        if px > 0 or py > 0:
            img = np.pad(img, pad_width=((py0, py1), (px0, px1), (0, 0)), mode="constant", constant_values=0.)
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

        instances : Instances = labels.pop("instances")
        if instances._bboxes.format != "xywh":
            instances.convert_bbox(format="xywh")
        if instances.normalized:
            instances.denormalize(*orig_shape[:2][::-1])

        labels['ratio_pad'] = ((1.0, 1.0), (0.0, 0.0))
        x_offset = -n_start_x + px0
        y_offset = -n_start_y + py0

        # positions in the cropped image
        instances._bboxes.add([x_offset, y_offset, 0, 0])

        for s in instances.segments:
            s[:, 0] += x_offset
            s[:, 1] += y_offset

        labels["instances"] = instances

        return labels
    
    def __call__(self, x):
        raise NotImplementedError("This method should be implemented in a subclass")

class CenterCrop(Crop):
    def __call__(self, labels : Dict) -> Dict:
        h, w = labels["img"].shape[:2]

        start_x = (w - self.xsize) // 2
        start_y = (h - self.ysize) // 2

        return self.crop_image(labels, start_x, start_y, self.xsize, self.ysize)

class RandomCrop(Crop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, labels : Dict) -> Dict:
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
            scale = np.random.uniform(1, min_target_source_ratio) ** (1/2)

        # When we scale up, this is done before cropping
        do_scale_before = scale > 1
        if do_scale_before:
            labels = scale_labels(labels, scale)
            target_xsize, target_ysize = self.xsize, self.ysize
        else:
            target_size = max(int(w * scale), int(h * scale))
            target_xsize, target_ysize = target_size, target_size
            # Reset the scale such that when the labels/image are scaled after cropping the size is self.xsize, self.ysize (assuming these are equal)
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

class FixInstances:
    def __init__(
            self, 
            area_thr : Union[float, int], 
            max_targets : Optional[int], 
            min_size : int
        ):
        """"
        A callable class that removes instances that are too small or which overlap less than a certain threshold with the image.

        Args:
            area_thr (`Union[float, int]`): The minimum proportion of the instance that must be within the image in order for it to be kept.
            max_targets (`Optional[int]`): The maximum number of instances to keep. If there are more instances than this, a random subset of instances will be kept. If `None`, all instances will be kept.
            min_size (`int`): The minimum size of the bounding box of the instance. Instances with a width or height less than this value will be removed.
        """
        self.area_thr = area_thr
        self.max_targets = max_targets if max_targets is None or max_targets > 0 else None
        self.min_size = min_size
    
    def __call__(self, labels : dict) -> dict:
        """
        Performs instance fixing.

        Args:
            labels (`dict`): Dictionary containing the instances.
        Returns:
            out (`dict`): A dictionary containing the updated instances.
        """
        return remove_instances(labels, area_thr=self.area_thr, max_targets=self.max_targets, min_size=self.min_size)

class RandomColorInv(object):
    def __init__(self, p : float=0.5):
        """
        Invert the colors of an image with a probability p.

        Args:
            p (`float`, optional): probability of inverting the colors. Defaults to 0.5
        """
        if p < 0:
            logger.warning("p should be in [0,1], got", p, "setting to 0")
            p = 0
        if p > 1:
            logger.warning("p should be in [0,1], got", p, "setting to 1")
            p = 1
        self.p = 1 - p

    def __call__(self, labels : Dict) -> Dict:
        img = labels['img']
        if random.uniform(0, 1) > self.p:
            assert img.dtype == np.uint8
            labels['img'] = 255 - img
        return labels
