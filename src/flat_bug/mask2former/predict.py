"""Tiled inference for the flatbug Mask2Former prototype.

Runs a fine-tuned ``Mask2FormerForUniversalSegmentation`` over overlapping tiles
of an arbitrarily large image and returns a `flat_bug.predictor.TensorPredictions`
— the same result object the YOLO pipeline produces. That buys plotting, crop
extraction and JSON serialization for free, and lets the predictions be scored
with ``fb_evaluate`` without any format conversion.

Usage:
    fb_predict_m2f -i /path/to/images -o runs/m2f/preds -w runs/m2f/best.pt

The per-image metadata this writes is the same JSON the YOLO pipeline writes, so a
run can be scored against COCO ground truth without any conversion:

    fb_evaluate -p 'runs/m2f/preds/**/metadata_*.json' -g gt.json -I /path/to/images -o runs/m2f/eval --combine

The run directory also gets a compiled ``coco_instances.json``, the same name and
format the YOLO pipeline writes, so ``scripts/eval/end_to_end_eval.sh``'s evaluation
half (``fb_evaluate -c`` plus ``eval-metrics.R``) can consume it unchanged.

Like the YOLO predictor it runs a scale pyramid by default (``--single-scale``
disables it), so the tile-and-stitch scheme matches and the two can be compared.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import uuid

import torch
import torch.nn.functional as F  # noqa: N812
from torchvision.io import ImageReadMode, decode_image
from tqdm.auto import tqdm

from flat_bug import logger, set_log_level
from flat_bug.coco_utils import fb_to_coco
from flat_bug.config import DEFAULT_CFG, read_cfg
from flat_bug.geometric import calculate_tile_offsets, find_contours
from flat_bug.mask2former.data import IMAGENET_MEAN, IMAGENET_STD
from flat_bug.mask2former.train import DEFAULT_CHECKPOINT, build_model
from flat_bug.predictor import TensorPredictions
from flat_bug.predictor import _executor as prediction_executor

DEFAULT_IMAGE_PATTERN = r"[^/]*\.([jJ][pP][eE]?[gG]|[pP][nN][gG])$"


class M2FPredictor:
    """Tiled, single-scale Mask2Former predictor producing `TensorPredictions`."""

    def __init__(
        self,
        model,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        tile_size: int = 512,
        minimum_tile_overlap: int | None = None,
        batch_size: int = 4,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_threshold: float = 0.2,
        overlap_metric: str = "IoU",
        min_max_obj_size: tuple[int, int] = (32, 10**8),
        edge_case_margin: int = 16,
    ):
        """Wrap an already-built Mask2Former model.

        Args:
            model: A `Mask2FormerForUniversalSegmentation` instance.
            device: Device to run inference on.
            dtype: Dtype to run inference in. ``float16`` is only sensible on CUDA.
            tile_size: Side length of the square tiles fed to the model. Should match
                the ``--image-size`` used during training.
            minimum_tile_overlap: Minimum overlap between neighbouring tiles.
                Defaults to half the tile size.
            batch_size: Number of tiles per forward pass.
            score_threshold: Minimum (class x mask) score for a query to be kept.
            mask_threshold: Sigmoid threshold used to binarize the mask logits.
            overlap_threshold: Overlap above which two detections are considered duplicates.
            overlap_metric: ``"IoU"`` or ``"IoS"``.
            min_max_obj_size: Valid range for the square-root of the bounding box area.
            edge_case_margin: Detections closer than this to a tile border are dropped;
                the image is padded beforehand so that real image-border objects survive.

        """
        self.device = torch.device(device)
        self.dtype = dtype
        self._model = model.to(device=self.device, dtype=self.dtype).eval()
        self.tile_size = tile_size
        if minimum_tile_overlap is None:
            # The config's overlap is in pixels and tuned for the YOLO TILE_SIZE. A
            # Mask2Former tile must match the resolution the model was trained at, so the
            # overlap is rescaled to keep the same *fraction* of a tile rather than the
            # same pixel count - using 384 px on a 512 tile would mean a stride of 128.
            minimum_tile_overlap = round(DEFAULT_CFG["MINIMUM_TILE_OVERLAP"] * tile_size / DEFAULT_CFG["TILE_SIZE"])
        if minimum_tile_overlap >= tile_size:
            logger.warning(
                f"minimum_tile_overlap ({minimum_tile_overlap}) >= tile_size ({tile_size}); "
                f"falling back to {tile_size // 2}"
            )
            minimum_tile_overlap = tile_size // 2
        self.minimum_tile_overlap = minimum_tile_overlap
        self.batch_size = batch_size
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.overlap_threshold = overlap_threshold
        self.overlap_metric = overlap_metric
        self.min_max_obj_size = tuple(min_max_obj_size)
        self.edge_case_margin = edge_case_margin

    @classmethod
    def from_checkpoint(
        cls,
        weights: str,
        base_checkpoint: str | None = None,
        num_classes: int | None = None,
        tile_size: int | None = None,
        **kwargs,
    ) -> M2FPredictor:
        """Build a predictor from a checkpoint written by `flat_bug.mask2former.train`.

        The architecture metadata (base HF checkpoint, number of classes, training
        image size) is read from the checkpoint when present, and falls back to the
        prototype defaults for checkpoints written before that metadata was stored.
        """
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            # A bare state dict, i.e. no architecture metadata to recover.
            ckpt = {"model": ckpt}
        state = ckpt["model"]
        if base_checkpoint is None:
            base_checkpoint = ckpt.get("base_checkpoint", DEFAULT_CHECKPOINT)
        if num_classes is None:
            num_classes = ckpt.get("num_classes", 1)
        if tile_size is None:
            tile_size = ckpt.get("image_size", 512)
        logger.info(f"Building {base_checkpoint} with {num_classes} class(es), tile size {tile_size}")
        model = build_model(num_classes=num_classes, checkpoint=base_checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(f"load_state_dict: {len(missing)} missing and {len(unexpected)} unexpected keys")
        return cls(model, tile_size=tile_size, **kwargs)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        """uint8/float CHW image -> ImageNet-normalized float tensor on the target device."""
        image = image.to(device=self.device)
        if not image.is_floating_point():
            image = image.float() / 255.0
        else:
            image = image.float()
        image = (image - IMAGENET_MEAN.to(self.device)) / IMAGENET_STD.to(self.device)
        return image.to(self.dtype)

    def _decode_tile(
        self, class_logits: torch.Tensor, mask_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Turn the per-query logits of a single tile into binary masks, scores and classes.

        Args:
            class_logits: ``(n_queries, n_classes + 1)`` — the last column is "no object".
            mask_logits: ``(n_queries, tile_size / 4, tile_size / 4)``.

        Returns:
            ``(masks, scores, classes)`` with ``masks`` of shape ``(n, tile_size, tile_size)``.

        """
        probs = class_logits.float().softmax(dim=-1)[:, :-1]
        class_scores, classes = probs.max(dim=-1)
        keep = class_scores > self.score_threshold
        if not bool(keep.any()):
            return (
                torch.zeros((0, self.tile_size, self.tile_size), dtype=torch.bool, device=class_logits.device),
                torch.zeros((0,)),
                torch.zeros((0,), dtype=torch.long),
            )

        masks = F.interpolate(
            mask_logits[keep].float().unsqueeze(0),
            size=(self.tile_size, self.tile_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).sigmoid()
        binary = masks > self.mask_threshold
        areas = binary.flatten(1).sum(dim=1)

        # Mask2Former's mask-aware score: how confident the model is *inside* the mask it drew.
        mask_scores = (masks * binary).flatten(1).sum(dim=1) / areas.clamp(min=1)
        scores = class_scores[keep] * mask_scores

        valid = (areas > 0) & (scores > self.score_threshold)
        return binary[valid], scores[valid].cpu(), classes[keep][valid].cpu()

    @torch.no_grad()
    def _detect(
        self, image: torch.Tensor, margin: int | None = None
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tile, forward, decode and collect detections in (padded) image coordinates."""
        h, w = image.shape[1:]
        offsets = calculate_tile_offsets(
            image_size=(w, h), tile_size=self.tile_size, minimum_overlap=self.minimum_tile_overlap
        )
        margin = self.edge_case_margin if margin is None else margin
        size = self.tile_size

        contours: list[torch.Tensor] = []
        boxes: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        classes: list[torch.Tensor] = []

        for start in range(0, len(offsets), self.batch_size):
            chunk = offsets[start : start + self.batch_size]
            batch = torch.stack([image[:, y : y + size, x : x + size] for _, (y, x) in chunk])
            outputs = self._model(pixel_values=batch)
            for i, (_, (y_off, x_off)) in enumerate(chunk):
                masks, tile_scores, tile_classes = self._decode_tile(
                    outputs.class_queries_logits[i], outputs.masks_queries_logits[i]
                )
                for mask, score, class_ in zip(masks, tile_scores, tile_classes):
                    contour = find_contours(mask, largest_only=True, simplify=True)
                    if len(contour) < 3:
                        continue
                    x1, y1 = contour.min(dim=0).values.tolist()
                    x2, y2 = contour.max(dim=0).values.tolist()
                    # Drop detections hugging a tile border - the image is padded so that
                    # objects on the real image border are never near the outermost tile borders.
                    if x1 < margin or y1 < margin or x2 > (size - margin) or y2 > (size - margin):
                        continue
                    offset = torch.tensor([x_off, y_off], dtype=contour.dtype, device=contour.device)
                    contours.append(contour + offset)
                    boxes.append(torch.tensor([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off], dtype=torch.float32))
                    scores.append(score)
                    classes.append(class_)

        if not contours:
            return [], torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
        return (
            [c.cpu() for c in contours],
            torch.stack(boxes),
            torch.stack(scores).float(),
            torch.stack(classes).long(),
        )

    def scales(self, max_dimension: int, single_scale: bool = False, scale_increment: float = 2 / 3) -> list[float]:
        """The scale ladder, mirroring `flat_bug.predictor.Predictor.pyramid_predictions`.

        The coarsest scale fits the whole image into one tile; the finest is native
        resolution. Detections from every scale are stitched together and de-duplicated.
        """
        if single_scale:
            return [1.0]
        scale = self.tile_size / max_dimension
        if scale >= 1:
            return [scale]
        ladder = []
        while scale <= 0.9:  # cut off near 1 so we do not run a near-native scale twice
            ladder.append(scale)
            scale /= scale_increment
        if scale != 1:
            ladder.append(1.0)
        return ladder

    def _detect_at_scale(
        self, normalized: torch.Tensor, scale: float, max_scale: bool, size: tuple[int, int]
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tile and stitch at one scale, returning detections in original image coordinates."""
        h, w = size
        if scale != 1:
            normalized = F.interpolate(
                normalized.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False, antialias=True
            ).squeeze(0)

        # At the coarsest scale nothing above it can catch border objects, so - as in the
        # YOLO pyramid - the edge filter is switched off and large objects are kept.
        margin = 0 if max_scale else self.edge_case_margin
        pad = margin * 2
        scaled_h, scaled_w = normalized.shape[1:]
        pad_right = max(0, self.tile_size - (scaled_w + 2 * pad))
        pad_bottom = max(0, self.tile_size - (scaled_h + 2 * pad))
        normalized = F.pad(normalized, (pad, pad + pad_right, pad, pad + pad_bottom), mode="constant", value=0)

        contours, boxes, confs, classes = self._detect(normalized, margin=margin)
        if not contours:
            return [], torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)

        # Undo the padding and the scaling, back into the source image's coordinates.
        lower, upper = torch.tensor([0, 0]), torch.tensor([w - 1, h - 1])
        contours = [(((c - pad).float() / scale).round().long()).clamp(min=lower, max=upper) for c in contours]
        boxes = (boxes - pad) / scale
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        return contours, boxes, confs, classes

    def __call__(
        self,
        image: torch.Tensor | str,
        path: str | None = None,
        single_scale: bool = False,
        scale_increment: float = 2 / 3,
    ) -> TensorPredictions:
        """Detect instances in a single image by tiling, stitching and de-duplicating.

        The image is tiled at every scale of the pyramid, each tile is passed through the
        model, the per-tile masks are mapped back into full-image coordinates, and the
        union is reduced by NMS - the same tile-and-stitch scheme the YOLO predictor uses,
        so the stitched result is what gets compared against ground truth.

        Args:
            image: A ``(3, H, W)`` tensor or a path to an image file.
            path: Source path, required when ``image`` is a tensor and the results
                are to be saved under the image's basename.
            single_scale: Run at native resolution only, skipping the pyramid.
            scale_increment: Ratio between successive scales.

        Returns:
            The stitched, NMS-ed predictions, in the coordinate space of the input image.

        """
        source = image if isinstance(image, str) else path
        if isinstance(image, str):
            image = decode_image(input=image, mode=ImageReadMode.RGB, apply_exif_orientation=True)
        elif not isinstance(image, torch.Tensor):
            raise TypeError(f"Unknown type for image: {type(image)}, expected str or torch.Tensor")
        original = image.cpu()
        h, w = original.shape[1:]

        normalized = self._normalize(original)
        ladder = self.scales(max(h, w), single_scale=single_scale, scale_increment=scale_increment)
        coarsest = min(ladder)
        logger.debug(f"scales: {[round(s, 3) for s in ladder]}")

        contours: list[torch.Tensor] = []
        boxes_per_scale, confs_per_scale, classes_per_scale = [], [], []
        for scale in reversed(ladder):
            is_coarsest = scale == coarsest
            found, boxes, confs, classes = self._detect_at_scale(normalized, scale, is_coarsest, (h, w))
            # Instances larger than the size range are only kept at the coarsest scale,
            # where nothing above can detect them - as in the YOLO pyramid.
            largest = float("inf") if is_coarsest else self.min_max_obj_size[1]
            sizes = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=0).sqrt()
            keep = (sizes >= self.min_max_obj_size[0]) & (sizes <= largest)

            contours.extend(c for c, k in zip(found, keep.tolist()) if k)
            boxes_per_scale.append(boxes[keep])
            confs_per_scale.append(confs[keep])
            classes_per_scale.append(classes[keep])

        boxes = torch.cat(boxes_per_scale) if boxes_per_scale else torch.zeros((0, 4))
        confs = torch.cat(confs_per_scale) if confs_per_scale else torch.zeros((0,))
        classes = torch.cat(classes_per_scale) if classes_per_scale else torch.zeros((0,), dtype=torch.long)

        predictions = TensorPredictions(image=original, image_path=source)
        predictions.PREFER_POLYGONS = True
        predictions.mask_width, predictions.mask_height = w, h
        if contours:
            predictions.contours = [c.numpy() for c in contours]
            predictions.boxes = boxes.to(predictions.dtype)
            predictions.confs = confs.to(predictions.dtype)
            predictions.classes = classes.to(predictions.dtype)
            predictions.scales = [1.0] * len(contours)
            predictions = predictions.non_max_suppression(
                overlap_threshold=self.overlap_threshold, metric=self.overlap_metric
            )
        return predictions


def _list_images(input: str, pattern: str, recursive: bool, max_images: int | None) -> list[str]:
    """Resolve a file or a directory into a sorted list of image paths."""
    if os.path.isfile(input):
        return [input]
    if not os.path.isdir(input):
        raise FileNotFoundError(f"'{input}' is neither a file nor a directory.")
    pattern_path = os.path.join(input, "**", "*") if recursive else os.path.join(input, "*")
    candidates = glob.glob(pattern_path, recursive=recursive)
    matcher = re.compile(pattern)
    files = sorted(f for f in candidates if os.path.isfile(f) and matcher.search(f.replace(os.sep, "/")))
    if not files:
        raise FileNotFoundError(f"No images matching '{pattern}' found in {input}")
    return files[:max_images] if max_images else files


def cli_args() -> dict:  # noqa: D103
    parser = argparse.ArgumentParser(
        prog="fb_predict_m2f",
        description="Tiled Mask2Former inference for flatbug (prototype).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="An image file or a directory of image files"
    )
    parser.add_argument(
        "-o", "--output", dest="output_dir", required=True,
        help="The result directory"
    )
    parser.add_argument(
        "-w", "--model-weights", dest="model_weights", required=True,
        help="A checkpoint written by fb_train_m2f"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Base HuggingFace checkpoint. Default: read from the weights file."
    )
    parser.add_argument(
        "-p", "--input-pattern", dest="input_pattern", default=DEFAULT_IMAGE_PATTERN,
        help="The pattern to match the images."
    )
    parser.add_argument(
        "-n", "--max-images", dest="max_images", type=int, default=None,
        help="Maximum number of images to process. Truncates in alphabetical order."
    )
    parser.add_argument(
        "-R", "--recursive", action="store_true",
        help="Process images nested within subdirectories of the input."
    )
    parser.add_argument(
        "-g", "--device", "--gpu", dest="device", default="auto",
        help="Which device to use for inference."
    )
    parser.add_argument(
        "-d", "--dtype", default=None,
        help="Which dtype to use for inference. Default is 'float16' for CUDA and 'float32' for CPU."
    )
    parser.add_argument(
        "--tile-size", dest="tile_size", type=int, default=None,
        help="Tile side length. Default: the image size the checkpoint was trained at."
    )
    parser.add_argument(
        "--tile-overlap", dest="tile_overlap", type=int, default=None,
        help="Minimum overlap between tiles. Default: the config overlap, rescaled to the tile size."
    )
    parser.add_argument(
        "--batch", type=int, default=4,
        help="Number of tiles per forward pass."
    )
    parser.add_argument(
        "--single-scale", action="store_true",
        help="Run at native resolution only, skipping the scale pyramid."
    )
    parser.add_argument(
        "--scale-increment", dest="scale_increment", type=float, default=2 / 3,
        help="Ratio between successive scales of the pyramid."
    )
    parser.add_argument(
        "--mask-threshold", dest="mask_threshold", type=float, default=0.5,
        help="Sigmoid threshold used to binarize the predicted masks."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="The config file (score/overlap thresholds and object size range)."
    )
    parser.add_argument(
        "-M", "--nms-metric", dest="nms_metric", type=str, default=None,
        help="Overlap metric to use for NMS, overrides the config. One of 'IoU' or 'IoS'."
    )
    parser.add_argument(
        "--id", type=str, default=None,
        help="Identifier (ID) for prediction run."
    )
    parser.add_argument(
        "--no-crops", action="store_true", help="Do not save the crops."
    )
    parser.add_argument(
        "--no-overviews", action="store_true", help="Do not save the overviews."
    )
    parser.add_argument(
        "--no-metadata", action="store_true", help="Do not save the metadata."
    )
    parser.add_argument(
        "-C", "--no-compiled-coco", dest="no_compiled_coco", action="store_true",
        help="Skip the production of a compiled COCO file (for all images)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose mode."
    )
    return vars(parser.parse_args())


def predict(
    input: str,
    output_dir: str,
    model_weights: str,
    checkpoint: str | None = None,
    input_pattern: str = DEFAULT_IMAGE_PATTERN,
    max_images: int | None = None,
    recursive: bool = False,
    device: str = "auto",
    dtype: str | None = None,
    tile_size: int | None = None,
    tile_overlap: int | None = None,
    batch: int = 4,
    single_scale: bool = False,
    scale_increment: float = 2 / 3,
    mask_threshold: float = 0.5,
    config: str | None = None,
    nms_metric: str | None = None,
    id: str | None = None,
    no_crops: bool = False,
    no_overviews: bool = False,
    no_metadata: bool = False,
    no_compiled_coco: bool = False,
    verbose: bool = False,
) -> None:
    """Run tiled Mask2Former inference over one or more images and save the results."""
    if verbose:
        set_log_level("DEBUG")

    cfg = read_cfg(os.path.normpath(config)) if config is not None else dict(DEFAULT_CFG)
    if nms_metric is not None:
        cfg["OVERLAP_METRIC"] = nms_metric

    if device is None or device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if "cuda" in device and not torch.cuda.is_available():
        raise ValueError(f"Device '{device}' is not available.")
    if dtype is None:
        dtype = "float16" if "cuda" in device else "float32"
    torch_dtype = getattr(torch, dtype)

    files = _list_images(os.path.normpath(input), input_pattern, recursive, max_images)
    output_dir = os.path.normpath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if id is None:
        id = str(uuid.uuid4())

    predictor = M2FPredictor.from_checkpoint(
        os.path.normpath(model_weights),
        base_checkpoint=checkpoint,
        tile_size=tile_size,
        device=device,
        dtype=torch_dtype,
        minimum_tile_overlap=tile_overlap,
        batch_size=batch,
        score_threshold=cfg["SCORE_THRESHOLD"],
        mask_threshold=mask_threshold,
        overlap_threshold=cfg["OVERLAP_THRESHOLD"],
        overlap_metric=cfg["OVERLAP_METRIC"],
        min_max_obj_size=cfg["MIN_MAX_OBJ_SIZE"],
        edge_case_margin=cfg["EDGE_CASE_MARGIN"],
    )

    coco: dict = {}
    for file in tqdm(files, desc="Predicting", dynamic_ncols=True):
        predictions = predictor(file, single_scale=single_scale, scale_increment=scale_increment)
        logger.debug(f"{os.path.basename(file)}: {len(predictions)} instances")
        predictions.save(
            output_dir,
            overview=not no_overviews,
            crops=not no_crops,
            metadata=not no_metadata,
            identifier=id,
        )
        if not no_compiled_coco:
            fb_to_coco(predictions.json_data, coco)

    prediction_executor.flush(progress=True)

    if not no_compiled_coco:
        # Same name the YOLO pipeline writes, so `scripts/eval/end_to_end_eval.sh`
        # can point at this run without modification.
        with open(os.path.join(output_dir, "coco_instances.json"), "w") as f:
            json.dump(coco, f)


def main() -> None:  # noqa: D103
    predict(**cli_args())


if __name__ == "__main__":
    main()
