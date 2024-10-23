import math
import os
import glob
import json

from typing import List, Tuple, Optional

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import pandas as pd
import numpy as np
import cv2

import torch

from flat_bug.predictor import Predictor, TensorPredictions
from flat_bug.coco_utils import annotations_to_numpy, split_annotations

## STATICS

ROI_SIZE = 1000
ROI_PADDING = 500
MOSAIC_SPACING = 0
MODEL = "model_snapshots/fb_large_2024-10-18_best.pt"

# ANNOTATED_IM_DIR = "dev/fb_yolo/insects"
ANNOTATION_COCO = "dev/fb_yolo/insects/labels/val/instances_default.json"
RAW_IM_DIR = "dev/fb_yolo/insects"


THIS_DIR = os.path.dirname(__file__)

OUR_ANNOT = os.path.join(THIS_DIR, "annotated_tiles")
OUR_RAW = os.path.join(THIS_DIR, "raw_tiles")
OUR_PRED = os.path.join(THIS_DIR, "pred_tiles")
OUR_MOSAIC = os.path.join(THIS_DIR, "mosaic")

## HELPERS

def match_examples(files : List[str], *filters):
    matches = []
    for this_filters in zip(*filters):
        for file in files:
            if all([filter in os.path.basename(file) for filter in this_filters]):
                matches.append(file)
    return matches

def annotations_to_tensor_predictions(annotations : List[np.ndarray], offset : Tuple[int, int], image : torch.tensor, path : str) -> TensorPredictions:
    c, h, w = image.shape
    if h != w:
        raise NotImplementedError("Plotting annotations is only implemented for square images. TODO: implement this.")
    im_size = h
    
    n_instances = len(annotations[0])

    # Offset boxes and switch x and y
    annotations[0][:, ::2] += offset[0]
    annotations[0][:, 1::2] += offset[1]
    annotations[0][:, :2] = annotations[0][:, :2][:, ::-1]
    annotations[0][:, 2:] = annotations[0][:, 2:][:, ::-1]

    box_oob = ~(np.any(annotations[0] > 0, 1) |  np.any(annotations[0] < im_size, 1))
    annotations[0] = annotations[0][~box_oob] 

    # Offset contours and switch x and y
    offset = np.array(offset, dtype=np.int32)
    for i in range(n_instances):
        if box_oob[i]:
            continue
        new_coords = np.flip(annotations[1][i] + offset, 1)
        annotations[1][i] = new_coords
    
    annotations[1] = [c for oob, c in zip(box_oob, annotations[1]) if not oob]

    tensor_predictions_data = {
        "boxes": annotations[0],
        "contours": [torch.tensor(c.copy(order="C")) for c in annotations[1]],
        "confs": [1 for _ in range(n_instances)],
        "classes": [1 for _ in range(n_instances)],
        "scales": [-1 for _ in range(n_instances)],
        "image_path": path,
        "image_width": w,
        "image_height": h,
        "mask_width": w,
        "mask_height": h,
        "identifier": None
    }

    tensor_predictions = TensorPredictions().load(tensor_predictions_data)
    tensor_predictions.image = image
    return tensor_predictions

def create_mosaic(ims : List[np.ndarray], spacing : int = 100, labels : Optional[List[str]]=None):
    n = len(ims)
    h, w, _ = ims[0].shape

    nrow = math.floor(n ** (1/2))
    ncol = math.ceil(n / nrow)

    mosaic = np.zeros(
        (
            h * nrow + spacing * (nrow - 1),
            w * ncol + spacing * (ncol - 1),
            3
        ),
        dtype = ims[0].dtype
    ) + 255

    def get_slice(ri : int, ci : int):
        ri, ci = ri + 1, ci + 1
        ymax = h * ri + spacing * (ri - 1)
        xmax = w * ci + spacing * (ci - 1)
        ymin = ymax - h
        xmin = xmax - w
        return np.s_[ymin:ymax, xmin:xmax, :]
    
    def create_label(ri : int, ci : int, label : str):
        lab = np.zeros((70, 150, 3), dtype=ims[0].dtype) + 255
        cv2.putText(
            lab, 
            label, 
            org=(5, 60), 
            fontFace=cv2.FONT_HERSHEY_COMPLEX, 
            fontScale=2, 
            thickness=3,
            color=(0, 0, 0)
        )
        return lab

    im_i = 0
    for r in range(nrow):
        for c in range(ncol):
            if im_i >= n:
                break
            im = ims[im_i].copy()
            if labels:
                im[:70, :150] = create_label(r, c, labels[im_i])
            mosaic[get_slice(r, c)] = im
            im_i += 1

    return mosaic

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Predictor(MODEL, cfg="dev/test/tmp.yaml", device=device, dtype=torch.float16)

    example_df = pd.read_csv(os.path.join(THIS_DIR, "clean_flatbug_datasets.csv"))
    with open(ANNOTATION_COCO, "r") as f:
        annotations = split_annotations(json.load(f))

    example_df["raw_file_path"] = match_examples(
        glob.glob(os.path.join(RAW_IM_DIR, "**", "*.jpg"), recursive=True),
        example_df.dataset,
        example_df._example_name    
    )
    example_df["annotations"] = [list(annotations_to_numpy(annotations[os.path.basename(file)])) for file in example_df.raw_file_path]

    os.makedirs(OUR_ANNOT, exist_ok=True)
    os.makedirs(OUR_RAW, exist_ok=True)
    os.makedirs(OUR_PRED, exist_ok=True)
    os.makedirs(OUR_MOSAIC, exist_ok=True)

    rois = {tp : [] for tp in ["annotation", "raw", "prediction"]}

    def preprocess_one_roi(row):
        _, row = row
        im = cv2.imread(row.raw_file_path)

        h, w = im.shape[0:2]

        # Correct padding calculations
        top = max(0, ROI_PADDING - row._example_y)
        bottom = max(0, (row._example_y + ROI_SIZE + ROI_PADDING) - h)
        left = max(0, ROI_PADDING - row._example_x)
        right = max(0, (row._example_x + ROI_SIZE + ROI_PADDING) - w)

        # Apply padding if needed
        if left > 0 or top > 0 or right > 0 or bottom > 0:
            im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Calculate offsets (with original X/Y coordinates to map back)
        offset = -(row._example_x - ROI_PADDING), -(row._example_y - ROI_PADDING)

        # Extract ROI using correct indices
        roi = im[
            top + (row._example_y - ROI_PADDING) : top + (row._example_y + ROI_SIZE + ROI_PADDING),
            left + (row._example_x - ROI_PADDING) : left + (row._example_x + ROI_SIZE + ROI_PADDING),
            :
        ]
        roi = np.copy(roi, order="C")
        roi_tensor = torch.tensor(roi).permute(2, 1, 0).clone(memory_format=torch.contiguous_format)

        return roi, roi_tensor, offset

    roi_offsets = process_map(preprocess_one_roi, example_df.iterrows(), desc="Reading images into RAM", leave=False, total=len(example_df))

    for i, row in tqdm(example_df.iterrows(), desc="Processing ROIs", leave=False, total=len(example_df)):
        roi_raw, roi_raw_tensor, offset = roi_offsets[i] 
        roi_annot : np.ndarray = annotations_to_tensor_predictions(row.annotations, offset, roi_raw_tensor, "").json_data
        roi_pred : np.ndarray = model.pyramid_predictions(roi_raw_tensor.to(device), example_df.raw_file_path).json_data

        rois["raw"].append(roi_raw)
        rois["annotation"].append(roi_annot)
        rois["prediction"].append(roi_pred)
    
    def save_rois(i):
        row = example_df.iloc[i]
        roi_raw_path = os.path.join(OUR_RAW, f"{row.short_name}.jpg")
        roi_annot_path = os.path.join(OUR_ANNOT, f"{row.short_name}.jpg")
        roi_pred_path = os.path.join(OUR_PRED, f"{row.short_name}.jpg")
        

        roi_raw = rois["raw"][i]
        roi_raw_tensor = torch.tensor(roi_raw).permute(2, 1, 0)
        
        roi_annot = TensorPredictions().load(rois["annotation"][i])
        roi_annot.image = roi_raw_tensor
        roi_annot = roi_annot.plot(confidence=False, contour_color=(34, 139, 34)) # Green contours
        roi_annot = np.transpose(roi_annot, (1, 0, 2))

        roi_pred = TensorPredictions().load(rois["prediction"][i])
        roi_pred.image = roi_raw_tensor
        roi_pred = roi_pred.plot(confidence=False, contour_color=(178, 34, 34)) # Red contours
        roi_pred = np.transpose(roi_pred, (1, 0, 2))

        roi_raw = roi_raw[ROI_PADDING:-ROI_PADDING, ROI_PADDING:-ROI_PADDING, :]
        roi_annot = roi_annot[ROI_PADDING:-ROI_PADDING, ROI_PADDING:-ROI_PADDING, :]
        roi_pred = roi_pred[ROI_PADDING:-ROI_PADDING, ROI_PADDING:-ROI_PADDING, :]

        cv2.imwrite(roi_raw_path, roi_raw)
        cv2.imwrite(roi_annot_path, roi_annot)
        cv2.imwrite(roi_pred_path, roi_pred)

        return roi_raw, roi_annot, roi_pred

    rois_proc = process_map(save_rois, range(len(example_df)), tqdm_class=tqdm, desc="Saving ROIs", leave=False, total=len(example_df))
    rois = {tp : ims for tp, ims in zip(["raw", "annotation", "prediction"], zip(*rois_proc))}

    for tp, ims in tqdm(rois.items(), desc="Creating mosaics", leave=False):
        mosaic = create_mosaic(ims, MOSAIC_SPACING, labels=example_df.short_name.tolist())
        cv2.imwrite(os.path.join(OUR_MOSAIC, f"{tp}.jpg"), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    """
    cd annotated_tiles && for i in *.jpg; do magick $i -fill '#0008' -draw 'rectangle  0,100,1000,0' -fill white   -font DejaVu-Sans-Mono-Book -pointsize 64  -annotate +40+70 "$(echo $i| cut -f 1 -d .)" $i; done && magick -size 1000x1000 canvas:white AAA.jpg && montage *.jpg -tile 4x6 -geometry 500x500+10+10 tiles.jpeg && cd ..
    cd raw_tiles && for i in *.jpg; do magick $i -fill '#0008' -draw 'rectangle  0,100,1000,0' -fill white   -font DejaVu-Sans-Mono-Book -pointsize 64  -annotate +40+70 "$(echo $i| cut -f 1 -d .)" $i; done &&  magick -size 1000x1000 canvas:white AAA.jpg && montage *.jpg -tile 4x6 -geometry 500x500+10+10 tiles.jpeg && cd ..
    """


