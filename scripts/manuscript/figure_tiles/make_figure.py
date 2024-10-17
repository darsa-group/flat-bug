import math
import os
import glob

from typing import List, Optional

from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2

import torch

from flat_bug.predictor import Predictor

MOSAIC_SPACING = 0

ANNOTATED_IM_DIR = "dev/fb_yolo/insects"
RAW_IM_DIR = "dev/fb_yolo/insects"
THIS_DIR = os.path.dirname(__file__)
OUR_ANNOT = os.path.join(THIS_DIR, "annotated_tiles")
OUR_RAW = os.path.join(THIS_DIR, "raw_tiles")
OUR_PRED = os.path.join(THIS_DIR, "pred_tiles")
OUR_MOSAIC = os.path.join(THIS_DIR, "mosaic")

MODEL = "model_snapshots/fb_tmp3_medium.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Predictor(MODEL, device=device, dtype=torch.float16)


df = pd.read_csv(os.path.join(THIS_DIR, "clean_flatbug_datasets.csv"))
df["annotation_file_path"] = [""] * df.shape[0]

for f in glob.glob(os.path.join(ANNOTATED_IM_DIR, "**", "*.jpg"), recursive=True):
    if "00-all" in f:
        continue
    bn = os.path.basename(f)
    for i, (d,t) in enumerate(zip(df.dataset, df._example_name)):
        if t in bn and d in f:
            # df["annotation_file_path"][i] = f
            df.loc[i, "annotation_file_path"] = f


for f in glob.glob(os.path.join(RAW_IM_DIR, "**", "*.jpg"), recursive=True):
    if "00-all" in f:
        continue
    bn = os.path.basename(f)
    for i, (d,t) in enumerate(zip(df.dataset, df._example_name)):
        if t in bn and d in f:
            # df["annotation_file_path"][i] = f
            df.loc[i, "raw_file_path"] = f


os.makedirs(OUR_ANNOT, exist_ok=True)
os.makedirs(OUR_RAW, exist_ok=True)
os.makedirs(OUR_PRED, exist_ok=True)
os.makedirs(OUR_MOSAIC, exist_ok=True)

rois = {tp : [] for tp in ["annotation", "raw", "prediction"]}

for i,r in tqdm(df.iterrows(), desc="Processing ROIs", leave=False, total=len(df)):
    im = cv2.imread(r.annotation_file_path)
    im_raw = cv2.imread(r.raw_file_path)

    h,w = im.shape[0:2]

    if w < 1000:
        left = math.ceil((1000 - w )/2)
        right = math.floor((1000 - w) / 2)
        im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        im_raw = cv2.copyMakeBorder(im_raw, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

    roi = im[r._example_y: r._example_y + 1000, r._example_x : r._example_x + 1000, :]
    roi_raw = im_raw[r._example_y: r._example_y + 1000, r._example_x : r._example_x + 1000, :]
    roi_pred = model.pyramid_predictions(torch.tensor(roi_raw).permute(2, 1, 0).to(device), im_raw).plot(conf=False)

    rois["annotation"].append(roi)
    rois["raw"].append(roi_raw)
    rois["prediction"].append(roi_pred)

    cv2.imwrite(os.path.join(OUR_ANNOT, f"{r.short_name}.jpg"), roi)
    cv2.imwrite(os.path.join(OUR_RAW, f"{r.short_name}.jpg"), roi_raw)
    cv2.imwrite(os.path.join(OUR_PRED, f"{r.short_name}.jpg"), roi_pred)

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

for tp, ims in tqdm(rois.items(), desc="Creating mosaics", leave=False):
    mosaic = create_mosaic(ims, MOSAIC_SPACING, labels=df.short_name.tolist())
    cv2.imwrite(os.path.join(OUR_MOSAIC, f"{tp}.jpg"), mosaic)

"""
cd annotated_tiles && for i in *.jpg; do magick $i -fill '#0008' -draw 'rectangle  0,100,1000,0' -fill white   -font DejaVu-Sans-Mono-Book -pointsize 64  -annotate +40+70 "$(echo $i| cut -f 1 -d .)" $i; done && magick -size 1000x1000 canvas:white AAA.jpg && montage *.jpg -tile 4x6 -geometry 500x500+10+10 tiles.jpeg && cd ..
cd raw_tiles && for i in *.jpg; do magick $i -fill '#0008' -draw 'rectangle  0,100,1000,0' -fill white   -font DejaVu-Sans-Mono-Book -pointsize 64  -annotate +40+70 "$(echo $i| cut -f 1 -d .)" $i; done &&  magick -size 1000x1000 canvas:white AAA.jpg && montage *.jpg -tile 4x6 -geometry 500x500+10+10 tiles.jpeg && cd ..
"""


