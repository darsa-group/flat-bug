import math
import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import glob
import os
import json

# MODEL_PATH = 'runs/detect/train9/weights/best.pt'
MODEL_PATH = 'runs/segment/train78/weights/last.pt'

IMG_DIR = "/home/quentin/Desktop/pitfall_small"
SEGMENTATION_SCALE = 0.25
OUT_DIR = "/home/quentin/Desktop/extract_insects"


detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=0.5,
    device="cuda:0",
)

def process_image(im_path: str):
    result = get_sliced_prediction(im_path, detection_model,
                                   slice_height=1024,
                                   slice_width=1024,
                                   overlap_height_ratio=0.25,
                                   overlap_width_ratio=0.25,
                                   postprocess_type="GREEDYNMM",
                                   )
    #fixme post processing
    # filter small contours
    # filter disjoint contours


    return result

import math

if __name__ =="__main__":
    for im_path in glob.glob(os.path.join(IMG_DIR, "*.jpg")):
        print(im_path)
        raw_im = cv2.imread(im_path)
        r_h, r_w = raw_im.shape[0:2]
        im = cv2.resize(raw_im, dsize=None, fx=SEGMENTATION_SCALE, fy=SEGMENTATION_SCALE)
        results = process_image(im)
        large_mask = np.zeros((r_h, r_w), np.float)
        for i, r in enumerate(results.object_prediction_list):
            cv2.resize(r.mask.bool_mask.astype(np.float), dst= large_mask, dsize = (r_w, r_h), interpolation=cv2.INTER_NEAREST)
            # large_mask_bool = large_mask.astype(np.bool)
            y0, y1 = math.ceil(r.bbox.miny / SEGMENTATION_SCALE), math.floor(r.bbox.maxy / SEGMENTATION_SCALE)
            x0, x1 = math.ceil(r.bbox.minx / SEGMENTATION_SCALE), math.floor(r.bbox.maxx / SEGMENTATION_SCALE)
            sub_im = np.copy(raw_im[y0:y1, x0:x1])
            sub_im[large_mask[y0:y1, x0:x1] == 0] = 0
            basename = f"{os.path.splitext(os.path.basename(im_path))[0]}_{i:0>5}_{x0}_{y0}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, basename), sub_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imshow("img", sub_im)
            cv2.waitKey(1)

