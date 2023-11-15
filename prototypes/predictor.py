import cv2
import numpy as np
import torch
import ultralytics

from flat_bug.predictor import Predictor
import math
from shapely.geometry import Polygon
import logging
from typing import List
from ultralytics.engine.results import Results
from ultralytics.engine.results import Masks
pred = Predictor("/home/quentin/repos/flat-bug-git/prototypes/runs/segment/train5/weights/best.pt")
# r: List[ultralytics.engine.results.Results] =

# todo write batch inference for tiles!


img_path = "./images/pitfall_0.25_2023-07-16_5_W_0m010.jpg"
# img = cv2.imread("./images/2023-07-16_5_W_5m_01_011.jpg")

prediction = pred.pyramid_predictions(img_path)
prediction.coco_entry()

#
#
# for p in all_preds:
#     m = p["contour"]
#
#
#
# p = all_preds[0]
# Results(orig_img=img, names = {0: "insects"}, path=None, boxes=None, masks=p["contour"])
# r = Results(orig_img=img, names = {0: "insects"}, path=None, boxes=None, masks=masks)


# contours = [np.round(c["contour"]).astype(np.int32) for c in all_preds]
#
# img = cv2.drawContours(img,contours,contourIdx = -1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
# cv2.imwrite("/tmp/test.jpg", img)
