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

pred = Predictor("/home/quentin/repos/flat-bug-git/prototypes/runs/segment/train5/weights/best.pt")
# r: List[ultralytics.engine.results.Results] =

# todo write batch inference for tiles!

MIN_MAX_OBJ_SIZE = (32, 512)
MINIMUM_TILE_OVERLAP=256
EDGE_CASE_MARGIN = 64
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def detect_instances(array):
    polys = []
    classes = []

    if array.shape[1] <= 1024:
        x_range = [0]
        x_n_tiles = 1
    else:
        x_n_tiles = math.ceil(1 + (array.shape[1] - 1024) / (1024 - MINIMUM_TILE_OVERLAP))
        x_stride = (array.shape[1] - 1024) // (x_n_tiles - 1)
        x_range = [r for r in range(0, array.shape[1] - 1023, x_stride)]

    if array.shape[0] <= 1024:
        y_range = [0]
        y_n_tiles = 1

    else:
        y_n_tiles = math.ceil(1 + (array.shape[0] - 1024) / (1024 - MINIMUM_TILE_OVERLAP))
        y_stride = (array.shape[0] - 1024) // (y_n_tiles - 1)
        y_range = [r for r in range(0, array.shape[0] - 1023, y_stride)]
    offsets = []
    for n, j in enumerate(y_range):
        for m, i in enumerate(x_range):
            offsets.append(((m, n), (i, j)))

    for i, ((m, n), o) in enumerate(offsets):
        # logging.info(f"{img.filename}, {i}/{len(offsets)}")
        im_1 = array[o[1]: (o[1] + 1024), o[0]: (o[0] + 1024)]
        p = pred.tiled_prediction(im_1)[0]
        p_bt = p.boxes.xyxy

        big_enough = torch.zeros_like(p_bt[:, 0], dtype=torch.bool)
        big_enough = big_enough.__or__(p_bt[:, 2] - p_bt[:, 0] > MIN_MAX_OBJ_SIZE[0])
        big_enough = big_enough.__or__(p_bt[:, 3] - p_bt[:, 1] > MIN_MAX_OBJ_SIZE[0])

        non_edge_cases = torch.ones_like(p_bt[:, 0], dtype=torch.bool)

        if m > 0:
            non_edge_cases = non_edge_cases.__and__(p_bt[:, 0] > EDGE_CASE_MARGIN)
        if m < x_n_tiles - 1:
            non_edge_cases = non_edge_cases.__and__(p_bt[:, 2] < 1024 - EDGE_CASE_MARGIN)

        if n > 0:
            non_edge_cases = non_edge_cases.__and__(p_bt[:, 1] > EDGE_CASE_MARGIN)
        if n < y_n_tiles - 1:
            non_edge_cases = non_edge_cases.__and__(p_bt[:, 3] < 1024 - EDGE_CASE_MARGIN)

        p = p[non_edge_cases.__and__(big_enough)]
        p = p[p.boxes.conf > SCORE_THRESHOLD]
        classes_for_one_inst = []
        poly_for_one_inst = []

        # fixme, this could be parallelised
        for i in range(len(p)):

            poly = p[i].masks.xy[0] + np.array(o, dtype=np.float)
            print(poly)
            if poly is not None:
                poly_for_one_inst.append(poly)
                classes_for_one_inst.append(
                    int(p[i].boxes.cls[0]) + 1)  # as we want one-indexed classes
        polys.append(poly_for_one_inst)
        classes.append(classes_for_one_inst)

    overlappers = []
    for i in range(len(offsets)):
        overlappers_sub = []
        for j in range(len(offsets)):
            if i != j and abs(offsets[j][1][0] - offsets[i][1][0]) < 1024 and abs(
                    offsets[j][1][1] - offsets[i][1][1]) < 1024:
                overlappers_sub.append(j)
        overlappers.append(overlappers_sub)

    # print(polys)
    all_valid = []  # (origin, pred_class, poly)

    # merge predictions from several overlaping tiles
    for origin, poly_one_pred in enumerate(polys):
        # print(len(poly_one_pred))
        for i, p1 in enumerate(poly_one_pred):

            add = True
            p_shape_1 = Polygon(np.squeeze(p1))

            for v in all_valid:
                if origin not in overlappers[v[0]]:
                    continue
                p2 = v[2]
                p_shape_2 = Polygon(np.squeeze(p2))

                # todo check bounding box overlap
                try:
                    iou = p_shape_1.intersection(p_shape_2).area / p_shape_1.union(p_shape_2).area
                except Exception as e:
                    iou = 1  # fixme topological exception
                    print("topo")
                if iou > IOU_THRESHOLD:
                    add = False
                    continue
            if add:
                all_valid.append((origin, classes[origin][i], p1))
    # annotation_list = []
    # for _, pred_class, poly in all_valid:
    #     stroke = self._palette.get_stroke_from_id(pred_class)
    #     class_name = self._palette.get_class_from_id(pred_class)
    #     a = Annotation(poly, parent_image=img, stroke_colour=stroke, name=class_name)
    #     annotation_list.append(a)
    print(all_valid)
    return all_valid


img = cv2.imread("./images/pitfall_0.25_2023-07-16_5_W_0m010.jpg")
all_pres = detect_instances(img)
contours = [np.round(c[2]).astype(np.int32) for c in all_pres]

img = cv2.drawContours(img,contours,contourIdx = -1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
cv2.imwrite("/tmp/test.jpg", img)
