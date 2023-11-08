import torch
import math
from shapely.geometry import Polygon
import logging

import numpy as np
import cv2
from ultralytics import YOLO


class Predictor(object):
    MIN_MAX_OBJ_SIZE = (16, 512)
    MINIMUM_TILE_OVERLAP = 256
    EDGE_CASE_MARGIN = 128
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.3

    def __init__(self, model_path):
        self._model = YOLO(model_path)

    def _detect_instances(self, ori_array, scale=1.0):
        polys = []
        classes = []
        h, w = ori_array.shape[0], ori_array.shape[1]

        if scale == 1:
            array = np.copy(ori_array)
        else:
            array = cv2.resize(ori_array, dsize=(int(w * scale), int(h * scale)))
        if array.shape[1] <= 1024:
            x_range = [0]
            x_n_tiles = 1
        else:
            x_n_tiles = math.ceil(1 + (array.shape[1] - 1024) / (1024 - self.MINIMUM_TILE_OVERLAP))
            x_stride = (array.shape[1] - 1024) // (x_n_tiles - 1)
            x_range = [r for r in range(0, array.shape[1] - 1023, x_stride)]

        if array.shape[0] <= 1024:
            y_range = [0]
            y_n_tiles = 1

        else:
            y_n_tiles = math.ceil(1 + (array.shape[0] - 1024) / (1024 - self.MINIMUM_TILE_OVERLAP))
            y_stride = (array.shape[0] - 1024) // (y_n_tiles - 1)
            y_range = [r for r in range(0, array.shape[0] - 1023, y_stride)]
        offsets = []
        for n, j in enumerate(y_range):
            for m, i in enumerate(x_range):
                offsets.append(((m, n), (i, j)))

        for i, ((m, n), o) in enumerate(offsets):
            # logging.info(f"{img.filename}, {i}/{len(offsets)}")
            im_1 = array[o[1]: (o[1] + 1024), o[0]: (o[0] + 1024)]
            p = self._model(im_1)[0]  # fixme, if we have spare memory, we can run batch inference here!!!
            p_bt = p.boxes.xyxy

            big_enough = torch.zeros_like(p_bt[:, 0], dtype=torch.bool)
            big_enough = big_enough.__or__(p_bt[:, 2] - p_bt[:, 0] > self.MIN_MAX_OBJ_SIZE[0])
            big_enough = big_enough.__or__(p_bt[:, 3] - p_bt[:, 1] > self.MIN_MAX_OBJ_SIZE[0])

            non_edge_cases = torch.ones_like(p_bt[:, 0], dtype=torch.bool)

            if m > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 0] > self.EDGE_CASE_MARGIN)
            if m < x_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 2] < 1024 - self.EDGE_CASE_MARGIN)

            if n > 0:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 1] > self.EDGE_CASE_MARGIN)
            if n < y_n_tiles - 1:
                non_edge_cases = non_edge_cases.__and__(p_bt[:, 3] < 1024 - self.EDGE_CASE_MARGIN)

            p = p[non_edge_cases.__and__(big_enough)]
            p = p[p.boxes.conf > self.SCORE_THRESHOLD]
            classes_for_one_inst = []
            poly_for_one_inst = []

            # fixme, this could be parallelised
            for i in range(len(p)):
                poly = p[i].masks.xy[0] + np.array(o, dtype=np.float)
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
                    if origin not in overlappers[v["tile"]]:
                        continue
                    p2 = v["contour"]
                    p_shape_2 = Polygon(np.squeeze(p2))

                    # todo check bounding box overlap
                    try:
                        iou = p_shape_1.intersection(p_shape_2).area / p_shape_1.union(p_shape_2).area
                    except Exception as e:
                        iou = 1  # fixme topological exception
                    if iou > self.IOU_THRESHOLD:
                        add = False
                        continue
                if add:
                    all_valid.append({"scale": scale,
                                      "tile": origin,
                                      "class": classes[origin][i],
                                      "contour": p1})
        for v in all_valid:
            v["contour"] /= scale
        return all_valid

    def pyramid_predictions(self, image, scale_increment=2/3):

        if isinstance(image, str):
            im = cv2.imread(image)
        else:
            im = image
        assert len(im.shape) == 3, f"Image{image} is not 3-dimentional"
        assert im.shape[2] == 3, f"Image{image} does not have 3 channels"

        h, w = im.shape[0], im.shape[1]
        min_dim = min(h, w)
        # fixme, what to do if the image is too small?
        # 0-pad
        scales = []
        s = 1
        while min_dim * s >= 1024:
            scales.append(s)
            s *= scale_increment
        logging.info(f"Running inference on scales: {scales}")

        all_preds = []
        for s in reversed(scales):
            preds = self._detect_instances(im, scale=s)
            for p in preds:
                add = True
                p_shape = Polygon(np.squeeze(p["contour"]))
                for val_p in all_preds:
                    val_p_shape = Polygon(np.squeeze(val_p["contour"]))
                    try:
                        iou = p_shape.intersection(val_p_shape).area / p_shape.union(val_p_shape).area
                    except Exception as e:
                        iou = 1  # fixme topological exception
                    if iou > self.IOU_THRESHOLD:
                        add = False
                        continue
                if add:
                    all_preds.append(p)
        return all_preds
