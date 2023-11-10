import os.path
import torch
import math
import logging
import numpy as np
import PIL.Image
import cv2
from shapely.geometry import Polygon
from ultralytics import YOLO


class Predictions(object):
    def __init__(self, image, original_image_path, contours, confs, classes, class_dict):
        self._OVERVIEWS_DIR = "overviews"
        self._CROPS_DIR = "crops"
        assert len(classes) == len(contours) == len(confs)
        self._class_dict = class_dict
        self._contours = contours
        self._classes = classes
        self._confs = confs
        self._image = image
        self._original_image_path = original_image_path
        # make bounding boxes for all contours fixme
        self._bboxes = []  # xywh
        for i, c in enumerate(self._contours):
            self._contours[i] = c.astype(np.int32)
            bb = cv2.boundingRect(c.astype(np.int32))
            self._bboxes.append(bb)

        if self._original_image_path:
            assert os.path.isfile(self._original_image_path)
            self._img_name_prefix = os.path.splitext(os.path.basename(self._original_image_path))[0]
        else:
            # fixme has here?!
            self._img_name_prefix = "image"

        if self._original_image_path is not None:
            self._dpis = self.get_dpis(self._original_image_path)
            array = cv2.imread(self._original_image_path)

            self._xy_scales = (self._image.shape[0] / array.shape[0],
                               self._image.shape[1] / array.shape[1],
                               )
        else:
            self._dpis = None
            array = self._image
            self._xy_scales = (1, 1)
        self._original_array = array

    def __getitem__(self, i):
        return Predictions(self._image,
                           self._original_image_path,
                           [self._contours[i]]
                           [self._confs[i]],
                           [self._classes[i]],
                           self._class_dict,
                           )

    def __len__(self):
        return len(self._contours)

    def draw(self, output_image):
        raise NotImplementedError

    def get_dpis(self, input):
        # fixme, this is a fallback to use jfif instead of exif!
        import exiftool
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(input)
            x_res = metadata[0]['JFIF:XResolution']
            y_res = metadata[0]['JFIF:YResolution']

        return x_res, y_res

    def make_crops(self, out_dir, draw_all_preds=True):

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "overviews"), exist_ok=True)

        array = self._original_array
        if draw_all_preds:
            overall_img = np.copy(self._image)

            for i, r in enumerate(self._bboxes):
                cv2.rectangle(img=overall_img, rec=r, color=(0, 255, 255),
                              thickness=5, lineType=cv2.LINE_AA)
                cv2.rectangle(img=overall_img, rec=r, color=(255, 0, 0),
                              thickness=3, lineType=cv2.LINE_AA)
                text = f"{i:0>4}"
                cv2.putText(img=overall_img, text=text, org=(r[0], r[1]), color=(255, 0, 0),
                            thickness=2, lineType=cv2.LINE_AA, fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            bottomLeftOrigin=False)

            cv2.drawContours(image=overall_img, contours=self._contours, contourIdx=-1, color=(0, 255, 255),
                             thickness=3, lineType=cv2.LINE_AA)
            cv2.drawContours(image=overall_img, contours=self._contours, contourIdx=-1, color=(255, 0, 0),
                             thickness=2, lineType=cv2.LINE_AA)

            basename = f"{self._img_name_prefix}.png"
            out_file = os.path.join(out_dir, self._OVERVIEWS_DIR, basename)
            cv2.imwrite(out_file, overall_img)

        for i, (bb, ct, cf, cl) in enumerate(zip(self._bboxes, self._contours, self._confs, self._classes)):
            x1, y1, w, h = bb
            x2 = x1 + w
            y2 = y1 + h

            x1 /= self._xy_scales[0]
            x2 /= self._xy_scales[0]
            y1 /= self._xy_scales[1]
            y2 /= self._xy_scales[1]

            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            roi = np.copy(array[y1:y2, x1:x2])

            mask = np.zeros_like(roi)
            #

            ct = np.divide(ct.astype(np.float), self._xy_scales).astype(np.int32)

            cv2.drawContours(image=mask, contours=[ct], contourIdx=-1, color=(255, 255, 255), thickness=-1,
                             lineType=cv2.LINE_8, offset=(-x1, -y1))

            roi = cv2.bitwise_and(roi, mask)
            if self._dpis:
                area_sqr_in = np.count_nonzero(mask) / (self._dpis[0] * self._dpis[1])
                area_sqr_mm = area_sqr_in * 645.16
                area_sqr_mm = round(area_sqr_mm)
            else:
                area_sqr_mm = 0
            basename = f"{self._img_name_prefix}_{i:0>4}_{x1}_{y1}_{area_sqr_mm:0>4}.jpg"
            out_file = os.path.join(out_dir, self._CROPS_DIR, basename)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            im = PIL.Image.fromarray(roi)
            im.save(out_file, dpi=self._dpis, quality=95)

    def coco_entry(self):

        h, w, _ = self._original_array.shape
        annotations = []

        image_info = {
            "id": None,
            "file_name": os.path.basename(self._original_image_path),
            "height": h,
            "width": w
        }

        for ct, bb, cl, conf in zip(self._contours, self._bboxes, self._classes, self._confs):

            scaled_ct = np.divide(ct.astype(np.float), self._xy_scales).astype(np.int32)
            xs, ys = self._xy_scales
            scaled_bbox = np.divide(np.array(bb, np.float),[xs,ys,xs,ys]).astype(np.int32).tolist()
            area = cv2.contourArea(scaled_ct)
            segmentation = [scaled_ct.flatten().tolist()]

            # Calculate the bounding box

            annotation_info = {
                "id": None,
                "image_id": image_info["id"],
                "category_id": cl,
                "segmentation": segmentation,
                "area": area,
                "bbox": scaled_bbox,
                "confidence": conf,
                "iscrowd": 0  # Assuming all instances are not crowded
            }
            annotations.append(annotation_info)

        return image_info, annotations



class Predictor(object):
    MIN_MAX_OBJ_SIZE = (16, 512)
    MINIMUM_TILE_OVERLAP = 256
    EDGE_CASE_MARGIN = 128
    SCORE_THRESHOLD = 0.7
    IOU_THRESHOLD = 0.5

    def __init__(self, model_path):
        self._model = YOLO(model_path)

    def _detect_instances(self, ori_array, scale=1.0):
        polys = []
        classes = []
        confs = []
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

        all_tiles = []
        for i, ((m, n), o) in enumerate(offsets):
            im_1 = array[o[1]: (o[1] + 1024), o[0]: (o[0] + 1024)]
            all_tiles.append(im_1)

        # batched inference does not run faster, but uses more memory... not promissing
        # n = self.BATCH_SIZE
        # all_tile_batches = [all_tiles[i * n:(i + 1) * n] for i in range((len(all_tiles) + n - 1) // n)]
        #
        # all_preds = []
        # for tiles in  all_tile_batches:
        #     preds = self._model(tiles)
        #     all_preds.extend(preds)

        for i, ((m, n), o) in enumerate(offsets):
            # logging.info(f"{img.filename}, {i}/{len(offsets)}")
            im_1 = array[o[1]: (o[1] + 1024), o[0]: (o[0] + 1024)]
            p = self._model(im_1, verbose=False)[
                0]  # fixme, if we have spare memory, we can run batch inference here!!!
            # p = all_preds[i]
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
            confs_for_one_inst = []

            # fixme, this could be parallelised
            for i in range(len(p)):
                poly = p[i].masks.xy[0] + np.array(o, dtype=np.float)
                if poly is not None:
                    poly_for_one_inst.append(poly)
                    classes_for_one_inst.append(
                        int(p[i].boxes.cls[0]) + 1)  # as we want one-indexed classes
                    confs_for_one_inst.append(float(p[i].boxes.conf[0]))

            polys.append(poly_for_one_inst)
            confs.append(classes_for_one_inst)
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
                        logging.warning("Topological exception")
                    if iou > self.IOU_THRESHOLD:
                        add = False
                        continue
                if add:
                    all_valid.append({"scale": scale,
                                      "tile": origin,
                                      "class": classes[origin][i],
                                      "confs": confs[origin][i],
                                      "contour": p1})
        for v in all_valid:
            v["contour"] /= scale
        return all_valid

    def pyramid_predictions(self, image, scale_increment=2 / 3, scale_before=1):
        if isinstance(image, str):
            im = cv2.imread(image)
            path = image
        else:
            path = None
            im = image
        h, w = im.shape[0], im.shape[1]
        if scale_before != 1:
            im = cv2.resize(im, dsize=(int(w * scale_before), int(h * scale_before)))

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
                        iou = 1  # fixme topological exception?
                        logging.warning("Topological exception")
                    if iou > self.IOU_THRESHOLD:
                        add = False
                        continue
                if add:
                    all_preds.append(p)

        cts = [p["contour"] for p in all_preds]
        confs = [p["confs"] for p in all_preds]
        classes = [p["class"] for p in all_preds]

        out = Predictions(im, path, cts, confs, classes,
                          self._model.names
                          )
        return out
