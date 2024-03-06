# from ultralytics import YOLO
# from PIL import Image
# model = YOLO("/home/quentin/Desktop/fb_weights/fb_2024-02-29_large_best.pt")
#
# results = model(["CPER_04_DORSAL-crop-03.jpg"])
#
# print(results)
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#
#     im.save('results.jpg')  # save image
import json
import os
from math import floor, ceil
import numpy as np

import cv2


class BaseMaskRefiner(object):
    _min_bbox_size = 40  # below this size, we keep original values
    _image_padding = 256
    def __init__(self):
        pass

    def _refine_one_mask(self, roi, contour, offset):
        return contour

    def run(self, coco_annotations, images_root_dir):

        with open(coco_annotations) as f:
            coco = json.load(f)
        cv2.namedWindow("mask_refiner")

        root_dir = images_root_dir
        image_file_map = {im["id"]: im["file_name"] for im in coco["images"]}
        # image_file_map_rev = {im["file_name"]: im["id"] for im in coco["images"]}

        cache_fn = ""
        for ann in coco["annotations"]:
            im_filename = image_file_map[ann["image_id"]]

            if im_filename != cache_fn:
                im = cv2.imread(os.path.join(root_dir, im_filename))
                cache_fn = im_filename
                im = cv2.copyMakeBorder(im, self._image_padding, self._image_padding,
                                        self._image_padding, self._image_padding, cv2.BORDER_CONSTANT, (0,0,0))
            x, y, w, h = ann["bbox"]
            if x < 0 or y < 0:
                del ann
                continue

            # roi = cv2.medianBlur(roi, 5)
            #
            # if len(ann["segmentation"]) > 1:
            #     areas = []
            #     for s in ann["segmentation"]:
            #         c = np.round([s]).astype(np.int32)
            #         c = c.reshape((len(c[0]) // 2, 1, 2))
            #         if len(c) < 3:
            #             a = 0
            #         else:
            #             a = cv2.contourArea(c)
            #         areas.append(a)
            #     seg = [ann["segmentation"][np.argmax(areas)]]
            # else:
            assert len(ann["segmentation"]) == 1
            seg = ann["segmentation"]
            contour = np.round(seg).astype(np.int32)
            contour = contour.reshape((len(contour[0]) // 2, 1, 2))

            x,y,w,h = cv2.boundingRect(contour)

            bbox = floor(x), floor(y), ceil(x + w), ceil(y + h)
            print(bbox)
            roi = cv2.cvtColor(im[bbox[1]: bbox[3] + self._image_padding * 2, bbox[0]: bbox[2] + self._image_padding * 2],
                               cv2.COLOR_BGR2RGB)

            x += 1
            y += 1
            if w > self._min_bbox_size and h > self._min_bbox_size:
                better_ct = self._refine_one_mask(roi, contour, offset = (-int(x) + self._image_padding, -int(y) + self._image_padding))
                if better_ct is not None:
                    better_ct = better_ct + (self._image_padding, self._image_padding)
                    candidate_contour = better_ct


                    # if abs(bb[2] - w) / w < .25 and abs(bb[3] - h) / h < .25:

                    cv2.drawContours(roi, [contour], -1, (255, 0, 0), 2, lineType=cv2.LINE_AA,
                                     offset=(-int(x-  self._image_padding), -int(y- self._image_padding)))
                    cv2.drawContours(roi, [candidate_contour], -1, (0, 0, 255), 2, lineType=cv2.LINE_AA,
                                     offset=(-int(x), -int(y)))
                    contour = candidate_contour
                    contour = contour - (self._image_padding, self._image_padding)
                    bb = cv2.boundingRect(contour)

                    ann["segmentation"] = [contour.flatten().tolist()]
                    ann["bbox"] = bb
                    ann["refined"] = True
                    cv2.imshow("mask_refiner", roi)
                    cv2.waitKey(1)
                else:
                    print("No countour after refinement")
                # refine_mask(roi, contour, offset=(-int(x),-int(y))
            else:
                print("skipping small contour")
            assert im is not None
        return coco


class SAMMaskRefiner(BaseMaskRefiner):
    def __init__(self, model_weights):
        from segment_anything import SamAutomaticMaskGenerator
        from segment_anything import sam_model_registry
        import torch
        self._sam = sam_model_registry["vit_h"](checkpoint=model_weights)
        dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._sam.to(device=dev)
        self._mask_generator = SamAutomaticMaskGenerator(self._sam)

    def _mask_goodness(self, prelim_mask, candidate_mask):
        union = np.sum(np.bitwise_or(prelim_mask, candidate_mask))
        intersect = np.sum(np.bitwise_and(prelim_mask, candidate_mask))
        p_points_in = intersect / np.sum(candidate_mask)
        # print(p_points_in, intersect / union)
        if p_points_in < 0.75:
            return 0.0

        return np.sum(candidate_mask)

    def _mask_to_polygon(self, mask):
        cts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = None
        max_area = 0

        for c in cts:
            a = cv2.contourArea(c)
            if a > max_area:
                max_area = a
                largest = c
        return largest

    def _refine_one_mask(self, roi, contour, offset):
        rough_mask = np.zeros((roi.shape[0], roi.shape[1]), np.uint8)
        rough_mask = cv2.drawContours(rough_mask, [contour], -1, (255,), -1, lineType=cv2.LINE_8,
                                      offset=offset)
        result = self._mask_generator.generate(roi)
        cv2.imshow("mask_refiner", rough_mask)

        rough_mask = rough_mask.astype(np.bool_)
        best_mask = None
        best_score = 0
        for r in result:
            for i in ["normal", "negative"]:
                if i == "negative":
                    m = np.bitwise_not(r["segmentation"])
                else:
                    m = r["segmentation"]

                score = self._mask_goodness(rough_mask, m)
                if score > best_score:
                    best_mask = m
                    best_score = score
                # cv2.imwrite(f"/tmp/test_{i}.png", m.astype(np.uint8) * 255)
        best_mask = np.bitwise_and(best_mask, rough_mask)

        pol = self._mask_to_polygon(best_mask)
        # cv2.drawContours(image_rgb, [pol], -1, (255, 0, 255), 3, lineType=cv2.LINE_AA)
        return pol
class YoloMaskRefiner(BaseMaskRefiner):
    def __init__(self, model_weights):
        from ultralytics import YOLO
        from PIL import Image
        self._model = YOLO("/home/quentin/Desktop/fb_weights/fb_2024-02-29_large_best.pt")
    def _refine_one_mask(self, roi, contour, offset):
        rough_mask = np.zeros((roi.shape[0], roi.shape[1]), np.uint8)
        rough_mask = cv2.drawContours(rough_mask, [contour], -1, (255,), -1, lineType=cv2.LINE_8,
                                       offset=offset)
        rough_mask = cv2.dilate(rough_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51)))
        #
        roi = cv2.bitwise_and(roi, roi, mask=rough_mask)
        # roi[np.where(roi == 0)] = 0
        results = self._model([roi])
        if results[0].masks is None:
            print("No detection")
            return contour
        new_contours = results[0].masks.xy
        if len(new_contours) != 1:
            print("more than one detection?")
            return contour

        contour = np.round(new_contours[0] - offset).astype(np.uint32)


        # for r in results:
        #     im_array = r.plot()  # plot a BGR numpy array of predictions
        #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #     im.save('results.jpg')  # save image
        print("Refined")
        return contour

mr = YoloMaskRefiner("~/Desktop/fb_weights/fb_2024-02-29_large_best.pt")
coco = mr.run("/home/quentin/Desktop/flat-bug-preannot/blair2020/coco_instances.json", "/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/blair2020")
with open("/tmp/refined.json", 'w') as f:
    json.dump(coco, f)
