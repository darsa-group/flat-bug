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
from typing import List, Tuple, Dict, Union
from math import floor, ceil

import numpy as np
import torch
import cv2

from torchvision import io as tio

from ultralytics.engine.results import Results, Masks

from flat_bug.predictor import Predictor
from flat_bug.yolo_helpers import postprocess


class AutoMaskRefiner:
    def __init__(self, weights, device=torch.device("cpu"), dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.model = Predictor(weights, device=self.device, dtype=self.dtype)

    def __call__(self, crops : List[torch.Tensor]) -> List[Results]:
        with torch.no_grad():
            return postprocess(
                preds = self.model._model(crops),
                imgs = crops,
                max_det = 100,
                min_confidence=0.1,
                iou_threshold=0,
                valid_size_range=self.model.MIN_MAX_OBJ_SIZE,
                edge_margin=0,
                nms=3,
                group_first=self.model.EXPERIMENTAL_NMS_OPTIMIZATION # Doesn't have an effect at the moment, feature disabled for postprocessing
            )
    
    def refine_contours(self, contours : list[torch.Tensor], image : Union[str, torch.Tensor]) -> list[torch.Tensor]:
        # Convert contours (or boxes) to boxes
        bboxes_bottom_left = torch.stack([c.min(dim=0).values for c in contours]).to(self.device).round().long()
        bboxes_top_right = torch.stack([c.max(dim=0).values for c in contours]).to(self.device).round().long()
        bboxes = torch.cat([bboxes_bottom_left, bboxes_top_right], dim=1)
        if bboxes.device != self.device:
            bboxes = bboxes.to(self.device)

        # Crop the image
        if isinstance(image, str):
            image = tio.read_image(image, tio.ImageReadMode.RGB).to(device=self.device, dtype=self.dtype) / 255.
        if image.device != self.device:
            image = image.to(self.device)
        if image.dtype != self.dtype:
            image = image.to(self.dtype)

        paddings = []
        scalings = []
        contours = []
        
        batch_start = list(range(0, len(bboxes), 16))
        batch_end = batch_start[1:] + [len(bboxes)]

        for start, end in zip(batch_start, batch_end):
            crops = [image[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in bboxes[start:end]]

            # Pad the crops to 1024x1024 if they are smaller, and resize them to 1024x1024 if they are larger
            for i, crop in enumerate(crops):
                h, w = crop.shape[-2:]
                # Calculate the necessary padding
                pad_h = max(0, 1024 - h)
                pad_w = max(0, 1024 - w)
                # Split the padding into top/bottom and left/right
                pad_h_top = pad_h // 2
                pad_h_bottom = pad_h - pad_h_top
                pad_w_left = pad_w // 2
                pad_w_right = pad_w - pad_w_left
                # Add the padding to the list
                paddings.append((pad_h_top, pad_h_bottom, pad_w_left, pad_w_right))
                # If padding is necessary, add the padding
                if pad_h > 0 or pad_w > 0:
                    h, w = h + pad_h, w + pad_w
                    crops[i] = torch.nn.functional.pad(crop, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode="constant", value=0)
                # Calculate the necessary scaling
                scale_h, scale_w = 1024 / h, 1024 / w
                # Add the scaling to the list
                scalings.append((scale_h, scale_w))
                # If scaling is necessary, scale the crop
                if scale_h != 1 or scale_w != 1:
                    crops[i] = torch.nn.functional.interpolate(crop.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False).squeeze(0)

            # Run the crops through the model
            contours.extend([result.masks.xy for result in self(torch.stack(crops))]) # Mutably extend the results list with the results of the model on each batch of crops

        print(paddings)

        # Undo the padding and scaling
        for i, (contour, padding, scaling) in enumerate(zip(contours, paddings, scalings)):
            if not contour:
                continue
            contour = torch.tensor(contour[0], device=self.device, dtype=self.dtype)
            # Undo the scaling
            scale_factor = torch.tensor(scaling).to(contour.device, dtype=contour.dtype)
            # Undo the padding
            padding_offset = -torch.tensor([padding[2], padding[0]]).to(contour.device, dtype=contour.dtype)
            contours[i] = ((contour + padding_offset) / scale_factor).round().long() + bboxes_bottom_left[i, :2]

        return contours

class BaseMaskRefiner(object):
    _min_bbox_size = 40  # below this size, we keep original values
    _image_padding = 128
    def __init__(self):
        pass

    def _refine_one_mask(self, roi, contour, offset):
        return contour

    def run(self, coco_annotations, images_root_dir, output_crop_dir):

        with open(coco_annotations) as f:
            coco = json.load(f)
        cv2.namedWindow("mask_refiner")

        root_dir = images_root_dir
        image_file_map = {im["id"]: im["file_name"] for im in coco["images"]}
        # image_file_map_rev = {im["file_name"]: im["id"] for im in coco["images"]}

        cache_fn = ""
        images = [i for i in sorted(image_file_map.values())]

        for ann in coco["annotations"]:
            im_filename = image_file_map[ann["image_id"]]
            if im_filename not in images[0:50]:
                print("skipping")
                continue

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

            roi = np.copy(im[bbox[1]: bbox[3] + self._image_padding * 2, bbox[0]: bbox[2] + self._image_padding * 2])

            crop_name = f"{x}_{y}_{self._image_padding}_{im_filename}"
            print(crop_name)
            cv2.imwrite(os.path.join(output_crop_dir, crop_name), roi)
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
    _image_padding = 64
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
        cv2.imshow("ROI", roi)
        cv2.imshow("mask_refiner2", rough_mask)

        rough_mask = rough_mask.astype(np.bool_)
        best_mask = None
        best_score = 0
        print("len(result)")
        print(len(result))
        for r in result:

            for i in ["normal", "negative"]:
                if i == "negative":
                    m = np.bitwise_not(r["segmentation"])
                else:
                    m = r["segmentation"]

                score = self._mask_goodness(rough_mask, m)
                print(score)
                if score > best_score:
                    best_mask = m
                    best_score = score
                # cv2.imwrite(f"/tmp/test_{i}.png", m.astype(np.uint8) * 255)
        # best_mask = np.bitwise_and(best_mask, rough_mask)
        best_mask = cv2.dilate(best_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        pol = self._mask_to_polygon(best_mask)
        # cv2.drawContours(image_rgb, [pol], -1, (255, 0, 255), 3, lineType=cv2.LINE_AA)
        return pol - offset
class YoloMaskRefiner(BaseMaskRefiner):
    def __init__(self, model_weights):
        from ultralytics import YOLO
        from PIL import Image
        self._model = YOLO("/home/quentin/Desktop/fb_weights/fb_2024-02-29_large_best.pt")
    def _refine_one_mask(self, roi, contour, offset):
        # rough_mask = np.zeros((roi.shape[0], roi.shape[1]), np.uint8)
        # rough_mask = cv2.drawContours(rough_mask, [contour], -1, (255,), -1, lineType=cv2.LINE_8,
        #                                offset=offset)
        # rough_mask = cv2.dilate(rough_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(51,51)))
        # #
        # roi = cv2.bitwise_and(roi, roi, mask=rough_mask)
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

if __name__ == "__main__":    

    mr = BaseMaskRefiner()
    # mr = YoloMaskRefiner("~/Desktop/fb_weights/fb_2024-03-07_large_best.pt")
    # mr = SAMMaskRefiner("../sam_vit_h_4b8939.pth")
    coco = mr.run("/home/quentin/Desktop/flat-bug-preannot/00_NHM-beetles/coco_instances.json",
                "/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/00_NHM-beetles",
                    "/home/quentin/Desktop/nhm_crops"
                )
    # coco = mr.run("/home/quentin/Desktop/flat-bug-preannot/blair2020/coco_instances.json", "/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/blair2020")
    with open("/tmp/abram_refined.json", 'w') as f:
        json.dump(coco, f)
