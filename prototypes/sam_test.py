import os.path

import numpy as np
from math import floor, ceil
import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator

from segment_anything import sam_model_registry
import json

import morphsnakes as ms

# import supervision as sv


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
# CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
BASE_PATH="/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/pitfall/"
COCO_ANNOTS = os.path.join(BASE_PATH, "instances_default.json")
COCO_ANNOTS_REFINED = os.path.join(BASE_PATH, "instances_default_refined.json")

# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
# sam.to(device=DEVICE)


def mask_goodness(prelim_mask, candidate_mask):
    union = np.sum(np.bitwise_or(prelim_mask, candidate_mask))
    intersect = np.sum(np.bitwise_and(prelim_mask, candidate_mask))
    p_points_in = intersect / np.sum(candidate_mask)
    # print(p_points_in, intersect / union)
    if p_points_in < 0.75:
        return 0.0

    return np.sum(candidate_mask)


def mask_to_polygon(mask):
    cts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = None
    max_area = 0

    for c in cts:
        a = cv2.contourArea(c)
        if a > max_area:
            max_area = a
            largest = c
    return largest


def refine_mask(image_rgb, rough_mask_poly, offset):
    rough_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), np.uint8)
    rough_mask = cv2.drawContours(rough_mask, [rough_mask_poly], -1, (255,), -1, lineType=cv2.LINE_8, offset=offset)

    result = mask_generator.generate(image_rgb)
    rough_mask = rough_mask.astype(np.bool)
    best_mask = None
    best_score = 0
    for r in result:
        for i in ["normal", "negative"]:
            if i == "negative":
                m = np.bitwise_not(r["segmentation"])
            else:
                m = r["segmentation"]

            score = mask_goodness(rough_mask, m)
            if score > best_score:
                best_mask = m
                best_score = score
            # cv2.imwrite(f"/tmp/test_{i}.png", m.astype(np.uint8) * 255)
    best_mask = np.bitwise_and(best_mask, rough_mask)

    pol = mask_to_polygon(best_mask)
    cv2.drawContours(image_rgb, [pol], -1, (255, 0, 255), 3, lineType=cv2.LINE_AA)

    return pol





def refine_contour(img, ct, size=400, offset=None):
    # img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # Initialization of the level-set.

    h, w = img.shape[0], img.shape[1]

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask.fill(0)

    mask = cv2.drawContours(mask, [ct], -1, (255,), -1, cv2.LINE_8, offset=offset)

    img = cv2.resize(img, (size, size))
    # img = cv2.medianBlur(img,3)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, kernel=np.ones((3,3), np.uint8))





    means =  []
    stds =  []
    masked = mask.flatten().astype(bool)
    for i in range(3):
        v = img[:, :, i].flatten()[np.bitwise_not(masked)]
        # print("np.mean(v)", np.mean(v))
        means.append(np.mean(v))
        stds.append(np.std(v))


    from scipy import stats

    likelihoods = [stats.norm.logpdf(img[:,:,i], loc=means[i], scale=stds[i]) for i in range(3)]
    l_map = np.stack(likelihoods,axis=2)
    l_map = np.sum(l_map, axis=2)
    l_map = -l_map
    l_map = l_map - np.min(l_map)
    l_map = l_map / np.max(l_map)
    l_map_like = np.sqrt(l_map)


    l_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    l_map = cv2.medianBlur(l_map, 3)
    l_map_blur = cv2.medianBlur(l_map, 75)
    l_map = cv2.subtract(l_map_blur, l_map) * 5
    l_map[l_map < 50] = 0
    l_map = l_map - np.min(l_map)
    l_map = l_map / np.max(l_map)
    cv2.imshow("l_map", l_map +  l_map_like)
    l_map = (l_map + l_map_like) / 2

    o = ms.morphological_chan_vese(l_map, iterations=100,
                                   init_level_set=mask,
                                   smoothing=2, lambda1=0.001, lambda2=0.1,
                                   # iter_callback= callback
                                   )
    o[mask == 0] = 0
    # todo, scale new mask UP!

    o = cv2.resize(o, (w, h), interpolation=cv2.INTER_NEAREST)
    o = cv2.dilate(o.astype(np.uint8), kernel=np.ones((5, 5), np.uint8))

    assert o.shape == (h, w)
    cts, _ = cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                              offset=(-offset[0], -offset[1]))

    largest_area = 0
    largest = None
    for c in cts:
        a = cv2.contourArea(c)
        if a > largest_area:
            largest = c
            largest_area = a

    # print(largest)

    return cv2.approxPolyDP(largest, .5, closed=True)



def refine_contour(img, ct, size=1024, offset=None):
    # img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    # Initialization of the level-set.

    h, w = img.shape[0], img.shape[1]

    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask.fill(0)
    mask = cv2.drawContours(mask, [ct], -1, (255,), -1, cv2.LINE_8, offset=offset)


    img = cv2.resize(img, (size, size))


    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.erode(img, kernel=np.ones((3, 3), np.uint8))
    img = cv2.medianBlur(img, 5)
    cv2.imshow("display_pit", img)
    val, o = cv2.threshold(img,None, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    print (val)
    val, o = cv2.threshold(img, val + 40, 255,cv2.THRESH_BINARY_INV)

    o = cv2.resize(o, (w, h), interpolation=cv2.INTER_NEAREST)


    assert o.shape == (h, w)
    o[mask == 0] = 0
    cts, _ = cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                              offset=(-offset[0], -offset[1]))

    largest_area = 0
    largest = None
    for c in cts:
        a = cv2.contourArea(c)
        if a > largest_area:
            largest = c
            largest_area = a

    # print(largest)

    return cv2.approxPolyDP(largest, .5, closed=True)

# mask_generator = SamAutomaticMaskGenerator(sam)


with open(COCO_ANNOTS) as f:
    coco = json.load(f)

cv2.namedWindow("display_pit")

root_dir = os.path.dirname(COCO_ANNOTS)
image_file_map = {im["id"]: im["file_name"] for im in coco["images"]}
image_file_map_rev = {im["file_name"]:im["id"] for im in coco["images"]}

cache_fn = ""
for ann in coco["annotations"]:
    im_filename = image_file_map[ann["image_id"]]
    if image_file_map_rev[im_filename] < 31:
        print(f"skipping {im_filename}")
        continue
    # todo we can cache here
    if im_filename != cache_fn:
        im = cv2.imread(os.path.join(root_dir, im_filename))
        cache_fn = im_filename
    x, y, w, h = ann["bbox"]
    if x < 0 or y <0:
        del ann
        continue
    bbox = floor(x), floor(y), ceil(x + w), ceil(y + h)

    roi = cv2.cvtColor(im[bbox[1]: bbox[3], bbox[0]: bbox[2]], cv2.COLOR_BGR2RGB)
    # roi = cv2.medianBlur(roi, 5)

    if len(ann["segmentation"]) > 1:
        areas = []
        for s in ann["segmentation"]:
            c = np.round([s]).astype(np.int32)
            c = c.reshape((len(c[0]) // 2, 1, 2))
            if len(c) < 3:
                a=0
            else:
                a = cv2.contourArea(c)
            areas.append(a)
        seg = [ann["segmentation"][np.argmax(areas)]]
    else:
        seg = ann["segmentation"]
    contour = np.round(seg).astype(np.int32)
    contour = contour.reshape((len(contour[0]) // 2, 1, 2))
    x += 1
    y +=1
    if w > 40 and h > 40:
        better_ct = refine_contour(roi, contour, offset=(-int(x-1), -int(y-1)))
        if better_ct is not None:
            candidate_contour = better_ct
            bb = cv2.boundingRect(contour)

            if abs(bb[2] - w) / w < .25  and abs(bb[3] - h)/h < .25:
                contour = candidate_contour
                cv2.drawContours(roi, [contour], -1, (255, 0, 0), 2, lineType=cv2.LINE_AA, offset=(-int(x), -int(y)))

                ann["segmentation"] = [contour.flatten().tolist()]
                ann["refined"] = True
            else:
                cv2.drawContours(roi, [contour], -1, (0, 0, 255), 2, lineType=cv2.LINE_AA, offset=(-int(x), -int(y)))
                ann["refined"] = False
            cv2.imshow("display3", roi)
            cv2.waitKey(1)
        # refine_mask(roi, contour, offset=(-int(x),-int(y)))
    assert im is not None

with open(COCO_ANNOTS_REFINED, 'w') as f:
    json.dump(coco, f)


# mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
# detections = sv.Detections.from_sam(result)
# annotated_image = mask_annotator.annotate(image_bgr, detections)
#

# cv2.imwrite("/tmp/test.png", best_mask.astype(np.uint8) * 255)
