import os.path

import numpy as np
from math import floor, ceil
import torch
import cv2

import json

#
# COMPLETE_COCO = "/home/quentin/Desktop/gernat2018/results-chopped/coco_dataset.json"
# OVERWRITE_WITH = "/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/gernat2018-chopped/instances_default.json"  # incomplete, but valid
#


#
#  DATASET=blair2020;  fb_predict.py -i /home/quentin/Desktop/flat-bug-sorted-data/pre-pro/${DATASET}  -w ~/Desktop/fb_2024-02-09_best.pt -o /home/quentin/Desktop/flat-bug-preannot/${DATASET}
# DATASET=
for DATASET in ["NHM-beetles-mini",
                "anTraX",
                "gernat2018",
                "au-scanned-sticky",
                "PeMaToEuroPep",
                "abram-2023-sticky-cards",
                "ALUS-mixed",
                "ALUS",
                ]:
    OVERWRITE_WITH = f"/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/{DATASET}/instances_default.json"  # incomplete, but valid
    COMPLETE_COCO = f"/home/quentin/Desktop/flat-bug-preannot/{DATASET}/coco_instances.json"
    #


    # COMPLETE_COCO = "/home/quentin/Desktop/biodiscover-prelim/coco_dataset.json"
    # OVERWRITE_WITH = "/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/biodiscover-arm/instances_default.json"  # incomplete, but valid
    #

    # COMPLETE_COCO = "/home/quentin/Desktop/dirt-prelim/coco_dataset.json"
    # OVERWRITE_WITH = "/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/DIRT/instances_default.json" # incomplete, but valid

    OUT_COCO = f"/tmp/instances_refined_{DATASET}.json"


    with open(COMPLETE_COCO) as f:
        coco = json.load(f)

    with open(OVERWRITE_WITH) as f:
        coco_overwrite = json.load(f)

    image_file_map = {im["id"]: im["file_name"] for im in coco["images"]}
    print(image_file_map)
    image_file_map_rev = {im["file_name"]: im["id"] for im in coco["images"]}
    print(image_file_map_rev)
    ims_to_overwrite = {im["id"]: im["file_name"] for im in coco_overwrite["images"]}
    ims_to_overwrite_rev = {im["file_name"]: im["id"] for im in coco_overwrite["images"]}

    print(ims_to_overwrite)
    a = []
    for ann in coco["annotations"]:
        im_filename = image_file_map[ann["image_id"]]

        if not im_filename in ims_to_overwrite_rev:
            a.append(ann)

            # print(f"overwriting {im_filename}")
    coco["annotations"] = a

    for ann in coco_overwrite["annotations"]:

        im_filename = ims_to_overwrite[ann["image_id"]]
        ann["image_id"] = image_file_map_rev[im_filename]
        ann["attributes"] = {"manual": True}
        coco["annotations"].append(ann)

    for i, ann in enumerate(coco["annotations"]):
        ann["id"] = i

    with open(OUT_COCO, 'w') as f:
        json.dump(coco, f)
