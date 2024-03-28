import os
import json
from argparse import ArgumentParser

from tqdm import tqdm
from xml.etree import ElementTree as ET
from glob import glob

from mask_refiner import AutoMaskRefiner

import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-w", "--model_path", type=str, help="Path to the model snapshot")
    parser.add_argument("-i", "--data_dir", type=str, help="Directory containing the data")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the refined data")
    parser.add_argument("-n", "--num_images", type=int, default=-1, help="Number of images to process. Defaults to -1 (all images)")
    args = parser.parse_args()

    refiner = AutoMaskRefiner(args.model_path, dtype=torch.float16, device=torch.device("cuda:0"))

    dir = args.data_dir
    outpath = args.output_path
    n = args.num_images

    data = {os.path.basename(f): sorted(glob(f + "/*")) for f in sorted(glob(dir + "/*"))}

    all_boxes = [[[int(e.text) for e in el.find("bndbox")] for el in ET.parse(xml).getroot() if el.tag == "object"] for xml in data["annots"]]
    all_boxes = [torch.tensor(boxes).reshape(-1, 2, 2) for boxes in all_boxes]
    images = data["images"]
    if n > 0:
        all_boxes = all_boxes[:n]
        images = images[:n]
    coco = None

    pbar = tqdm(enumerate(zip(all_boxes, images)), total=len(images), desc="Refining masks", unit="image", dynamic_ncols=True)
    for i, (boxes, image) in pbar:
        res = refiner.refine_contours(boxes, image, progress=True)
        coco = refiner.to_coco(*res, image, coco)

    with open(outpath, "w") as f:
        json.dump(coco, f)
