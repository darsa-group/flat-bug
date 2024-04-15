import os, json, re
from argparse import ArgumentParser

from tqdm import tqdm
from glob import glob

import xml.etree.ElementTree as ET

from mask_refiner import AutoMaskRefiner

import torch

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-w", "--model_path", type=str, help="Path to the model snapshot")
    parser.add_argument("-i", "--data_dir", type=str, help="Directory containing the data")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the refined data")
    parser.add_argument("-n", "--num_images", type=int, default=-1, help="Number of images to process. Defaults to -1 (all images)")
    parser.add_argument("-p", "--pattern", type=str, default=".*", help="Pattern to match the images. Defaults to '.*', i.e. all images.")
    parser.add_argument("--format", default="PASCAL VOC", help="Format of the annotations. Defaults to 'PASCAL VOC'.")
    args = parser.parse_args()

    refiner = AutoMaskRefiner(args.model_path, dtype=torch.float16, device=torch.device("cuda:0"))

    dir = args.data_dir
    outpath = args.output_path
    n = args.num_images
    pattern = re.compile(args.pattern)

    data = {os.path.basename(d): [f for f in sorted(glob(d + "/*")) if re.search(pattern, os.path.splitext(f)[0])] for d in sorted(glob(dir + "/*"))}


    match args.format:
        case "PASCAL VOC":
            all_boxes = [[[int(e.text) for e in el.find("bndbox")] for el in ET.parse(xml_file).getroot() if el.tag == "object"] for xml_file in data["annots"]]
        case "DIOPSIS":
            all_boxes = [[item['shape'] for item in json.load(open(json_file))['annotations']] for json_file in data["annotations"]]
            def box_to_array(box):
                return [box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']]
            all_boxes = [[box_to_array(box) for box in boxes] for boxes in all_boxes]
        case _:
            raise ValueError(f"Unknown format: {args.format}")

    
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
