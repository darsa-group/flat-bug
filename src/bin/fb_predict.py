#!/usr/bin/env python3

import argparse
import logging
import os
import glob
from flat_bug.predictor import Predictor
import json

if __name__ == '__main__':
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-i", "--input-data", dest="input_dir",
                            help="A directory that contains subdirectories for each COCO sub-datasets."
                                 "Each sub-dataset contains a single json file named 'instances_default.json' "
                                 "and the associated images"
                            )
    args_parse.add_argument("-o", "--output-dir", dest="results_dir",
                            help="The result directory")
    args_parse.add_argument("-w", "--model-weights", dest="model_weights",
                            help="The .pt file")
    args_parse.add_argument("-s", "--scale-before", dest="scale_before", default=1.0, type=float,
                            help="Downscale the image before detection, but crops from the original image."
                                 "Note that the COCO dataset dimentions match the scaled image" # fixme, is that true?!
                            )

    args = args_parse.parse_args()
    option_dict = vars(args)

    assert os.path.isdir(option_dict["input_dir"])
    assert os.path.isfile(option_dict["model_weights"])
    pred = Predictor(option_dict["model_weights"])

    # fixme, build from pred._model!
    categories = {"id": 1, "name": "insect"}

    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [categories]  # Your category
    }

    j = 1
    for i, f in enumerate(glob.glob(os.path.join(option_dict["input_dir"], "*.jpg"))):

        logging.info(f"Processing {os.path.basename(f)}")
        try:
            prediction = pred.pyramid_predictions(f, scale_before=option_dict["scale_before"])

            im_info, annots = prediction.coco_entry()
            im_info["id"] = i + 1
            for a in annots:
                a["id"] = j
                a["image_id"] = i + 1
                j += 1

            prediction.make_crops(out_dir=option_dict["results_dir"])
        except Exception as e:
            logging.error(f"Issue whilst processing {f}")
            #fixme, what is going on with /home/quentin/todo/toup/20221008_16-01-04-226084_raw_jpg.rf.0b8d397da3c47408694eeaab2cde06e5.jpg?
            logging.error(e)
            raise e
        coco_data["images"].append(im_info)
        coco_data["annotations"].extend(annots)
    with open(os.path.join(option_dict["results_dir"], "coco_dataset.json"), "w") as json_file:
        json.dump(coco_data, json_file)
