#!/usr/bin/env python3

import argparse
import logging
import os
import glob
from flat_bug.predictor import Predictor
import torch
import uuid

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
    args_parse.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Which device to use for inference. Default is 'cuda:0', i.e. the first GPU.")
    args_parse.add_argument("-d", "--dtype", type=str, default="float16", help="Which dtype to use for inference. Default is 'float16'.")

    args = args_parse.parse_args()
    option_dict = vars(args)

    assert os.path.isdir(option_dict["input_dir"])
    assert os.path.isfile(option_dict["model_weights"])

    device = torch.device(option_dict["gpu"])
    dtype = getattr(torch, option_dict["dtype"])
    if not torch.cuda.is_available() and "cuda" in option_dict["gpu"]:
        raise ValueError(f"Device '{option_dict['gpu']}' is not available.")
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise ValueError(f"Dtype '{option_dict['dtype']}' is not supported.")

    pred = Predictor(option_dict["model_weights"], device=device, dtype=dtype)

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
            # Run the model
            prediction = pred.pyramid_predictions(f, scale_increment=1/2, scale_before=option_dict["scale_before"])
            # Save the results
            result_directory = prediction.save(
                output_directory = option_dict["results_dir"],
                mask_crops = True,
                identifier = str(uuid.uuid4()),
            )
        except Exception as e:
            logging.error(f"Issue whilst processing {f}")
            #fixme, what is going on with /home/quentin/todo/toup/20221008_16-01-04-226084_raw_jpg.rf.0b8d397da3c47408694eeaab2cde06e5.jpg?
            logging.error(e)
            raise e