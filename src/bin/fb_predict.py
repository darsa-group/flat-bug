#!/usr/bin/env python3

import argparse
import logging
import os
import glob

from flat_bug.coco_utils import fb_to_coco
from flat_bug.predictor import Predictor
import torch
import uuid
from tqdm import tqdm

if __name__ == '__main__':
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-i", "--input-data", dest="input_dir",
                            help="A directory that contains subdirectories for each COCO sub-datasets."
                                 "Each sub-dataset contains a single json file named 'instances_default.json' "
                                 "and the associated images"
                            )
    args_parse.add_argument("-p", "--input-pattern", dest="input_pattern", default="**.jpg",
                            help="The pattern to match the images. Default is '**.jpg'")
    args_parse.add_argument("-n", "--max-images", type=int, default=None, help="Maximum number of images to process. Default is None. Truncates in alphabetical order.")
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
    args_parse.add_argument("-f", "--fast", action="store_true", help="Use fast mode.")
    args_parse.add_argument("--no-crops", action="store_true", help="Do not save the crops.")
    args_parse.add_argument("--no-overviews", action="store_true", help="Do not save the overviews.")
    args_parse.add_argument("-S", "--no-save", action="store_true", help="Do not save the results.")
    args_parse.add_argument("-C", "--no-compiled-coco", action="store_true", help="Skip the production of a compiled COCO file (for all images).")
    args_parse.add_argument("--single-scale", action="store_true", help="Use single scale.")
    args_parse.add_argument("--verbose", action="store_true", help="Verbose mode.")
    

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
    pred.MIN_MAX_OBJ_SIZE = 16, 768 # Size is measured as the square root of the area
    pred.MAX_MASK_SIZE = 1024 # Loss of precision may occur if the mask is larger than this, but all shapes are possible. 
    pred.SCORE_THRESHOLD = 0.3
    pred.IOU_THRESHOLD = 0.15
    pred.MINIMUM_TILE_OVERLAP = 384
    pred.EDGE_CASE_MARGIN = 128 + 64
    pred.PREFER_POLYGONS = True # Convert masks to polygons as soon as possible, and only use the polygons for further processing - no loss of precision, but only single polygons without holes can be represented, performance impact may depend on hardware and use-case
    pred.EXPERIMENTAL_NMS_OPTIMIZATION = True
    pred.TIME = option_dict["verbose"] # Should be enabled with a verbose parameter, and maybe logged? Also not sure if it incurrs a performance penalty

    # # Legacy hyperparameters
    # pred.MIN_MAX_OBJ_SIZE = 16, 1024
    # pred.MINIMUM_TILE_OVERLAP = 256
    # pred.EDGE_CASE_MARGIN = 128
    # pred.SCORE_THRESHOLD = 0.5
    # # pred.IOU_THRESHOLD = 0.5
    # pred.PREFER_POLYGONS = True # This wasn't a hyperparameter before, but it reproduces the old behavior


    # fixme, build from pred._model!
    categories = {"id": 1, "name": "insect"}

    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [categories]  # Your category
    }

    files = sorted(glob.glob(os.path.join(option_dict["input_dir"], option_dict["input_pattern"])))
    if option_dict["max_images"] is not None:
        files = files[:option_dict["max_images"]]
    j = 1
    all_json_results = []
    pbar = tqdm(enumerate(files), total=len(files), desc="Processing images", dynamic_ncols=True, unit="image")
    for i, f in pbar:
        if option_dict["verbose"]:
            print(f"Processing {os.path.basename(f)}")
        pbar.set_postfix_str(f"Processing {os.path.basename(f)}")
        logging.info(f"Processing {os.path.basename(f)}")
        try:
            # Run the model
            prediction = pred.pyramid_predictions(f, scale_increment=1/2, scale_before=option_dict["scale_before"], single_scale=option_dict["single_scale"])
            # Save the results
            if not option_dict["no_save"]:
                result_directory = prediction.save(
                    output_directory = option_dict["results_dir"],
                    fast = option_dict["fast"],
                    overview = not option_dict["no_overviews"],
                    crops = not option_dict["no_crops"],
                    mask_crops = not option_dict["fast"],
                    identifier = "ChangeThisTEMPORARY", #str(uuid.uuid4()),
                )
                json_files = [f for f in  glob.glob(os.path.join(result_directory, "*.json"))]
                assert len(json_files) == 1
                all_json_results.append(json_files[0])
        except Exception as e:
            logging.error(f"Issue whilst processing {f}")
            #fixme, what is going on with /home/quentin/todo/toup/20221008_16-01-04-226084_raw_jpg.rf.0b8d397da3c47408694eeaab2cde06e5.jpg?
            logging.error(e)
            raise e
    if not option_dict["no_compiled_coco"]:
        import json

        compiled_coco = os.path.join(option_dict["results_dir"], "coco_instances.json")
        pred_coco = {}
        
        flat_bug_predictions = [json.load(open(p)) for p in all_json_results]
        for d in flat_bug_predictions:
            fb_to_coco(d, pred_coco)
        with open(compiled_coco,"w") as f:
            json.dump(pred_coco, f)

