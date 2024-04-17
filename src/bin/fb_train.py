#!/usr/bin/env python3
import argparse
import logging
import os.path
import yaml

from flat_bug.trainers import MySegmentationTrainer
from ultralytics import settings

# fixme, resume should continue on the same "run folder"
if __name__ == '__main__':

    DEFAULT_CONF = {
        "batch": 8,
        "imgsz": 1024,
        "model": "yolov8m-seg.pt",
        "task": "detect",
        # "task": "segment", #fixme why not segment?!
        "epochs": 5000,
        "device": "cuda",
        "patience": 500,
        "optimizer": 'auto',
        "save_period": 5,
        # "optimizer": 'SGD',
        # "lr0": 0.01,
        # "lrf": 0.005,
        "workers": 4,
        "fb_max_instances": 150,
        "fb_max_images": -1,
        "fb_custom_eval": False,
        "fb_custom_eval_num_images": -1,
        "fb_exclude_datasets" : [],
        "cache": "ram"
    }
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-d", "--data-dir", dest="data_dir",
                            help="The directory containing the prepared data (i.e., the output of  `fb_prepare.py`",
                            type=str)

    args_parse.add_argument("-c", "--config-file", dest="config_file",
                            help="A YAML-formatted config file that overrides the default training meta-parameters",
                            default=None)
    args_parse.add_argument("-r", "--resume", dest="resume",
                            help="resume training",
                            action='store_true')


    args = args_parse.parse_args()
    option_dict = vars(args)

    assert os.path.isdir(option_dict["data_dir"])

    data = os.path.join(option_dict["data_dir"], "data.yaml")

    #fixme issue when providing new dataset path, sill using old one?! see when i used pollen data
    settings.update({'datasets_dir': option_dict["data_dir"]})

    overrides = DEFAULT_CONF

    if option_dict["config_file"]:
        with open(option_dict["config_file"]) as f:
            yaml_config = yaml.safe_load(f)
            overrides.update(yaml_config)

    overrides["data"] = data
    if option_dict["resume"]:
        assert os.path.isfile(overrides["model"]), f"Trying to resume from a model that does not seem to be a valid file: {overrides['model']}"
        overrides["resume"] = overrides["model"]
    else:
        overrides["resume"] = False

    # This is just a hack to fix this: https://github.com/pytorch/pytorch/issues/37377 - only relevant for DDP
    if isinstance(overrides["device"], (tuple, list)) or (isinstance(overrides["device"], str) and len(overrides["device"].split(",") > 1)):
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

    t = MySegmentationTrainer(overrides=overrides)

    if not option_dict["resume"]:
        t.start_epoch = 0
    t.train()
