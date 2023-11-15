import argparse
import logging
import os.path
import yaml

from flat_bug.trainers import MySegmentationTrainer
from ultralytics import settings

DEFAULT_CONF = {
    "batch": 6,
    "imgsz": 1024,
    # "model": "/home/quentin/repos/flat-bug-git/runs/segment/train24/weights/last.pt",
    "model": "yolov8m-seg.pt",
    "task": "detect",
    # "task": "segment",
    "epochs": 1000,
    "device": "cuda",
    "patience": 100,
    "lr0": 0.01,
    "lrf": 0.005,
    "workers": 4  # fixme
}

if __name__ == '__main__':
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-d", "--data-dir", dest="data_dir",
                            help="The directory containing the prepared data (i.e., the output of  `fb_prepare.py`",
                            type=str)

    args_parse.add_argument("-c", "--config-file", dest="config_file",
                            help="A YAML-formatted config file that overrides the default training meta-parameters",
                            default=None)

    args = args_parse.parse_args()
    option_dict = vars(args)

    assert os.path.isdir(option_dict["data_dir"])

    data = os.path.join(option_dict["data_dir"], "data.yaml")

    settings.update({'datasets_dir': option_dict["data_dir"]})

    overrides = DEFAULT_CONF

    if option_dict["config_file"]:
        with open(option_dict["config_file"]) as f:
            yaml_config = yaml.safe_load(f)
            overrides.update(yaml_config)

    overrides["data"] = data
    t = MySegmentationTrainer(overrides=overrides)
    t.train()
