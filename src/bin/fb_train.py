#!/usr/bin/env python3
import argparse
import os.path
import yaml

from flat_bug.trainers import MySegmentationTrainer

import ultralytics.data.utils as ultralytics_data_utils 
import ultralytics.utils as ultralytics_utils
from pathlib import Path

# fixme, resume should continue on the same "run folder"
def main():
    DEFAULT_CONF = {
        "batch": 8,
        "imgsz": 1024,
        "model": "yolov8m-seg.pt",
        "task": "detect",
        # "task": "segment", #fixme why not segment?! RE: It is overwritten in the __init__ method of ultralytics.models.yolo.segment.train.SegmentationTrainer
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
        "cache": False
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


    args, extra = args_parse.parse_known_args()
    cli_overrides = {}
    for key, value in zip(extra[::2], extra[1::2]):
        if not key.startswith("--"):
            raise ValueError(f"Unknown argument: {key}\n" + args_parse.format_help())
        key = key.removeprefix("--")
        if not key in DEFAULT_CONF:
            raise ValueError(f"Unknown argument: {key}\n" + args_parse.format_help())
        if key.startswith("fb_"):
            raise ValueError(f"Options starting with 'fb_' should be specified in the config file, not as command line arguments")
        # fixme: probably unsafe...
        try:
            value = eval(value)
        except:
            pass

        cli_overrides[key] = value
        
    option_dict = vars(args)

    option_dict["data_dir"] = os.path.abspath(os.path.normpath(option_dict["data_dir"]))
    assert os.path.isdir(option_dict["data_dir"]), f'Directory {option_dict["data_dir"]} not found.'

    # I think this should be fixed by resolving the path before passing it to the trainer and setting DATASETS_DIR in the scope of ultralytics.data.utils
    # (see https://github.com/ultralytics/ultralytics/blob/588bbbe4aed122e3d24353856484148bc5ef05ad/ultralytics/data/utils.py#L301)
    # #fixme issue when providing new dataset path, sill using old one?! see when i used pollen data
    # settings.update({'datasets_dir': option_dict["data_dir"]})

    # Load default training parameters
    overrides = DEFAULT_CONF

    # Update with parameters from the config file
    if option_dict["config_file"]:
        with open(option_dict["config_file"]) as f:
            yaml_config = yaml.safe_load(f)
            overrides.update(yaml_config)
    
    # Update with cli overrides
    overrides.update(cli_overrides)

    # Update data directory and resume flag from the command line
    overrides["data"] = os.path.join(option_dict["data_dir"], "data.yaml")
    # OBS: This is a *very* cursed hack around the fact that ultralytics have decided that you cannot change the settings at runtime. 
    ultralytics_data_utils.DATASETS_DIR = Path(option_dict["data_dir"]) # We technically only need to change it here, but I'll change it both places for consistency
    ultralytics_utils.DATASETS_DIR = Path(option_dict["data_dir"])

    if option_dict["resume"]:
        assert os.path.isfile(overrides["model"]), f"Trying to resume from a model that does not seem to be a valid file: {overrides['model']}"
        overrides["resume"] = overrides["model"]
    else:
        overrides["resume"] = False

    # This is just a hack to fix this: https://github.com/pytorch/pytorch/issues/37377 - only relevant for DDP
    if isinstance(overrides["device"], (tuple, list)) :
        num_devices = len(overrides["device"])
    elif isinstance(overrides["device"], str):
        num_devices = len(overrides["device"].split(","))
    else:
        num_devices = 1 # Fixme: Is this a real case, or just a type error?
    if isinstance(overrides["device"], (tuple, list)) or (isinstance(overrides["device"], str) and len(overrides["device"].split(",")) > 1):
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        os.environ['OMP_NUM_THREADS'] = str(overrides["workers"])

    # Ensure that `~` is not interpreted literally in arguments
    for k in overrides:
        if isinstance(k, str) and k in ["model", "data", "project", "pretrained"]:
            overrides[k] = os.path.expanduser(overrides[k])

    # DEBUG
    print("#######################################################")
    print("OVERRIDES")
    print(overrides)
    print("#######################################################")

    # Instantiate trainer
    trainer = MySegmentationTrainer(overrides=overrides)

    if not option_dict["resume"]:
        trainer.start_epoch = 0
    trainer.train()

if __name__ == "__main__":
    main()