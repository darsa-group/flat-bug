#!/usr/bin/env python3
"""``flatbug`` training script.

The ``flatbug`` training script uses a lightly modified YOLO training interface
(https://docs.ultralytics.com/modes/train/), with a few additional parameters.

See `scripts/experiments/best_train/default.yaml` for an example training config.

Training refuses to start from a flat-bug checkout with uncommitted changes or untracked files,
so that every set of weights can be traced to a commit (see ``flat_bug.manifest``). Each run
writes ``manifest.train.yaml`` and ``data_inventory.csv.gz`` into its run folder.

Usage:
    ``fb_train [-d DATA_DIR] [-c CONFIG_FILE] [-r] [--allow-dirty]``


Options:
    -h, --help            show this help message and exit
    -d DATA_DIR, --data-dir DATA_DIR
                        The directory containing the prepared data (i.e., the output of  `fb_prepare.py`
    -c CONFIG_FILE, --config-file CONFIG_FILE
                        A YAML-formatted config file that overrides the default training meta-parameters
    -r, --resume          resume training
    --allow-dirty         train even if the checkout has uncommitted changes or untracked files;
                          the changes are saved next to the manifest as code.diff
"""

import argparse
import os.path
from pathlib import Path

import ultralytics.data.utils as ultralytics_data_utils
import ultralytics.utils as ultralytics_utils
import yaml

from flat_bug import logger, manifest
from flat_bug.trainers import FlatBugSegmentationTrainer


# fixme, resume should continue on the same "run folder"
def main():  # noqa: D103
    DEFAULT_CONF = {
        "batch": 8,
        "imgsz": 1024,
        "model": "yolo26m-seg.pt",
        "task": "segment",
        "epochs": 5000,
        "device": "cuda",
        "patience": 500,
        "optimizer": "auto",
        "save_period": 5,
        # "optimizer": 'SGD',
        # "lr0": 0.01,
        # "lrf": 0.005,
        "name": None,
        "workers": 4,
        "fb_max_instances": 150,
        "fb_max_images": -1,
        "fb_custom_eval": False,
        "fb_custom_eval_num_images": -1,
        "fb_exclude_datasets": [],
        # Train even if the flat-bug checkout has uncommitted changes or untracked files (refused
        # by default). The changes are saved next to the training manifest as code.diff.
        "fb_allow_dirty": False,
        "cache": False,
    }
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parse.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir",
        help="The directory containing the prepared data (i.e., the output of  `fb_prepare.py`",
        type=str,
    )

    args_parse.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        help="A YAML-formatted config file that overrides the default training meta-parameters",
        default=None,
    )

    args_parse.add_argument("-r", "--resume", dest="resume", help="resume training", action="store_true")

    args_parse.add_argument(
        "--allow-dirty",
        dest="allow_dirty",
        action="store_true",
        help=(
            "Train even if the flat-bug checkout has uncommitted changes or untracked files.\n"
            "Refused by default, so that weights can be traced to a commit; when allowed, the changes\n"
            "are saved next to the training manifest as code.diff. Same as `fb_allow_dirty: true`."
        ),
    )

    args, extra = args_parse.parse_known_args()
    cli_overrides = {}
    for key, value in zip(extra[::2], extra[1::2]):
        if not key.startswith("--"):
            raise ValueError(f"Unknown argument: {key}\n" + args_parse.format_help())
        key = key.removeprefix("--")
        if key not in DEFAULT_CONF:
            raise ValueError(f"Unknown argument: {key}\n" + args_parse.format_help())
        if key.startswith("fb_"):
            raise ValueError(
                "Options starting with 'fb_' should be specified in the config file, not as command line arguments"
            )
        # fixme: probably unsafe...
        try:
            value = eval(value)
        except Exception:
            pass

        cli_overrides[key] = value
    print(cli_overrides)

    option_dict = vars(args)

    option_dict["data_dir"] = os.path.abspath(os.path.normpath(option_dict["data_dir"]))
    assert os.path.isdir(option_dict["data_dir"]), f"Directory {option_dict['data_dir']} not found."

    # I think this should be fixed by resolving the path before passing
    # it to the trainer and setting DATASETS_DIR in the scope of ultralytics.data.utils
    # (see https://github.com/ultralytics/ultralytics/blob/588bbbe4aed122e3d24353856484148bc5ef05ad/ultralytics/data/utils.py#L301)
    # #fixme issue when providing new dataset path, sill using old one?! see when i used pollen data
    # settings.update({'datasets_dir': option_dict["data_dir"]})

    # Load default training parameters
    if not option_dict["resume"]:
        overrides = DEFAULT_CONF
    else:
        overrides = {}

    # Update with parameters from the config file
    if option_dict["config_file"]:
        with open(option_dict["config_file"]) as f:
            yaml_config = yaml.safe_load(f)
            overrides.update(yaml_config)

    # Update with cli overrides
    overrides.update(cli_overrides)
    if option_dict["allow_dirty"]:
        overrides["fb_allow_dirty"] = True

    # Fail before any data loading or DDP start-up, not after.
    try:
        manifest.check_clean(bool(overrides.get("fb_allow_dirty", False)))
    except manifest.DirtyCheckoutError as e:
        raise SystemExit(str(e)) from None

    # Update data directory and resume flag from the command line
    overrides["data"] = os.path.join(option_dict["data_dir"], "data.yaml")
    # OBS: This is a *very* cursed hack around the fact that ultralytics
    # have decided that you cannot change the settings at runtime.
    # We technically only need to change it here, but I'll change it both places for consistency
    ultralytics_data_utils.DATASETS_DIR = Path(option_dict["data_dir"])
    ultralytics_utils.DATASETS_DIR = Path(option_dict["data_dir"])

    if option_dict["resume"]:
        assert os.path.isfile(overrides["model"]), (
            f"Trying to resume from a model that does not seem to be a valid file: {overrides['model']}"
        )
        overrides["resume"] = overrides["model"]
        if (old_optim := overrides.pop("optimizer", None)) is not None:
            logger.warning(
                f"Ignored optimizer '{old_optim}' - YOLO does not support changing the optimizer while training."
            )
    else:
        overrides["resume"] = False

    # ruff: disable[F841] - TODO: fixme, we don't actually support multiple DDP
    # This is just a hack to fix this: https://github.com/pytorch/pytorch/issues/37377 - only relevant for DDP
    if isinstance(overrides["device"], (tuple, list)):
        num_devices = len(overrides["device"])
    elif isinstance(overrides["device"], str):
        num_devices = len(overrides["device"].split(","))
    else:
        num_devices = 1  # Fixme: Is this a real case, or just a type error?
    # ruff: enable[F841]
    if isinstance(overrides["device"], (tuple, list)) or (
        isinstance(overrides["device"], str) and len(overrides["device"].split(",")) > 1
    ):
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        os.environ["OMP_NUM_THREADS"] = str(overrides["workers"])

    # Ensure that `~` is not interpreted literally in arguments
    for k in overrides:
        if isinstance(k, str) and k in ["model", "data", "project", "pretrained"]:
            overrides[k] = os.path.expanduser(overrides[k])

    logger.debug("#######################################################")
    logger.debug("OVERRIDES")
    logger.debug(overrides)
    logger.debug("#######################################################")

    # Instantiate trainer
    trainer = FlatBugSegmentationTrainer(overrides=overrides)
    trainer.fb_config_file = option_dict["config_file"]  # recorded verbatim in manifest.train.yaml

    if not option_dict["resume"]:
        trainer.start_epoch = 0
    trainer.train()


if __name__ == "__main__":
    main()
