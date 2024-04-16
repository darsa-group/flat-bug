import yaml, os, re, subprocess, sys

import yaml.serializer

from flat_bug.datasets import get_datasets

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from scripts.experiments.experiment_helpers import EXEC_DIR, DATASETS, set_default_config, get_config, custom_print, run_command

BASE_NAME = "fb_leave_one_out"
BASE_PATH = os.path.join(EXEC_DIR, "scripts", "experiments", "leave_one_out_cv_and_finetuning")
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")
DATADIR = os.path.join(EXEC_DIR, "dev", "fb_yolo")

set_default_config(DEFAULT_CONFIG)

def get_temp_config_path():
    return os.path.join(BASE_PATH, "temp_experiment_yolo_config.yaml")

def do_yolo_train_run(config):
    config_path = get_temp_config_path()
    if os.path.exists(config_path):
        os.remove(config_path)

    with open(config_path, "w") as conf:
        yaml.dump(config, conf, default_flow_style=False, sort_keys=False)
    
    command = f'/home/altair/.conda/envs/test/bin/python -u src/bin/fb_train.py -c "{config_path}" -d "{DATADIR}"'
    run_command(command)

    if os.path.exists(config_path):
        os.remove(config_path)

if __name__ == "__main__":
    # Check for the existence of the necessary files and directories, and the correct execution directory
    assert os.path.exists(DEFAULT_CONFIG), f"Default config file not found: {DEFAULT_CONFIG}"
    assert os.path.exists(DATADIR) and os.path.isdir(DATADIR), f"Data directory not found: {DATADIR}"
    assert os.getcwd() == EXEC_DIR, f"Current working directory ({os.getcwd()}) is not the execution directory: {EXEC_DIR}"

    # Filter out the prospective datasets
    relevant_datasets = [] # [d for d in DATASETS if not re.search("00-prospective", d)]

    full_config = get_config()
    full_config["name"] = f"{BASE_NAME}_FULL"
    experiment_configs = {"FULL" : full_config}
    for dataset in relevant_datasets:
        this_config = get_config()
        this_config["name"] = f"{BASE_NAME}_{dataset}"
        this_config["fb_exclude_datasets"] += [dataset]
        experiment_configs[dataset] = this_config

    for name, config in experiment_configs.items():
        print(f"Running experiment: {name}")
        print("Experiment config:", config)
        do_yolo_train_run(config)
