import yaml, os, sys

import tempfile

from flat_bug.datasets import get_datasets

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from scripts.experiments.experiment_helpers import EXEC_DIR, DATASETS, set_default_config, get_config, custom_print, run_command

BASE_NAME = "fb_compare_backbone_sizes"
BASE_PATH = os.path.join(EXEC_DIR, "scripts", "experiments", "compare_backbone_sizes")
DATADIR = os.path.join(EXEC_DIR, "dev", "fb_yolo")
TMP_DIR = "<UNSET>"

set_default_config(os.path.join(BASE_PATH, "default.yaml"))

def get_temp_config_path():
    global TMP_DIR
    if TMP_DIR == "<UNSET>":
        TMP_DIR = tempfile.mkdtemp()
    return os.path.join(TMP_DIR, "temp_experiment_yolo_config.yaml")

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
    assert os.path.exists(DATADIR) and os.path.isdir(DATADIR), f"Data directory not found: {DATADIR}"
    assert os.getcwd() == EXEC_DIR, f"Current working directory ({os.getcwd()}) is not the execution directory: {EXEC_DIR}"

    # Filter out the prospective datasets
    backbone_sizes = ["L", "M", "S", "N"]
    backbone_paths = {
        "L" : "./yolov8l-seg.pt",
        "M" : "./yolov8m-seg.pt",
        "S" : "./yolov8s-seg.pt",
        "N" : "./yolov8n-seg.pt"
    }

    gpu = 1

    experiment_configs = dict()
    for size in backbone_sizes:
        this_config = get_config()
        this_config["name"] = f"{BASE_NAME}_{size}"
        this_config["model"] = backbone_paths[size]
        this_config["device"] = f"cuda:{gpu}"
        experiment_configs[size] = this_config

    for name, config in experiment_configs.items():
        print(f"Running experiment: {name}")
        print("Experiment config:", config)
        do_yolo_train_run(config)

    # Clean up the temporary directory
    if TMP_DIR != "<UNSET>":
        tmp_files = os.listdir(TMP_DIR)
        for tmp_file in tmp_files:
            os.remove(os.path.join(TMP_DIR, tmp_file))
        os.rmdir(TMP_DIR)
        TMP_DIR = "<UNSET>"

    print("All experiments completed.")

    
