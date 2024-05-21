
import os, sys, re, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import EXEC_DIR, DATADIR, DATASETS, set_default_config, get_config, do_yolo_train_run, clean_temporary_dir

from typing import List

from collections import OrderedDict

BASE_NAME = "leave_two_out"
BASE_PATH = os.path.join(EXEC_DIR, "scripts", "experiments", "leave_two_out_dataset_mapping")
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(os.path.join(BASE_PATH, "default.yaml"))

def parse_include_datasets(path : str) -> List[str]:
    """
    Parses the include datasets file.
    """
    remove_comment = lambda line: line[:line.find("#")] if "#" in line else line
    with open(path, "r") as f:
        datasets = [remove_comment(line).strip() for line in f if not line.startswith("/")]
    return [dataset for dataset in datasets if dataset]

if __name__ == "__main__":
    args_parse = argparse.ArgumentParser()
    args_parse.add_argument("--dry-run", action="store_true", help="Print the experiment configurations without running them.")
    args = args_parse.parse_args()
    dry_run = args.dry_run

    # Check for the existence of the necessary files and directories, and the correct execution directory
    assert os.path.exists(DEFAULT_CONFIG), f"Default config file not found: {DEFAULT_CONFIG}"
    assert os.path.exists(DATADIR) and os.path.isdir(DATADIR), f"Data directory not found: {DATADIR}"
    assert os.getcwd() == EXEC_DIR, f"Current working directory ({os.getcwd()}) is not the execution directory: {EXEC_DIR}"

    # Filter out the prospective datasets
    relevant_datasets = parse_include_datasets(os.path.join(BASE_PATH, "include_datasets"))

    # Create the base configs for the full and leave-one-out experiments
    full_config = get_config()
    full_config["name"] = f"{BASE_NAME}_FULL"
    # We use order-preserving dictionaries to store the experiment configs,
    # ensuring that the experiments are run in the correct order
    experiment_configs = OrderedDict({"FULL" : full_config})
    for dataset_i in relevant_datasets:
        for dataset_j in relevant_datasets:
            this_excluded_datasets = sorted([dataset_i, dataset_j])
            this_config = get_config()
            this_config["name"] = "{}_{}_{}".format(BASE_NAME, *this_excluded_datasets)
            if this_config["name"] in experiment_configs:
                continue
            this_config["fb_exclude_datasets"].extend(this_excluded_datasets)
            experiment_configs[this_config["name"]] = this_config

    # Run the experiments
    for name, config in experiment_configs.items():
        print(f"Running experiment: {name}")
        print("Experiment config:", config)
        do_yolo_train_run(config, dry_run=dry_run)

    clean_temporary_dir()

    print(f"All (n={len(experiment_configs)}) experiments completed successfully.")
