
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import set_default_config, get_config, get_cmd_args, read_slurm_params, ExperimentRunner

from typing import List

from collections import OrderedDict

BASE_NAME = "fb_leave_two_out"
BASE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(DEFAULT_CONFIG)

def parse_include_datasets(path : str) -> List[str]:
    """
    Parses the include datasets file.
    """
    remove_comment = lambda line: line[:line.find("#")] if "#" in line else line
    with open(path, "r") as f:
        datasets = [remove_comment(line).strip() for line in f if not line.startswith("/")]
    return [dataset for dataset in datasets if dataset]

if __name__ == "__main__":
    args, extra = get_cmd_args()

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
            this_config["fb_exclude_datasets"].extend(list(set(this_excluded_datasets)))
            experiment_configs[this_config["name"]] = this_config

    experiment_runner = ExperimentRunner(inputs=experiment_configs.values(), devices=args.devices, dry_run=args.dry_run, slurm=args.slurm, slurm_params=read_slurm_params(args.partition, **extra))
    experiment_runner.run().complete()
