import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import set_default_config, get_config, get_cmd_args, read_slurm_params, ExperimentRunner

BASE_NAME = "best_train"
BASE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(DEFAULT_CONFIG)

if __name__ == "__main__":
    args, extra = get_cmd_args()

    backbone_size = "m"
    backbone_path = f"./yolov8{backbone_size}-seg.pt"

    config = get_config()
    config["model"] = backbone_path
    config["name"] = f"{BASE_NAME}_{backbone_size}"

    if "cpus_per_task" in extra and extra["cpus_per_task"] >= config["workers"]:
        n_workers = extra["cpus_per_task"]
    else:
        n_workers = config["workers"]
        if "cpus_per_task" in extra:
            print(f"WARNING: Requested cpus_per_task ({extra['cpus_per_task']}) is less than the required number of workers ({n_workers}). Ignoring the cpus_per_task parameter and continuing with {n_workers} workers.")
    
    extra.update({"cpus_per_task" : n_workers})
    if not "job_name" in extra:
        extra.update({"job_name" : BASE_NAME})
    experiment_runner = ExperimentRunner(inputs=[config], devices=args.devices, dry_run=args.dry_run, slurm=args.slurm, slurm_params=read_slurm_params(args.partition, **extra))
    experiment_runner.run().complete()
    
