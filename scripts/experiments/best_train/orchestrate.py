import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import set_default_config, get_config, get_cmd_args, read_slurm_params, ExperimentRunner

BASE_NAME = "best_train"
BASE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(DEFAULT_CONFIG)

if __name__ == "__main__":
    args = get_cmd_args()

    backbone_size = "m"
    backbone_path = f"./yolov8{backbone_size}-seg.pt"

    config = get_config()
    config["model"] = backbone_path
    config["name"] = f"{BASE_NAME}_{backbone_size}"

    experiment_runner = ExperimentRunner(inputs=[config], devices=args.devices, dry_run=args.dry_run, slurm=args.slurm, slurm_params=read_slurm_params(args.partition))
    experiment_runner.run().complete()
    
