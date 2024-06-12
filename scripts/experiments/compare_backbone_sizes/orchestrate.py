import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import set_default_config, get_config, get_cmd_args, read_slurm_params, ExperimentRunner

BASE_NAME = "fb_compare_backbone_sizes"
BASE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(DEFAULT_CONFIG)

if __name__ == "__main__":
    args = get_cmd_args()

    backbone_sizes = ["L", "M", "S", "N"]
    backbone_paths = {
        "L" : "./yolov8l-seg.pt",
        "M" : "./yolov8m-seg.pt",
        "S" : "./yolov8s-seg.pt",
        "N" : "./yolov8n-seg.pt"
    }

    experiment_configs = []
    for size in backbone_sizes:
        this_config = get_config()
        this_config["name"] = f"{BASE_NAME}_{size}"
        this_config["model"] = backbone_paths[size]
        experiment_configs.append(this_config)

    experiment_runner = ExperimentRunner(inputs=experiment_configs.values(), devices=args.devices, dry_run=args.dry_run, slurm=args.slurm, slurm_params=read_slurm_params(args.partition))
    experiment_runner.run().complete()
    
