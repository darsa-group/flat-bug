import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import set_default_config, get_config, get_cmd_args, read_slurm_params, ExperimentRunner
from typing import List

BASE_NAME = "fb_compare_backbone_sizes"
BASE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(DEFAULT_CONFIG)

if __name__ == "__main__":
    args, extra = get_cmd_args(
        name = BASE_NAME,
        additional_args = [
            (["--sizes"], {"dest" : "sizes", "type" : str, "nargs" : "+", "default" : ["L", "M", "S", "N"], "help" : "The backbone size(s) to train. Choices are L/l, M/m, S/s and N/n."})
        ]
    )

    backbone_sizes = [s.upper() for s in args.sizes]
    backbone_paths = {
        "L" : "~/flat-bug/yolov8l-seg.pt",
        "M" : "~/flat-bug/yolov8m-seg.pt",
        "S" : "~/flat-bug/yolov8s-seg.pt",
        "N" : "~/flat-bug/yolov8n-seg.pt"
    }
    if any([s not in backbone_paths for s in backbone_sizes]):
        raise ValueError(f"Invalid model size(s) '{backbone_sizes}' expected some of '{list(backbone_paths.keys())}'")

    full_config = get_config()

    experiment_configs = []
    for size in backbone_sizes:
        this_config = full_config.copy()
        this_config["name"] = f"{BASE_NAME}_{size}"
        this_config["model"] = backbone_paths[size]
        experiment_configs.append(this_config)
    
    experiment_runner = ExperimentRunner(inputs=experiment_configs, devices=args.devices, attempt_resume=args.try_resume, slurm=args.slurm, slurm_params=read_slurm_params(**extra), dry_run=args.dry_run)
    experiment_runner.run().complete()
    
