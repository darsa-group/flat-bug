import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import set_default_config, get_config, get_cmd_args, read_slurm_params, ExperimentRunner

BASE_NAME = "best_train"
BASE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(DEFAULT_CONFIG)

if __name__ == "__main__":
    args, extra = get_cmd_args(name = BASE_NAME)

    config = get_config()
    config["name"] = BASE_NAME

    experiment_runner = ExperimentRunner(inputs=[config], devices=args.devices, attempt_resume=args.try_resume, slurm=args.slurm, slurm_params=read_slurm_params(**extra), dry_run=args.dry_run)
    experiment_runner.run().complete()
    
