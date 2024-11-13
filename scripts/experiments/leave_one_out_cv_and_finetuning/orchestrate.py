
import os, sys, re
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import DATASETS, set_default_config, get_config, get_cmd_args, read_slurm_params, ExperimentRunner

from collections import OrderedDict

BASE_NAME = "fb_leave_one_out"
BASE_PATH = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(BASE_PATH, "default.yaml")

set_default_config(DEFAULT_CONFIG)

if __name__ == "__main__":
    args, extra = get_cmd_args(name = BASE_NAME)

    if "dependency" in extra:
        print("WARNING: Use of the --dependency flag for this script is slightly non-standard. The initial training array job will be run with the dependency, but the fine-tuning jobs will not, although they will themselves depend on the training array job.")

    # Filter out the prospective datasets
    relevant_datasets = [d for d in DATASETS if not re.search("00-prospective", d)]

    # Create the base configs for the full and leave-one-out experiments
    full_config = get_config()
    full_config["name"] = f"{BASE_NAME}_FULL"
    
    # We use order-preserving dictionaries to store the experiment configs,
    # ensuring that the experiments are run in the correct order
    experiment_configs = OrderedDict({full_config["name"]: full_config})
    for dataset in relevant_datasets:
        this_config = get_config()
        this_config["name"] = f"{BASE_NAME}_{dataset}"
        this_config["fb_exclude_datasets"] += [dataset]
        experiment_configs[dataset] = this_config

    # Create the fine-tuning configs
    fine_tuning_configs = OrderedDict()
    for name, config in experiment_configs.items():
        fine_tune_config = config.copy()
        fine_tune_config["name"] = f"{BASE_NAME}_fine_tune_{name}"
        fine_tune_config["model"] = os.path.join(fine_tune_config["project"], config["name"], "weights", "best.pt")
        fine_tune_config["fb_exclude_datasets"] = [d for d in fine_tune_config["fb_exclude_datasets"] if d != name]
        fine_tune_config["epochs"] = 10
        fine_tune_config["resume"] = True
        fine_tuning_configs[name] = fine_tune_config

    # First run the full training
    main_experiment_runner = ExperimentRunner(inputs=experiment_configs.values(), devices=args.devices, attempt_resume=args.try_resume, slurm=args.slurm, slurm_params=read_slurm_params(**extra), dry_run=args.dry_run)
    main_experiment_runner.run().complete()

    # Then run the fine-tuning training 
    main_experiment_jobid = main_experiment_runner.slurm_job_id
    if main_experiment_jobid:
        extra.update({"dependency" : f'afterok:{main_experiment_jobid}'})
    finetune_experiment_runner = ExperimentRunner(inputs=fine_tuning_configs.values(), devices=args.devices, attempt_resume=args.try_resume, slurm=args.slurm, slurm_params=read_slurm_params(**extra), dry_run=args.dry_run)
    finetune_experiment_runner.run().complete()
