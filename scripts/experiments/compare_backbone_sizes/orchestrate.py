import os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scripts.experiments.experiment_helpers import EXEC_DIR, DATADIR, set_default_config, get_config, do_yolo_train_run, clean_temporary_dir

BASE_NAME = "fb_compare_backbone_sizes"
BASE_PATH = os.path.join(EXEC_DIR, "scripts", "experiments", "compare_backbone_sizes")

set_default_config(os.path.join(BASE_PATH, "default.yaml"))

if __name__ == "__main__":
    args_parse = argparse.ArgumentParser()
    args_parse.add_argument("--dry-run", action="store_true", help="Print the experiment configurations without running them.")
    args = args_parse.parse_args()
    dry_run = args.dry_run

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

    gpu = [0, 1]

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
        do_yolo_train_run(config, dry_run=dry_run)

    # Clean up the temporary directory
    clean_temporary_dir()
    
    print(f"All (n={len(experiment_configs)}) experiments completed successfully.")

    
