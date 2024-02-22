import os
import argparse
from glob import glob

args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
args_parse.add_argument("--dir", help="The directory to prepare for end-to-end evaluation. You should probably not keep other stuff inside this directory.")
args_parse.add_argument("--id", type=str, default="", help="An identifier for the evaluation. Default is an empty string.")

if __name__ == "__main__":
    args = args_parse.parse_args()
    option_dict = vars(args)
    directory = option_dict["dir"]
    reference_directory = os.path.join(directory, "reference")
    assert os.path.isdir(directory), f"Root evaluation directory '{directory}' does not exist."
    assert os.path.isdir(reference_directory), f"Reference directory '{reference_directory}' does not exist."

    gt_directory = os.path.join(reference_directory, "gt")
    gt_file = os.path.join(gt_directory, "instances_default.json")
    assert os.path.isdir(gt_directory), f"Ground truth directory '{gt_directory}' does not exist."
    assert os.path.isfile(gt_file), f"Ground truth file '{gt_file}' does not exist."

    validation_directory = os.path.join(reference_directory, "val")
    assert os.path.isdir(validation_directory), f"Validation directory '{validation_directory}' does not exist."
    validation_files = glob(os.path.join(validation_directory, "**.jpg"))
    assert len(validation_files) > 0, f"No validation files found in '{validation_directory}'."
    print(f"Found {len(validation_files)} validation files.")

    results_directory = os.path.join(directory, "results")
    if not os.path.isdir(results_directory):
        os.makedirs(results_directory)
        print(f"Created results directory '{results_directory}'.")
    else:
        result_files = ["precision.png", "recall.png", "results.csv"]
        for f in result_files:
            f = os.path.join(results_directory, option_dict["id"] + f)
            if os.path.isfile(f):
                print(f"Warning: File '{f}' already exists. It will be overwritten.")

    output_directory = os.path.join(directory, "output")
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory '{output_directory}'.")
    else:
        # Clear the output directory
        for d in os.listdir(output_directory):
            for f in glob(os.path.join(output_directory, d, "**")):
                os.remove(f)
            os.removedirs(os.path.join(output_directory, d))
        os.mkdir(output_directory)
        print(f"Cleared output directory '{output_directory}'.")

    eval_directory = os.path.join(directory, "eval")
    if not os.path.isdir(eval_directory):
        os.makedirs(eval_directory)
        print(f"Created eval directory '{eval_directory}'.")
    else:
        # Clear the eval directory
        for f in glob(os.path.join(eval_directory, "**")):
            os.remove(f)
        print(f"Cleared eval directory '{eval_directory}'.")