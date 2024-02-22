import os
import argparse
from glob import glob

args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
args_parse.add_argument("--dir", help="The directory to prepare for end-to-end evaluation. You should probably not keep other stuff inside this directory.")
args_parse.add_argument("--clear-all", action="store_true", help="Clear all files and directories in the output and eval directories.")

if __name__ == "__main__":
    args = args_parse.parse_args()
    option_dict = vars(args)
    directory = option_dict["dir"]
    assert os.path.isdir(directory), f"Root evaluation directory '{directory}' does not exist."

    if option_dict["clear_all"]:
        output_directory = os.path.join(directory, "output")
        if not os.path.isdir(output_directory):
            print(f"No output directory to clean up at '{output_directory}'.")
        else:
            # Clear the output directory
            for d in os.listdir(output_directory):
                for f in glob(os.path.join(output_directory, d, "**")):
                    os.remove(f)
                os.removedirs(os.path.join(output_directory, d))
            print(f"Removed output directory '{output_directory}'.")

        eval_directory = os.path.join(directory, "eval")
        if not os.path.isdir(eval_directory):
            print(f"No eval directory to clean up at '{eval_directory}'.")
        else:
            # Clear the eval directory
            for f in glob(os.path.join(eval_directory, "**")):
                os.remove(f)
            os.removedirs(eval_directory)
            print(f"Removed eval directory '{eval_directory}'.")