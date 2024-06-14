import os, sys
import glob, argparse, re
import subprocess
import csv, yaml

from typing import Union, Optional, List, Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.experiments.experiment_helpers import run_command, remove_directory, split_by_sample, parse_unknown_arguments, read_slurm_params, ExperimentRunner
from flat_bug.datasets import get_datasets
from flat_bug.eval_utils import pretty_print_csv

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

def eval_model(
        weights : str, 
        config : Optional[str], 
        directory : str, 
        output_directory : str, 
        local_directory : str=None, 
        device : Optional[str]=None, 
        pattern : Optional[str]=None, 
        store_all : bool=False, 
        dry_run : bool = False, 
        execute : bool=True
    ) -> str:
    """
    Evaluate a model on a dataset.

    Args:
        weights (str): The path to the weights file.
        config (str): The path to the config file. Should be a YAML file either ending in '.yaml' or '.yml'.
        directory (str): The directory where the data is located and where the results will be saved the directory should have a 'reference' directory with the ground truth json in 'instances_default.json' and the matching images'.
        output_directory (str): The path to the output directory where the results will be saved. If not supplied, it is assumed to be the same as the directory. Defaults to None.
        local_directory (str, optional): The path to the local directory where the ground truth json is located. If not supplied, it is assumed to be the same as the directory. Defaults to None.
        device (str, optional): The PyTorch device string to use for inference. If not supplied, it is assumed to be cuda:0. Defaults to None.
        pattern (str, optional): The regex pattern to use for selecting the inference files. If not supplied, it is assumed to be the default pattern. Defaults to None.
        store_all (bool, optional): If set, all results will be saved. Defaults to False.
        dry_run (bool, optional): If set, the evaluation will not be run. If the command would be executed it is printed instead. Defaults to False.
        execute (bool, optional): If set, the evaluation will be run, otherwise the command will be returned. Defaults to True.

    Returns:
        str: The (executed) command (to run).
    """

    # Check if the weights file exists
    assert os.path.exists(weights), f"Weights file not found: {weights}"
    if config is not None:
        # Check if the config file exists
        assert os.path.exists(config), f"Config file not found: {config}"
    # Check if the directory exists
    assert os.path.exists(directory) and os.path.isdir(directory), f"Data directory not found: {directory}"
    # Check if the output directory exists
    # assert os.path.exists(output_directory) and os.path.isdir(output_directory), f"Output directory not found: {output_directory}"
    # Check if the local directory exists
    if local_directory is not None:
        # Check if the local directory exists
        assert os.path.exists(local_directory), f"Local directory not found: {local_directory}"

    # Create the command
    command = f'bash {os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval","end_to_end_eval.sh")} -w "{weights}" -d "{directory}" -o "{output_directory}"'
    if config is not None:
        command += f' -c "{config}"'
    if local_directory is not None:
        command += f' -l "{local_directory}"'
    if device is not None:
        command += f' -g "{device}"'
    if pattern is not None:
        command += f' -p "{pattern}"'
    
    # Run the command
    if execute:
        if dry_run:
            print(command)
        else:
            run_command(command)
        result_csv = os.path.join(output_directory, "results", "results.csv")
        if dry_run:
            print(f"Results would be saved to: {result_csv}")
        else:
            pretty_print_csv(result_csv, delimiter=",")
        if not store_all:
            # Remove the "<OUTPUT_DIR>/preds"
            pred_directory = os.path.join(output_directory, "preds")
            if dry_run:
                print(f"Would remove directory: {pred_directory}")
            else:
                remove_directory(pred_directory, recursive=True)
            # Save the first evaluation result for each dataset and remove the rest
            eval_files = glob.glob(os.path.join(output_directory, "eval", "*"))
            eval_datasets = get_datasets(eval_files)
            for dataset, files in eval_datasets.items():
                files = split_by_sample(files)
                for i, (sample, sample_files) in enumerate(files.items()):
                    if i == 0:
                        # Keep the plots and CSV for the first sample of each dataset
                        if dry_run:
                            print(f"Would keep the following files for dataset {dataset} and sample {sample}: {sample_files[0]}")
                            break
                        continue
                    for file in sample_files:
                        # Keep the combined results CSV
                        if os.path.basename(file) == "combined_results.csv":
                            continue
                        os.remove(file)
    return command
    
def eval_model_wrapper(
        params : dict, 
        execute : bool=True, 
        device : Optional[str]=None, 
        dry_run : Optional[bool]=None
    ) -> str:
    if device is not None:
        params.pop("device", None)
    params.pop("execute", None)
    if dry_run is not None and dry_run:
        print("Executing evaluation as dry run.")
    return eval_model(**params, execute=execute, device=device)

def get_weights_in_directory(directory : str) -> List[str]:
    """
    Get the weights file in the directory. The weights may be stored in arbitrarily nested subdirectories.

    Args:
        directory (str): The directory where weights are located.

    Returns:
        str: The path to the best weight file.
    """
    # Check if the directory exists
    assert os.path.exists(directory) and os.path.isdir(directory), f"Directory not found: {directory}"

    # Get all the weight files in the directory
    weight_dir = os.path.join(directory, "weights")
    if not os.path.exists(weight_dir):
        weight_dir = directory
    weight_files = glob.glob(os.path.join(weight_dir, "**", "*.pt"), recursive=True)
    # Check if there are any weight files
    assert len(weight_files) > 0, f"No weight files found in the directory: {weight_dir}"
    
    return sorted(weight_files, key=os.path.getmtime)

def get_gpus() -> List[str]:
    """
    Get the available GPUs.

    Returns:
        list: The available GPUs.
    """
    # Get the GPU information
    gpu_info = subprocess.Popen("nvidia-smi --query-gpu=index --format=csv,noheader,nounits", shell=True, stdout=subprocess.PIPE).stdout.read().decode("utf-8")
    # Get the GPUs
    gpus = [f"cuda:{int(gpu)}" for gpu in gpu_info.split("\n") if gpu != ""]
    return gpus

def combine_result_csvs(
        result_directories : List[str], 
        new_directory : str, 
        dry_run : bool=False
    ) -> str:
    """
    Combines the result CSVs in the result directories. The result CSVs are assumed to be in the 'results' subdirectory of the result directories and named 'results.csv'.

    The function simply creates a new csv file in the new directory with the combined results. The combined results contains all the rows from all the result CSVs, with a new column added: 'model'.

    The 'model' column is populated with the name of the model directory.
    
    Args:
        result_directories (list of str): A list of result directories.
        new_directory (str): The directory where the combined results will be saved.

    Returns:
        str: The path to the combined results CSV.
    """
    # Check that the new directory exists
    if not os.path.exists(new_directory) and os.path.isdir(new_directory):
        raise FileNotFoundError(new_directory)
    # Check that there are result directories
    if not len(result_directories) > 0:
        raise ValueError("No result directories provided.")
    # Get the path to the new result CSV
    new_result_csv = os.path.join(new_directory, "results.csv")
    # If it is a dry run, return the path to the new result CSV
    if dry_run:
        return new_result_csv
    # Check if the new result CSV exists and delete it if it does
    if os.path.exists(new_result_csv):
        os.remove(new_result_csv)
    # Combine the result CSVs into the new result CSV with the following additional columns from the metadata file:
    new_column_names = ['model', 'commit', 'time']
    with open(new_result_csv, 'w', newline='') as new_file:
        csv_writer = None
        for result_directory in result_directories:
            result_csv_path = os.path.join(result_directory, "results", "results.csv")
            metadata_path = os.path.join(result_directory, "metadata.yml")
            
            if not os.path.exists(result_csv_path):
                raise FileNotFoundError(f"Result CSV not found: {result_csv_path}")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as file:
                metadata = yaml.safe_load(file)
                if metadata is None:
                    raise ValueError(f"Metadata file is empty: {metadata_path}")
                for column in new_column_names:
                    if column not in metadata:
                        raise ValueError(f"Metadata file does not contain the '{column}' key: {metadata_path}")
                new_column_data = [metadata[column] for column in new_column_names]
            
            with open(result_csv_path, 'r') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader)
                if csv_writer is None:
                    csv_writer = csv.writer(new_file)
                    csv_writer.writerow(new_column_names + headers)
                
                for row in csv_reader:
                    csv_writer.writerow(new_column_data + row)
    # Return the path to the new result CSV
    return new_result_csv

class DeferredCall:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)

    def __str__(self):
        args_repr = [repr(arg) for arg in self.args]
        kwargs_repr = [f"{k}={repr(v)}" for k, v in self.kwargs.items()]
        return f"{self.__class__.__name__}: `{self.func.__name__}({', '.join(args_repr + kwargs_repr)})`"

    def __repr__(self):
        return str(self)

def combine_result_csvs_wrapper(args : Union[List, Tuple, Dict], execute : bool=True, device : Any=None, **kwargs) -> str:
    if isinstance(args, (list, tuple)):
        deferred_call = DeferredCall(combine_result_csvs, *args)
    elif isinstance(args, dict):
        deferred_call = DeferredCall(combine_result_csvs, **args)
    else:
        raise TypeError(f"Expected args to be a list, tuple or dict, not {type(args)}.")
    if execute:
        return deferred_call()
    else:
        return deferred_call

if __name__ == "__main__":
    
    arg_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parse.add_argument("-d", "--directory", dest="directory", help="A directory or glob to directories containing separate subdirectories for each model to evaluate.", type=str, required=True)
    arg_parse.add_argument("-g", "--ground_truth", dest="ground_truth", help="The path to the ground truth file.", type=str, required=True)
    arg_parse.add_argument("-i", "--input", dest="input", help="A directory containing the data to evaluate the models on.", type=str, required=True)
    arg_parse.add_argument("-o", "--output", dest="output", help="Override the directory where the results will be saved.", type=str, required=False)
    arg_parse.add_argument("--input_pattern", dest="input_pattern", help="The pattern to use for selecting the input files. Defaults to selecting all files.", type=str)
    arg_parse.add_argument("--weight_pattern", dest="weight_pattern", help="The pattern to use for selecting the weight files. Defaults to 'best' when all_weights is not set, otherwise all files are selected.", type=str)
    arg_parse.add_argument("--config", dest="config", help="The path to the config file.", type=str, required=False)
    arg_parse.add_argument("--all_weights", dest="all_weights", help="If set, all weights in the directory will be evaluated.", action="store_true")
    arg_parse.add_argument("--device", dest="device", help="The device to use for inference. If not set, the default device is used.", type=str, nargs="+", default="0")
    arg_parse.add_argument("--dry_run", dest="dry_run", help="If set, the evaluation will not be run.", action="store_true")
    arg_parse.add_argument("--save_all", dest="save_all", help="If set, all results will be saved.", action="store_true")
    arg_parse.add_argument("--ignore_existing", dest="ignore_existing", help="If set, existing result directories will be ignored.", action="store_true")
    arg_parse.add_argument("--name", dest="name", help="The name of comparison.", type=str, default="")
    arg_parse.add_argument("--slurm", dest="slurm", help="If set, the evaluation will be run on a SLURM cluster.", action="store_true")
    args, extra = arg_parse.parse_known_args()
    try:
        extra = parse_unknown_arguments(extra)
    except ValueError as e:
        raise ValueError(
                f"Error parsing extra arguments: `{' '.join(extra)}`. {e}\n\n"
                f"{arg_parse.format_help()}"
        )
    if not args.output is None:
        assert os.path.exists(args.output) and os.path.isdir(args.output), f'Output directory not found: {args.output}'
        RESULT_DIR = args.output 

    # Get the model director(y/ies)
    model_directories = glob.glob(args.directory)
    # Check if there are any model directories
    assert len(model_directories) > 0, "No model directories found."
    # Get the data directory
    data_directory = args.input
    # Check if the data directory exists
    assert os.path.exists(data_directory) and os.path.isdir(data_directory), f'Data directory not found: {data_directory}'

    # Get the config file
    config_file = args.config
    # Check if the config file exists
    if config_file is not None:
        assert os.path.exists(config_file), f'Config file not found: {config_file}'

    result_directories = {d : [] for d in model_directories}

    all_eval_params = []

    # Evaluate each model
    for model_directory in model_directories:
        # Get the best weight file
        all_weight_files = get_weights_in_directory(model_directory)
        if args.all_weights:
            num_before_pattern = len(all_weight_files)
            if args.weight_pattern is not None:
                all_weight_files = [file for file in all_weight_files if re.search(args.weight_pattern, file)]
            if len(all_weight_files) == 0:
                if num_before_pattern > 0:
                    pattern_message = f' | Perhaps the pattern ("{args.weight_pattern}") is too restrictive, {num_before_pattern} files found before pattern filtering.'
                else:
                    pattern_message = ''
                print(f'No weight files found in the directory: {model_directory}{pattern_message}')
                continue
            weight_ids = [os.path.basename(file).split(".")[0] for file in all_weight_files]
        else:
            if args.weight_pattern is None:
                args.weight_pattern = "best"
            all_weight_files = [file for file in all_weight_files if re.search(args.weight_pattern, file)]
            # assert len(all_weight_files) == 1, f'Exactly one best weight file should be found. Found: {len(all_weight_files)}.'
            weight_ids = ["" for file in all_weight_files]

        for weight_file, id in zip(all_weight_files, weight_ids):
            # Get the result directory path for the current model and weight file
            this_result_dir = os.path.join(RESULT_DIR, args.name, os.path.basename(model_directory), id)
            # Remember the result directory for each model
            result_directories[model_directory].append(this_result_dir)
            # Check if the result directory exists and is not empty
            if os.path.exists(this_result_dir) and len(os.listdir(this_result_dir)) > 0:
                # If the ignore_existing flag is set, ignore the existing result directory and skip the evaluation
                if args.ignore_existing:
                    print(f'Ignoring existing result directory: {this_result_dir}')
                    continue
                else:
                    raise ValueError(f'Result directory already exists: {this_result_dir}')
            if not args.dry_run:
                # Create the result directory
                os.makedirs(this_result_dir, exist_ok=True)
            # Set the shared evaluation parameters
            eval_params = {
                "weights" : os.path.abspath(weight_file), 
                "directory" : os.path.abspath(data_directory), 
                "output_directory" : os.path.abspath(this_result_dir),
                "config" : os.path.abspath(config_file) if config_file is not None else config_file,
                "local_directory" : os.path.abspath(args.ground_truth),
                "pattern" : args.input_pattern,
                "dry_run" : args.dry_run,
                "store_all" : args.save_all
            }
            # Add the evaluation parameters to the producer-consumer queue for asynchronous evaluation
            all_eval_params.append(eval_params)
    
    all_result_directories = []
    [all_result_directories.extend(dirs) for dirs in result_directories.values()]

    if not "job_name" in extra:
        extra.update({"job_name" : f'compare_models{"_" if args.name else ""}{args.name}'})

    runner = ExperimentRunner(eval_model_wrapper, all_eval_params, devices=args.device, dry_run=args.dry_run, slurm=args.slurm, slurm_params=read_slurm_params(**extra))
    runner.run().complete()

    extra.update({"dependency" : f'afterok:{runner.slurm_job_id}', "cpus_per_task" : 1})
    finalizer = ExperimentRunner(combine_result_csvs_wrapper, [[all_result_directories, os.path.join(RESULT_DIR, args.name), args.dry_run]], devices=args.device, dry_run=args.dry_run, slurm=args.slurm, slurm_params=read_slurm_params(**extra))
    finalizer.run().complete()