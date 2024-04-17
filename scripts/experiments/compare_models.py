import os, sys, glob, argparse, re, queue, threading, subprocess, csv

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.experiments.experiment_helpers import EXEC_DIR, run_command, remove_directory, split_by_sample
from flat_bug.datasets import get_datasets

RESULT_DIR = os.path.join(EXEC_DIR, "scripts", "experiments", "results")

def eval_model(weights, directory, output_directory, local_directory=None, device=None, pattern=None, dry_run=False, store_all=False):
    """
    Evaluate a model on a dataset.

    Args:
        weights (str): The path to the weights file.
        directory (str): The directory where the data is located and where the results will be saved the directory should have a 'reference' directory with the ground truth json in 'instances_default.json' and the matching images'.
        output_directory (str): The path to the output directory where the results will be saved. If not supplied, it is assumed to be the same as the directory. Defaults to None.
        local_directory (str, optional): The path to the local directory where the ground truth json is located. If not supplied, it is assumed to be the same as the directory. Defaults to None.
        device (str, optional): The PyTorch device string to use for inference. If not supplied, it is assumed to be cuda:0. Defaults to None.
        pattern (str, optional): The regex pattern to use for selecting the inference files. If not supplied, it is assumed to be the default pattern. Defaults to None.
    """

    # Check if the weights file exists
    assert os.path.exists(weights), f"Weights file not found: {weights}"
    # Check if the directory exists
    assert os.path.exists(directory) and os.path.isdir(directory), f"Data directory not found: {directory}"
    # Check if the output directory exists
    # assert os.path.exists(output_directory) and os.path.isdir(output_directory), f"Output directory not found: {output_directory}"
    # Check if the local directory exists
    if local_directory is not None:
        # Check if the local directory exists
        assert os.path.exists(local_directory), f"Local directory not found: {local_directory}"

    # Create the command
    command = f"bash {os.path.join(EXEC_DIR, 'scripts', 'eval', 'end_to_end_eval.sh')} -w {weights} -d {directory} -o {output_directory}"
    if local_directory is not None:
        command += f" -l {local_directory}"
    if device is not None:
        command += f" -g {device}"
    if pattern is not None:
        command += f" -p {pattern}"
    
    # Run the command
    if dry_run:
        print(command)
    else:
        run_command(command, "/home/altair/.conda/envs/test/bin/python")

    # Get the result CSV
    result_csv = os.path.join(output_directory, "results", "results.csv")
    # Pretty print the result CSV
    if dry_run:
        print(f'Summary results would be saved to: {result_csv}')
    else:
        df = pd.read_csv(result_csv)
        print(result_csv, ":\n", df)

    if not store_all:
        # Remove the "<OUTPUT_DIR>/preds"
        pred_directory = os.path.join(output_directory, "preds")
        if dry_run:
            print("Removing directory: ", pred_directory)
        else:
            remove_directory(pred_directory, recursive=True)
        # Save the first evaluation result for each dataset and remove the rest
        eval_files = glob.glob(os.path.join(output_directory, "eval", "*"))
        eval_datasets = get_datasets(eval_files)
        for dataset, files in eval_datasets.items():
            files = split_by_sample(files)
            for i, (sample, sample_files) in enumerate(files.items()):
                if i == 0:
                    if dry_run:
                        print(f"Keeping <{sample}>: ", sample_files)
                    continue
                if dry_run:
                    break
                for file in sample_files:
                    os.remove(file)

def get_weights_in_directory(directory):
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
    weight_files = glob.glob(os.path.join(directory, "**", "*.pt"), recursive=True)
    # Check if there are any weight files
    assert len(weight_files) > 0, f"No weight files found in the directory: {directory}"
    
    return sorted(weight_files, key=os.path.getmtime)

def get_gpus():
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

def combine_result_csvs(result_directories, new_directory, dry_run=False):
    """
    Combines the result CSVs in the result directories. The result CSVs are assumed to be in the 'results' subdirectory of the result directories and named 'results.csv'.

    The function simply creates a new csv file in the new directory with the combined results. The combined results contains all the rows from all the result CSVs, with a new column added: 'model'.

    The 'model' column is populated with the name of the model directory.
    
    Args:
        result_directories (list): A list of result directories.
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
    # Combine the result CSVs into the new result CSV
    with open(new_result_csv, 'w', newline='') as new_file:
        csv_writer = None
        for result_directory in result_directories:
            result_csv_path = os.path.join(result_directory, "results", "results.csv")
            if not os.path.exists(result_csv_path):
                raise FileNotFoundError(f"Result CSV not found: {result_csv_path}")
            
            with open(result_csv_path, 'r') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader)
                if csv_writer is None:
                    csv_writer = csv.writer(new_file)
                    csv_writer.writerow(['model'] + headers)
                
                model = os.path.basename(result_directory)
                for row in csv_reader:
                    csv_writer.writerow([model] + row)
    # Return the path to the new result CSV
    return new_result_csv


def process_queue_item(queue : "queue.Queue", device, eval_function):
    while True:
        if queue:
            # Retrieve the next item from the queue
            item = queue.get()
            if item is None:
                queue.put(None)
                break
            # Add the 'device' key with the GPU value
            item['device'] = device
            # Call the evaluation function with the modified item
            eval_function(**item)

if __name__ == "__main__":
    
    arg_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    arg_parse.add_argument("-d", "--directory", dest="directory", help="A directory or glob to directories containing separate subdirectories for each model to evaluate.", type=str)
    arg_parse.add_argument("-g", "--ground_truth", dest="ground_truth", help="The path to the ground truth file.", type=str)
    arg_parse.add_argument("-i", "--input", dest="input", help="A directory containing the data to evaluate the models on.", type=str)
    arg_parse.add_argument("--input_pattern", dest="input_pattern", help="The pattern to use for selecting the input files. Defaults to selecting all files.", type=str)
    arg_parse.add_argument("--weight_pattern", dest="weight_pattern", help="The pattern to use for selecting the weight files. Defaults to selecting only the best weight files.", type=str)
    arg_parse.add_argument("--all_weights", dest="all_weights", help="If set, all weights in the directory will be evaluated.", action="store_true")
    arg_parse.add_argument("--multi_gpu", dest="multi_gpu", help="If set, the evaluation will be done on multiple GPUs.", action="store_true")
    arg_parse.add_argument("--device", dest="device", help="The device to use for inference. If not set, the default device is used. Cannot be used with --multi_gpu.", type=str)
    arg_parse.add_argument("--dry_run", dest="dry_run", help="If set, the evaluation will not be run.", action="store_true")
    arg_parse.add_argument("--save_all", dest="save_all", help="If set, all results will be saved.", action="store_true")
    arg_parse.add_argument("--ignore_existing", dest="ignore_existing", help="If set, existing result directories will be ignored.", action="store_true")
    arg_parse.add_argument("--no_compile", dest="no_compile", help="If set, the results will not be compiled into a single CSV.", action="store_true")

    args = arg_parse.parse_args()

    if args.device is not None:
        assert not args.multi_gpu, "Cannot specify a device when using multi-gpu mode."

    # Get the model director(y/ies)
    model_directories = glob.glob(args.directory)
    # Check if there are any model directories
    assert len(model_directories) > 0, "No model directories found."
    # Get the data directory
    data_directory = args.input
    # Check if the data directory exists
    assert os.path.exists(data_directory) and os.path.isdir(data_directory), f'Data directory not found: {data_directory}'

    if args.multi_gpu:
        # Initialize a producer-consumer queue
        eval_queue = queue.Queue()
        # Get the available GPUs
        gpus = get_gpus()
        # Check if there are any GPUs
        assert len(gpus) > 0, "No GPUs found."
        # Initialize the evaluation consumer threads
        eval_threads = []
        for gpu in gpus:
            # Create a new thread targeting the process_queue_item function
            eval_thread = threading.Thread(target=process_queue_item, args=(eval_queue, gpu, eval_model), daemon=True)
            eval_threads.append(eval_thread)
            eval_thread.start()

    result_directories = {d : [] for d in model_directories}

    # Evaluate each model
    for model_directory in model_directories:
        # Get the best weight file
        all_weight_files = get_weights_in_directory(model_directory)
        if args.all_weights:
            all_weight_files = [file for file in all_weight_files if not ("best" in file or "last" in file)]
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
            all_weight_files = [file for file in all_weight_files if re.search("best", file)]
            assert len(all_weight_files) == 1, f'Exactly one best weight file should be found. Found: {len(all_weight_files)}.'
            weight_ids = [""]

        for weight_file, id in zip(all_weight_files, weight_ids):
            # Get the result directory path for the current model and weight file
            this_result_dir = os.path.join(RESULT_DIR, os.path.basename(model_directory), id)
            # Remember the result directory for each model
            result_directories[model_directory].append(this_result_dir)
            # Check if the result directory exists
            if os.path.exists(this_result_dir):
                # If the ignore_existing flag is set, ignore the existing result directory and skip the evaluation
                if args.ignore_existing:
                    print(f'Ignoring existing result directory: {this_result_dir}')
                    continue
                else:
                    raise ValueError(f'Result directory already exists: {this_result_dir}')
            if not args.dry_run:
                # Create the result directory
                os.makedirs(this_result_dir)
            # Set the shared evaluation parameters
            eval_params = {
                "weights" : weight_file, 
                "directory" : data_directory, 
                "output_directory" : this_result_dir,
                "local_directory" : args.ground_truth,
                "pattern" : args.input_pattern,
                "dry_run" : args.dry_run,
                "store_all" : args.save_all
            }
            if not args.multi_gpu:
                # For single GPU evaluation, set the device parameter
                eval_params["device"] = "cuda:0" if args.device is None else args.device,
                # Evaluate synchronously
                eval_model(**eval_params)
            else:
                # Add the evaluation parameters to the producer-consumer queue for asynchronous evaluation
                eval_queue.put(eval_params)
    
    # Wait for the evaluation threads to finish
    if args.multi_gpu:
        eval_queue.put(None)
        for eval_thread in eval_threads:
            eval_thread.join()

    if not args.no_compile:
        # Combine the results
        for model_directory in model_directories:
            this_result_dirs = result_directories[model_directory]
            if len(this_result_dirs) == 0:
                print(f'No results found for model: {model_directory}')
                continue
            new_result_csv = combine_result_csvs(this_result_dirs, os.path.join(RESULT_DIR, os.path.basename(model_directory)), dry_run=args.dry_run)
            print(f'Combined results saved to: {new_result_csv}')
