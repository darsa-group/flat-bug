import os, sys, glob, argparse, re, queue, threading, subprocess, csv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.experiments.experiment_helpers import EXEC_DIR, run_command

RESULT_DIR = os.path.join(EXEC_DIR, "scripts", "experiments", "results")

# For eval_model we use a bash script called end_to_end_eval.sh located at EXEC_DIR/scripts/eval/end_to_end_eval.sh
# It has the following usage:
    # Usage: $0 -w weights -d directory [-l local_directory] [-o output_directory] [-g PyTorch_device_string] [-p inference_file_regex_pattern]
    #     -w weights (MANDATORY):
    #         The path to the weights file.

    #     -d directory (MANDATORY): 
    #         The directory where the data is located and where the results will be saved the directory
    #         should have a 'reference' directory with the ground truth json in
    #         'instances_default.json' and the matching images'.

    #     -l local_directory (OPTIONAL): 
    #         The path to the local directory where the ground truth json is located.
    #         If not supplied, it is assumed to be the same as the directory.

    #     -o output_directory (OPTIONAL): 
    #         The path to the output directory where the results will be saved.
    #         If not supplied, it is assumed to be the same as the directory.

    #     -g PyTorch_device_string (OPTIONAL): 
    #         The PyTorch device string to use for inference.
    #         If not supplied, it is assumed to be cuda:0.

    #     -p inference_file_regex_pattern (OPTIONAL): 
    #         The regex pattern to use for selecting the inference files.
    #         If not supplied, it is assumed to be the default pattern.

def eval_model(weights, directory, output_directory, local_directory=None, device=None, pattern=None):
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
        assert os.path.exists(local_directory) and os.path.isdir(local_directory), f"Local directory not found: {local_directory}"

    # Create the command
    command = f"bash {os.path.join(EXEC_DIR, 'scripts', 'eval', 'end_to_end_eval.sh')} -w {weights} -d {directory} -o {output_directory}"
    if local_directory is not None:
        command += f" -l {local_directory}"
    if device is not None:
        command += f" -g {device}"
    if pattern is not None:
        command += f" -p {pattern}"
    
    # Run the command
    print(command)
    # run_command(command, "/home/altair/.conda/envs/test/bin/python")

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

def combine_result_csvs(result_directories, new_directory):
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
    # Create the new result CSV
    new_result_csv = os.path.join(new_directory, "results.csv")
    if os.path.exists(new_result_csv):
        os.remove(new_result_csv)
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
    arg_parse.add_argument("-i", "--input", dest="input", help="A directory containing the data to evaluate the models on.", type=str)
    arg_parse.add_argument("-p", "--weight_pattern", dest="weight_pattern", help="The pattern to use for selecting the weight files. Defaults to selecting only the best weight files.", type=str)
    arg_parse.add_argument("--all_weights", dest="all_weights", help="If set, all weights in the directory will be evaluated.", action="store_true")
    arg_parse.add_argument("--multi_gpu", dest="multi_gpu", help="If set, the evaluation will be done on multiple GPUs.", action="store_true")

    args = arg_parse.parse_args()

    # Get the model director(y/ies)
    model_directories = glob.glob(args.directory)
    # Check if there are any model directories
    assert len(model_directories) > 0, "No model directories found."
    # Get the data directory
    data_directory = args.input
    # Check if the data directory exists
    assert os.path.exists(data_directory) and os.path.isdir(data_directory), f"Data directory not found: {data_directory}"

    if args.multi_gpu:
        eval_queue = queue.Queue()
        # Get the available GPUs
        gpus = get_gpus()
        # Check if there are any GPUs
        assert len(gpus) > 0, "No GPUs found."
        # Create the evaluation threads
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
            if args.weight_pattern is not None:
                all_weight_files = [file for file in all_weight_files if re.search(args.weight_pattern, file)]
            weight_ids = [os.path.basename(file).split(".")[0] for file in all_weight_files]
        else:
            all_weight_files = [file for file in all_weight_files if re.search("best", file)]
            assert len(all_weight_files) == 1, f"Exactly one best weight file should be found. Found: {len(all_weight_files)}."
            weight_ids = [""]

        for weight_file, id in zip(all_weight_files, weight_ids):
            # Evaluate
            this_result_dir = os.path.join(RESULT_DIR, os.path.basename(model_directory), id)
            # assert not os.path.exists(this_result_dir), f"Result directory already exists: {this_result_dir}"
            # os.makedirs(this_result_dir)
            result_directories[model_directory].append(this_result_dir)
            if not args.multi_gpu:
                eval_params = {
                    "weights" : weight_file, 
                    "directory" : data_directory, 
                    "output_directory" : this_result_dir,
                    "device" : "cuda:0"
                }
                eval_model(**eval_params)
            else:
                eval_params = {
                    "weights" : weight_file, 
                    "directory" : data_directory, 
                    "output_directory" : this_result_dir
                }
                eval_queue.put(eval_params)
    
    if args.multi_gpu:
        eval_queue.put(None)
        for eval_thread in eval_threads:
            eval_thread.join()

    # Combine the results
    for model_directory in model_directories:
        new_result_csv = combine_result_csvs(result_directories[model_directory], os.path.join(RESULT_DIR, os.path.basename(model_directory)))
        print(f"Combined results saved to: {new_result_csv}")
