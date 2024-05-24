import os, sys, subprocess, glob, tempfile, argparse
import yaml, re

from typing import Optional, Union, List, Dict, Any, Callable, Iterable

import queue, threading, submitit

DATADIR = "<UNSET>"
DATASETS = [
    '00-prospective-ALUS-mixed', '00-prospective-InsectCV', '00-prospective-chavez2024', '00-prospective-crall2023', 
    '01-partial-AMI-traps', '01-partial-Diopsis', '01-partial-NHM-beetles-crops', '01-partial-abram2023', '01-partial-gernat2018', 
    'ALUS', 'AMI-traps', 'AMT', 'ArTaxOr', 
    'BIOSCAN', 'CollembolAI', 'DIRT', 'DiversityScanner', 
    'Mothitor', 'PeMaToEuroPep', 
    'amarathunga2022', 'anTraX',
    'biodiscover-arm', 'biodiversa-arm', 
    'cao2022', 'pinoy2023', 
    'pitfall', 
    'scanned-sticky-cards', 'sittinger2023', 'sticky-pi',
    'ubc-pitfall-traps', 'ubc-scanned-sticky-cards'
]
DEFAULT_CONFIG = "<UNSET>"

def set_datadir(datadir : str):
    global DATADIR
    if datadir == "<UNSET>":
        raise ValueError("Data directory cannot must be set to other than '<UNSET>'.")
    if not os.path.exists(datadir):
        raise ValueError(f"Data directory {datadir} does not exist.")
    DATADIR = datadir

def set_default_config(config : str):
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
    assert os.path.exists(DEFAULT_CONFIG), f"Default config file not found: {DEFAULT_CONFIG}"

def get_config():
    if DEFAULT_CONFIG == "<UNSET>":
        raise RuntimeError("The default config file has not been set. Use `experiment__helpers.set_default_config()` to set it.")
    with open(DEFAULT_CONFIG, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)

    return config

TQDM_DETECT_PATTERN = re.compile(r"([\d\.]+\s*|\s\?)\w+\/\w+(\]|, )")
TQDM_FINISHED_PATTERN = re.compile(r"(\| (\d+)\/\2 \[)")

def custom_print(s):
    if re.search(TQDM_DETECT_PATTERN, s) and not ("100%" in s and re.search(TQDM_FINISHED_PATTERN, s)):
            s = s.removeprefix("\n").removesuffix("\n") + "\r"
    sys.stdout.write(s)
    sys.stdout.flush()

def run_command(command, python_binary=None):
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    if python_binary is not None:
        environment["PATH"] = python_binary + ":" + environment["PATH"]
        environment["PYTHONPATH"] = python_binary
    output_buffer = ""
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment) as process:
        while True:
            output = process.stdout.read(1)
            if not output and process.poll() is not None:
                break
            if output:
                output_buffer += output
                if output == "\n":
                    custom_print(output_buffer)
                    output_buffer = ""
        process.wait()

def remove_directory(directory, recursive=False):
    """
    Safely removes a directory containing files, no nested directories.
    """
    if not os.path.exists(directory):
        return
    files_and_directories = glob.glob(os.path.join(directory, "*"))
    files = []
    # Check that no nested directories are present
    for file_or_dir in files_and_directories:
        if os.path.isdir(file_or_dir):
            if recursive:
                remove_directory(file_or_dir)
            else:
                raise ValueError(f"Directory contains nested directories: {directory}")
        else:
            files.append(file_or_dir)
    for file in files:
        os.remove(file)
    os.rmdir(directory)

SAMPLE_SANITIZE_PATTERN = re.compile(r"^[^_]+_(.+)(_heatmap|_matches|\.csv)")

def split_by_sample(files):
    """
    Splits the files by sample.
    """
    samples = {}
    for file in files:
        match = SAMPLE_SANITIZE_PATTERN.match(os.path.basename(file))
        if match:
            sample = match.group(1)
            if sample not in samples:
                samples[sample] = []
            samples[sample].append(file)
    return {k : sorted(v) for k, v in samples.items()}

TMP_DIR = "<UNSET>"

def get_temp_config_dir():
    global TMP_DIR
    if TMP_DIR == "<UNSET>":
        TMP_DIR = tempfile.mkdtemp(prefix="fb_tmp_experiment_configs_", dir=os.environ["HOME"])
    return TMP_DIR

def get_temp_config_path():
    return tempfile.NamedTemporaryFile(dir=get_temp_config_dir(), mode="w", delete=False).name

def clean_temporary_dir():
    global TMP_DIR
    if TMP_DIR != "<UNSET>":
        tmp_files = os.listdir(TMP_DIR)
        for tmp_file in tmp_files:
            os.remove(os.path.join(TMP_DIR, tmp_file))
        os.rmdir(TMP_DIR)
        TMP_DIR = "<UNSET>"

class HelpfulArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\n\n{self.prog}: error: {message}\n")

def get_cmd_args():
    """
    A simple wrapper for shared command line arguments and parsing between experiment orchestration scripts.

    Command line arguments:
        -i, --datadir: The directory containing the data.
        --dry-run: Print the experiment configurations without running them.
        --devices: The GPU(s) to use for the experiments.
        --slurm: Use SLURM for the experiments.
        -p, --partition: The SLURM partition to use. Must be specified when using SLURM.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    args_parse = HelpfulArgumentParser()
    args_parse.add_argument("-i", "--datadir", help="The directory containing the data.", required=True)
    args_parse.add_argument("--devices", "--device", nargs="+", help="The GPU(s) to use for the experiments.", default=0)
    args_parse.add_argument("--slurm", action="store_true", help="Use SLURM for the experiments.")
    args_parse.add_argument("-p", "--partition", help="The SLURM partition to use.")
    args_parse.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print the experiment configurations without running them.")
    args = args_parse.parse_args()
    if args.slurm and args.partition is None:
        args_parse.print_help()
        print()
        raise ValueError("SLURM partition must be specified when using SLURM.")
    set_datadir(args.datadir)
    return args

def read_slurm_params(partition : str, path : Optional[str] = None):
    """
    Simple wrapper to read SLURM parameters from a YAML file, or use the default SLURM parameters if not supplied.

    The partition must be specified, as it is not included in the default SLURM parameters due to it being cluster-specific.

    Args:
        partition (str): The SLURM partition to use.
        path (Optional[str]): The path to the SLURM parameters YAML file. Defaults to None.

    Returns:
        Dict[str, Any]: The SLURM parameters. The keys are prefixed with 'slurm_', necessary for the submitit executor.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "default_slurm_params.yaml")
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    params["partition"] = partition
    # As a reminder, shared/generic (non-prefixed) parameters are: {'name': <class 'str'>, 'timeout_min': <class 'int'>, 'mem_gb': <class 'float'>, 'nodes': <class 'int'>, 'cpus_per_task': <class 'int'>, 'gpus_per_node': <class 'int'>, 'tasks_per_node': <class 'int'>, 'stderr_to_stdout': <class 'bool'>}.
    # Change all non-shared parameters to have the 'slurm_' prefix
    shared_params = ['name', 'timeout_min', 'mem_gb', 'nodes', 'cpus_per_task', 'gpus_per_node', 'tasks_per_node', 'stderr_to_stdout']
    for key in list(params.keys()):
        if key not in shared_params:
            params[f"slurm_{key}"] = params.pop(key)
    return params

def do_yolo_train_run(config, dry_run : bool=False, execute : bool=True, device : Optional[Union[int, str]]=None):
    """
    Wrapper for conducting a Flat-Bug YOLO training run, with 
    """
    global DATADIR
    if DATADIR == "<UNSET>":
        raise RuntimeError("The data directory has not been set. Use `experiment_helpers.set_datadir(<path>)` to set it.")
    ITEMIZE = '\n  - '

    # Set/override the device if specified
    if device is not None:
        config["device"] = f"cuda:{device}" if isinstance(device, int) or device.isdigit() else device

    # The config file is written to a "temporary" directory, which can be cleaned once the commands have been executed. In the case of non-SLURM execution, this is done automatically if using the `ExperimentRunner` class.
    config_path = get_temp_config_path()
    with open(config_path, "w") as conf:
        yaml.dump(config, conf, default_flow_style=False, sort_keys=False)
    
    print(f"Running experiment: {config['name']} with config:{ITEMIZE + ITEMIZE.join([f'{k}: {v}' for k, v in config.items()])}")
    command = f'fb_train -c "{config_path}" -d "{DATADIR}"'
    if execute:
        if dry_run:
            print(command)
        else:
            run_command(command)
    return command


class ExperimentRunner:
    def __init__(self, experiment_fn : Callable = do_yolo_train_run, inputs : Iterable = [], devices : Optional[Union[List[Union[int, str]], int, str]] = None, slurm : bool=False, slurm_params : Optional[Dict[str, Any]] = None, **kwargs):
        """
        A class to handle running multiple experiments, either sequentially, in parallel on multiple GPUs, or on a SLURM cluster.

        Args:
            experiment_fn (Callable): The function to run the experiment. Must accept a single element from inputs and a dictionary of keyword arguments, as well as the argument `execute` that defaults to True, which determines whether the function should execute the command or just return a bash command string.
            inputs (Iterable): The inputs to the experiment function.
            devices (Optional[Union[List[Union[int, str]], int, str]]): The GPU(s) to use for the experiments. Defaults to None.
            slurm (bool): Whether to run the experiments on a SLURM cluster. Defaults to False.
            slurm_params (Optional[Dict[str, Any]]): The parameters to pass to the SLURM executor. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the experiment function.
        
        Methods:
            run: Runs the experiments.
            wait: Waits for the experiments to finish.

        Example:
        ```python
        from scripts.experiments.experiment_helpers import ExperimentRunner, run_command

        def test_fn(input, execute=True, **kwargs):
            cmd = f"echo 'Running experiment with input {input}'"
            if execute:
                run_command(cmd)
            return cmd

        runner = ExperimentRunner(experiment_fn=test_fn, inputs=[0,1,2], devices=[0, 1], slurm=False, slurm_params={})
        runner.run()
        runner.wait()
        ```
        """
        self.experiment_fn = experiment_fn
        self.kwargs = kwargs
        self.experiment_queue = queue.Queue()
        for input in inputs:
            self.experiment_queue.put(input)
        self._length = len(inputs)
        self.devices = devices
        self.slurm = slurm
        self.slurm_params = slurm_params
        if slurm:
            self.executor = submitit.AutoExecutor(folder=os.path.join(os.getcwd(), "slurm_logs"), slurm_max_num_timeout=0)
            self.executor.update_parameters(**slurm_params)

        # Initialize consumer/job lists
        self.consumer_threads : List[threading.Thread] = []
        self.slurm_jobs : List[submitit.Job] = []

        if self.slurm and self.devices:
            raise ValueError("Cannot use both slurm and GPUs for the experiments.")
        
    def __len__(self):
        return self._length

    @staticmethod
    def consumer_thread(fn : Callable, queue : queue.Queue, **kwargs):
        while not queue.empty():
            input = queue.get()
            fn(input, **kwargs)

    @staticmethod
    def pretty_parse_slurm_results(job : submitit.Job) -> str:
        max_lines = 5
        max_char_per_line = 100
        try:
            result = [str(r) for r in job.results() if r]
        except Exception as e:
            result = str(e)
        lines = result if isinstance(result, list) else result.split("\n")
        if len(lines) == 0:
            return "No output."
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines.append(f"... {len(lines) - max_lines} more lines ...")
        return "\n".join([line[:max_char_per_line] for line in lines])
        

    def run(self):
        # Check that the consumer threads and slurm jobs are empty
        assert not self.consumer_threads, "Consumer threads list is not empty."
        assert not self.slurm_jobs, "Slurm jobs list is not empty."
        if isinstance(self.devices, list) and len(self.devices) > 1:
            # For multiple GPUs, we use a producer-consumer model, with one consumer per GPU
            for device in self.devices:
                this_kwargs = self.kwargs.copy()
                this_kwargs["device"] = device
                thread = threading.Thread(target=self.consumer_thread, args=(self.experiment_fn, self.experiment_queue), kwargs=this_kwargs, daemon=True)
                thread.start()
                self.consumer_threads.append(thread)
        else:
            if isinstance(self.devices, list):
                if len(self.devices) == 1:
                    self.devices = self.devices[0]
                elif len(self.devices) == 0:
                    self.devices = None
                else:
                    raise ValueError(f"Invalid (number of) devices for sequential or slurm execution: {self.devices}")
            # Run the experiments sequentially
            if self.slurm:
                cmds = []
            while not self.experiment_queue.empty():
                input = self.experiment_queue.get()
                if self.slurm:
                    cmd = self.experiment_fn(input, execute=False, device=self.devices, **self.kwargs)
                    cmds.append(cmd)
                else:
                    self.experiment_fn(input, execute=True, device=self.devices, **self.kwargs)
            if self.slurm:
                if self.kwargs.get("dry_run", False):
                    self.slurm_jobs = self.executor.map_array(print, cmds)
                else:
                    self.slurm_jobs = self.executor.map_array(run_command, cmds)

    def wait(self):
        # Wait for the consumer threads to finish
        [self.consumer_threads.pop().join() for _ in range(len(self.consumer_threads))]
        # Wait for the slurm jobs to finish
        [print(f'Job {i} finished with:', self.pretty_parse_slurm_results(self.slurm_jobs.pop())) for i in range(len(self.slurm_jobs))]

    def complete(self):
        if self.slurm:
            # When using SLURM, the experiments are submitted as an array job and the script exits immediately, so we don't need to have a process alive for the duration of the experiments
            print(f"All (n={len(self)}) experiments submitted as SLURM array job.")
            print(f"Consider cleaning the temporary directory ({get_temp_config_dir()}) once the experiments are completed.")
        else:
            # Block until all experiments are completed
            self.wait()

            # Clean up the temporary directory
            clean_temporary_dir()
            
            print(f"All (n={len(self)}) experiments completed successfully.")
