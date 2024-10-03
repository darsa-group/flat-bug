import os, sys, subprocess, time, argparse
import glob, tempfile, re
import yaml, io, zipfile

from argparse import Namespace
from typing import Self, Optional, Union, List, Tuple, Dict, Any, Callable, Iterable, IO

import queue, threading, submitit

from submitit.slurm.slurm import _get_default_parameters as _default_submitit_slurm_params

DATA_DIR = "<UNSET>"
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
PROJECT_DIR = "<UNSET>"
OUTPUT_DIR = "<UNSET>"

def set_datadir(datadir : str, strict : bool=True):
    global DATA_DIR
    if datadir == "<UNSET>":
        raise ValueError("Data directory must be set to other than '<UNSET>'.")
    if not os.path.exists(datadir) and strict:
        raise ValueError(f"Data directory {datadir} does not exist.")
    DATA_DIR = datadir

def set_projectdir(projectdir : str):
    global PROJECT_DIR
    if projectdir == "<UNSET>":
        raise ValueError("Project directory must be set to other than '<UNSET>'.")
    if not os.path.exists(projectdir):
        os.makedirs(projectdir, exist_ok=True)
    PROJECT_DIR = projectdir

def set_default_config(config : str):
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
    assert os.path.exists(DEFAULT_CONFIG), f"Default config file not found: {DEFAULT_CONFIG}"

def get_config() -> Dict[str, Any]:
    if DEFAULT_CONFIG == "<UNSET>":
        raise RuntimeError("The default config file has not been set. Use `experiment__helpers.set_default_config()` to set it.")
    with open(DEFAULT_CONFIG, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    if PROJECT_DIR != "<UNSET>":
        config["project"] = PROJECT_DIR

    return config

TQDM_DETECT_PATTERN = re.compile(r"([\d\.]+\s*|\s\?)\w+\/\w+(\]|, )")
TQDM_FINISHED_PATTERN = re.compile(r"(\| (\d+)\/\2 \[)")

def custom_print(line : str):
    if re.search(TQDM_DETECT_PATTERN, line) and not ("100%" in line and re.search(TQDM_FINISHED_PATTERN, line)):
            line = line.removeprefix("\n").removesuffix("\n") + "\r"
    sys.stdout.write(line)
    sys.stdout.flush()

def print_and_sleep(text : str):
    custom_print(text)
    time.sleep(3)

def run_command(
        command : Union[str | Callable], 
        python_binary : Optional[str]=None
    ):
    if callable(command):
        command = command()
    elif isinstance(command, str):
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
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
    else:
        raise ValueError("Command must be a string or callable.")

def remove_directory(
        directory : str, 
        recursive : bool=False
    ):
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

def split_by_sample(files : List[str]) -> Dict[str, List[str]]:
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

def set_temp_config_dir(**kwargs):
    global TMP_DIR
    if "dir" in kwargs:
        if not os.path.exists(kwargs["dir"]):
            os.makedirs(kwargs["dir"], exist_ok=True)
    TMP_DIR = tempfile.mkdtemp(**kwargs)

def get_temp_config_dir() -> str:
    global TMP_DIR
    if TMP_DIR == "<UNSET>":
        set_temp_config_dir(dir=os.environ["HOME"], prefix="fb_tmp_experiment_configs_")
    return TMP_DIR

def get_temp_config_path() -> str:
    return tempfile.NamedTemporaryFile(dir=get_temp_config_dir(), mode="w", delete=False).name

def set_outputdir(outputdir : str):
    global OUTPUT_DIR
    if outputdir == "<UNSET>":
        raise ValueError("Output directory must be set to other than '<UNSET>'.")
    if not os.path.exists(outputdir):
        os.makedirs(outputdir, exist_ok=True)
    OUTPUT_DIR = outputdir
    set_projectdir(outputdir)
    set_temp_config_dir(dir=outputdir, prefix="fb_tmp_experiment_configs_")

def get_outputdir(strict : bool=True) -> str:
    global OUTPUT_DIR
    if OUTPUT_DIR == "<UNSET>":
        if strict:
            raise RuntimeError("The output directory has not been set. Use `experiment_helpers.set_OUTPUT_DIR(<path>)` to set it.")
        else:
            return False
    return OUTPUT_DIR

def clean_temporary_dir():
    global TMP_DIR
    if TMP_DIR != "<UNSET>":
        tmp_files = os.listdir(TMP_DIR)
        for tmp_file in tmp_files:
            os.remove(os.path.join(TMP_DIR, tmp_file))
        os.rmdir(TMP_DIR)
        TMP_DIR = "<UNSET>"

class ZipOrDirectory:
    def __init__(self, path : str):
        if not isinstance(path, str):
            raise TypeError(f"Supplied path '{path}' should be a `str` not `{type(path)}`")
        self._path = os.path.abspath(os.path.normpath(path))
        self._zip = zipfile.ZipFile(self._path) if zipfile.is_zipfile(self._path) else None
        self._zip_root = self._get_zip_root()
    
    def _get_zip_root(self) -> str:
        if self._zip is None:
            return ""
        top_level = list(set([content.split("/")[0] for content in self._zip.namelist()]))
        if len(top_level) == 1:
            return top_level[0] + "/"
        return ""
    
    def _zip_prep_path(self, path : str) -> str:
        if os.sep != "/" and os.sep in path:
            path = path.replace(os.sep, "/")
        if self._zip_root:
            path = f'{self._zip_root}{path}'
        return path
    
    def open(self, path : str, mode : str = "r", *args, **kwargs) -> IO[bytes]:
        path = os.path.normpath(path)
        if isinstance(self._zip, zipfile.ZipFile):
            mode = mode.replace("t", "")
            raw_file = self._zip.open(self._zip_prep_path(path), mode=mode, *args, **kwargs)
            if not "b" in mode:
                return io.TextIOWrapper(raw_file)
            return raw_file
        else:
            return open(os.path.join(self._path, path), mode=mode, *args, **kwargs)
    
    def contains(self, path : str) -> bool:
        path = os.path.normpath(path)
        if isinstance(self._zip, zipfile.ZipFile):
            return self._zip_prep_path(path) in self._zip.namelist()
        else:
            return os.path.exists(os.path.join(self._path, path))
    
    def close(self) -> None:
        if isinstance(self._zip, zipfile.ZipFile):
            self._zip.close()
    
    def __del__(self) -> None:
        self.close()

class HelpfulArgumentParser(argparse.ArgumentParser):
    def error(self, message : str):
        self.print_help(sys.stderr)
        self.exit(2, f"\n\n{self.prog}: error: {message}\n")

def parse_unknown_arguments(extra : List[str]) -> Dict[str, Any]:
    """
    Parses unknown arguments from the command line.

    Unknown arguments must be named arguments in the form `--key value`, `-key value` or `key=value`.

    Args:
        extra (List[str]): The list of extra arguments.

    Returns:
        Dict[str, Any]: The parsed unknown arguments.
    """
    unknown_args = {}
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg.startswith("--"):
            key = arg.removeprefix("--")
            value = extra[i+1]
            i += 1
        elif arg.startswith("-"):
            key = arg.removeprefix("-")
            value = extra[i+1]
            i += 1
        elif "=" in arg:
            key, value = arg.split("=")
        else:
            position = sum([len(arg) for arg in extra[:i]]) + i
            raise ValueError(f"Unable to parse extra misspecified or unnamed argument: `{arg}` at position {position}:{position + len(arg)}.")
        if value.isdigit():
            value = int(value)
        elif value.isdecimal():
            value = float(value)
        unknown_args[key] = value
        i += 1
    return unknown_args

def get_cmd_args() -> Tuple[Namespace, Dict[str, str]]:
    """
    A simple wrapper for shared command line arguments and parsing between experiment orchestration scripts.

    Command line arguments:
        -i, --datadir: The directory containing the data.
        --dry-run: Print the experiment configurations without running them.
        --devices: The GPU(s) to use for the experiments.
        --slurm: Use SLURM for the experiments.
        ... and any additional SBATCH arguments to pass to the experiments. Only relevant if using --slurm.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    args_parse = HelpfulArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parse.add_argument("-i", "--datadir", help="The directory containing the data.", required=True)
    args_parse.add_argument("-o", "--output", dest="output", help="The directory to save the output.", required=False)
    args_parse.add_argument("--devices", "--device", nargs="+", help="The GPU(s) to use for the experiments.", default=0)
    args_parse.add_argument("--soft", dest="soft", help="Ignore missing input directories.", action="store_true")
    args_parse.add_argument("--slurm", action="store_true", help="Use SLURM for the experiments.")
    args_parse.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print the experiment configurations without running them.")
    args_parse.add_argument(
        "--do-not-specify-extra",
        dest="extra",
        nargs=argparse.REMAINDER, 
        help=\
            "Extra SBATCH arguments to pass to the experiments. DO NOT ACTUALLY SPECIFY --do-not-specify-extra, just pass the arguments after the other arguments.\n"
            "Extra must be passed as named arguments in the form `--key value`, `-key value` or `key=value`.\n"
            "For most experiments these will be passed as SLURM parameter overrides, for example you could specify:\n"
            "`python <some_script.py> -i <DIR> --slurm --partition <PARTITION> --dependency afterok:<JOB_ID>`."
    )
    args, extra = args_parse.parse_known_args()
    try:
        extra = parse_unknown_arguments(extra)
    except ValueError as e:
        raise ValueError(
                f"Error parsing extra arguments: `{' '.join(extra)}`. {e}\n\n"
                f"{args_parse.format_help()}"
            )
    if not args.extra is None:
        probable_desired_command = sys.executable + " " + " ".join([arg for arg in sys.argv if arg != "--do-not-specify-extra"])
        raise ValueError(
                f"DO NOT ACTUALLY SPECIFY --do-not-specify-extra, just pass the extra arguments after known.\n\n"
                "You probably meant to use:\n"
                f"\t{probable_desired_command}\n\n"
                f"{args_parse.format_help()}"
            )
    if not args.output is None:
        args.output = os.path.normpath(os.path.expanduser(args.output))
    if not args.output is None:
        project_dir = args.output
    else:
        project_dir = os.path.abspath("./runs/segment")
    set_outputdir(project_dir)
    set_datadir(args.datadir, strict=not args.soft)
    return args, extra

def read_slurm_params(
        path : Optional[str] = os.path.join(os.path.dirname(__file__), "default_slurm_params.yaml"),
        **kwargs
    ) -> Dict[str, Any]:
    """
    Simple wrapper to read SLURM parameters from a YAML file, or use the default SLURM parameters if not supplied.

    Args:
        path (Optional[str]): The path to the SLURM parameters YAML file. Default and None is "default_slurm_params.yaml" in the same directory as this script.
        **kwargs: Additional keyword arguments to pass to the SLURM parameters (e.g. partition). These will override the parameters in the YAML file if they are also present.

    Returns:
        Dict[str, Any]: The SLURM parameters. The keys are prefixed with 'slurm_', necessary for the submitit executor.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "default_slurm_params.yaml")
    with open(path, "r") as f:
        params : Dict = yaml.safe_load(f)
    params.update(kwargs)

    ## THIS IS NOT NECESSARY AFTER SWITCHING FROM `submitit.AutoExecutor` TO `submitit.SlurmExecutor`
    # As a reminder, shared/generic (non-prefixed) parameters are: {'name': <class 'str'>, 'timeout_min': <class 'int'>, 'mem_gb': <class 'float'>, 'nodes': <class 'int'>, 'cpus_per_task': <class 'int'>, 'gpus_per_node': <class 'int'>, 'tasks_per_node': <class 'int'>, 'stderr_to_stdout': <class 'bool'>}.
    # Change all non-shared parameters to have the 'slurm_' prefix
    # shared_params = ['name', 'timeout_min', 'mem_gb', 'nodes', 'cpus_per_task', 'gpus_per_node', 'tasks_per_node', 'stderr_to_stdout']
    # for key in list(params.keys()):
    #     if key not in shared_params:
    #         params[f"slurm_{key}"] = params.pop(key)
    #### END OF NOT NECESSARY

    # Get list of submitit-slurm parameters
    available_params = _default_submitit_slurm_params().keys()
    
    # Disentangle submitit slurm parameters from additional (advanced) slurm parameters
    additional_params = dict()
    for key in list(params.keys()):
        if key not in available_params:
            additional_params[key] = params.pop(key)
    
    # Snipe and replace the bespoke "slurm_setup" parameter (allows passing a list of commands to the "setup" parameter using a text-file)
    if "slurm_setup" in additional_params:
        slurm_setup_path = os.path.join(os.path.dirname(__file__), "slurm_config", additional_params.pop("slurm_setup"))
        if not (isinstance(slurm_setup_path, str) and os.path.exists(slurm_setup_path)):
            raise FileNotFoundError(f"Invalid SLURM setup file specified: {slurm_setup_path}.")
        with open(slurm_setup_path, "r") as f:
            slurm_setup_commands = f.read().strip().split("\n")
        if slurm_setup_commands[0] == 0:
            slurm_setup_commands.pop(0) 
        assert len(slurm_setup_commands) > 0, f"Empty SLURM setup file specified." 
        params["setup"] = additional_params.get("setup", []) + slurm_setup_commands
    
    # Submit additional parameters via the "additional_parameters" parameter
    if additional_params:
        params["additional_parameters"] = additional_params

    return params

def do_yolo_train_run(
        config : Dict, 
        dry_run : bool=False, 
        execute : bool=True, 
        device : Optional[Union[int, str, List[Union[int, str]]]]=None
    ) -> str:
    """
    Wrapper for conducting a Flat-Bug YOLO training run, with `fb_train`.

    Args:
        config (Dict): The configuration dictionary.
        dry_run (bool): Whether to print the command without running it. Defaults to False.
        execute (bool): Whether to run the command. Defaults to True.
        device (Optional[Union[int, str]]): The GPU to use for the experiment. Defaults to None.

    Returns:
        str: The (executed) command (to run).
    """
    if DATA_DIR == "<UNSET>":
        raise RuntimeError("The data directory has not been set. Use `experiment_helpers.set_datadir(<path>)` to set it.")
    ITEMIZE = '\n  - '

    # Set/override the device if specified
    if device is not None:
        sanitize_device = lambda x : f"cuda:{x}" if isinstance(x, int) or x.isdigit() else x
        config["device"] = sanitize_device(device) if isinstance(device, (str, int)) else [sanitize_device(d) for d in device]

    # The config file is written to a "temporary" directory, which can be cleaned once the commands have been executed. In the case of non-SLURM execution, this is done automatically if using the `ExperimentRunner` class.
    config_path = get_temp_config_path()
    with open(config_path, "w") as conf:
        yaml.dump(config, conf, default_flow_style=False, sort_keys=False)
    
    print(f"Running experiment: {config['name']} with config:{ITEMIZE + ITEMIZE.join([f'{k}: {v}' for k, v in config.items()])}")
    command = f'fb_train -c "{config_path}" -d "{DATA_DIR}"'
    if execute:
        if dry_run:
            print(command)
        else:
            run_command(command)
    return command


class ExperimentRunner:
    def __init__(
            self : Self, 
            experiment_fn : Callable = do_yolo_train_run, 
            inputs : Iterable = [], 
            devices : Optional[Union[List[Union[int, str]], int, str]] = None, 
            slurm : bool=False, 
            slurm_params : Optional[Dict[str, Any]] = None, 
            **kwargs
        ):
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
            slurm_folder = os.path.join(os.getcwd(), "slurm_logs") if get_outputdir(strict=False) is False else os.path.join(get_outputdir(), "slurm_logs")
            self.executor = submitit.SlurmExecutor(folder=slurm_folder, max_num_timeout=0)
            self.executor.update_parameters(**slurm_params)

        # Initialize consumer/job lists
        self.consumer_threads : List[threading.Thread] = []
        self.slurm_jobs : List[submitit.Job] = []
        
    def __len__(self):
        return self._length
    
    @property
    def multi_gpu(self):
        return isinstance(self.devices, list) and len(self.devices) > 1 and not self.slurm

    @staticmethod
    def consumer_thread(
            fn : Callable, 
            queue : queue.Queue, 
            **kwargs
        ):
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

    @property
    def slurm_job_ids(self : Self) -> List[str]:
        if not self.slurm:
            return [""]
        else:
            return [job.job_id for job in self.slurm_jobs]
    
    @property
    def slurm_job_id(self : Self) -> str:
        if not self.slurm:
            return self.slurm_job_ids[0]
        else:
            ids = list(set([re.search(r"^(\d+)", job.job_id).group(1) for job in self.slurm_jobs]))
            if len(ids) != 1:
                raise ValueError(f"Multiple job IDs found: {ids}")
            return ids[0]

    def run(self : Self) -> Self:
        # Check that the consumer threads and slurm jobs are empty
        assert not self.consumer_threads, "Consumer threads list is not empty."
        assert not self.slurm_jobs, "Slurm jobs list is not empty."
        if self.multi_gpu:
            # For multiple GPUs, we use a producer-consumer model, with one consumer per GPU
            for device in self.devices:
                this_kwargs = self.kwargs.copy()
                this_kwargs["device"] = device
                thread = threading.Thread(target=self.consumer_thread, args=(self.experiment_fn, self.experiment_queue), kwargs=this_kwargs, daemon=True)
                thread.start()
                self.consumer_threads.append(thread)
        else:
            if isinstance(self.devices, (list, tuple, set)):
                if len(self.devices) == 1:
                    self.devices = self.devices[0]
                elif len(self.devices) == 0:
                    self.devices = None
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
                    self.slurm_jobs = self.executor.map_array(print_and_sleep, cmds)
                else:
                    self.slurm_jobs = self.executor.map_array(run_command, cmds)
        
        return self

    def wait(self : Self) -> Self:
        # Wait for the consumer threads to finish
        [self.consumer_threads.pop().join() for _ in range(len(self.consumer_threads))]
        # Wait for the slurm jobs to finish
        [print(f'Job {i} finished with:', self.pretty_parse_slurm_results(self.slurm_jobs.pop())) for i in range(len(self.slurm_jobs))]

        return self

    def complete(self : Self) -> Self:
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

        return self
