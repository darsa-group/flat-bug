import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.experiments.experiment_helpers import get_cmd_args, read_slurm_params, ExperimentRunner, run_command

def print_environment_status_for_debug(test_string : str, dry_run : bool=False, execute : bool=True, device=None):
    command = f"""
        echo "Job attempted to run using devices (GPUs): {device}";
        echo "{test_string}";
        echo "Found environment setup:";
        echo "VIRTUAL_ENV=$(printenv VIRTUAL_ENV || echo 'NOT_FOUND!')";
        echo "SLURM_TMPDIR=$(printenv SLURM_TMPDIR || echo 'NOT_FOUND!')";
        echo "Python binary location: $(which python)";
        echo "Python version: $(python --version)";
        if command -v pip &> /dev/null; then
            echo "pip version: $(pip --version)";
            echo "Installed pip packages:";
            pip list;
        else
            echo "pip command not found. Is pip installed?";
        fi
        if command -v nvidia-smi &> /dev/null; then
            echo "nvidia-smi output:";
            nvidia-smi;
        else
            echo "nvidia-smi command not found. Are you on a machine with NVIDIA GPUs?";
        fi
        """
    if execute:
        if dry_run:
            print(command)
        else:
            run_command(command)
    return command
    
if __name__ == "__main__":
    args, extra = get_cmd_args()

    test_inputs = [f"THIS IS A TEST OF JOB {i}" for i in range(3)]
    
    experiment_runner = ExperimentRunner(print_environment_status_for_debug, inputs=test_inputs, devices=args.devices, dry_run=args.dry_run, slurm=args.slurm, slurm_params=read_slurm_params(**extra))
    experiment_runner.run().complete()
    
