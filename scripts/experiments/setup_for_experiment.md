# Guide for setting up the `flat-bug` experiments on the Compute Canada HPC cluster
This is a minimal guide for the proper setup of the `flat-bug` experiment environment on the Digital Research Alliance of Canada HPC cluster. 

This setup focuses on reproducibility and leveraging the specific SLURM infrastructure on the cluster. The guide is written to follow the tips and guidelines specified in the clusters' [Technical documentation](https://docs.alliancecan.ca/wiki/Technical_documentation) as closely as possibly, while enhancing the speed and efficiency of the jobs and allocated resources.

## Import flat-bug
The first step is to download `flat-bug` from Github, This needs the proper Github configuration and credentials available in the environment (SSH).
# SSH
I will omit the SSH setup from this guide, but I suggest using a SSH config file (see [this blog](https://linuxize.com/post/using-the-ssh-config-file/) for details) and an SSH-agent, the latter can be enabled by adding this to your local `.bashrc` (if using Linux):
```bash
eval $(ssh-agent)
ssh-add
```
You can check if this is setup by running:
```bash
ssh-add -l
```
on the login-node.
# Clone repository
```bash
# Git setup
git config --global user.name <USER_NAME>
git config --global user.emauil <EMAIL>
# Clone repository
git clone git@github.com:darsa-group/flat-bug.git
git checkout dev_experiments # or another branch
cd flat-bug
```

## Create a base virtual environment
It is not strictly necessary to create a virtual environment with `flat-bug` on the login-node for all experiments (only `compare_models.py` strictly needs it, and could quite easily be amended such that it is not necessary). However, it is highly encouraged for two reasons (1) it makes you more familiar with the process and (2) it allows you to use the `Python` interpreter associated with the environment to properly parse the scripts in `flat-bug` and provide proper type hints, warnings and documentation on hover (if using an IDE like VSCode). 
```bash
# Load the necessary modules
module load python/3.11.5 opencv/4.10.0 gcc scipy-stack/2024a r/4.4.0
# Create a virtual environment
virtualenv --no-download fb_env
# Install flat-bug in editable mode
pip install -e .
```

## Download and prepare dataset
In order to run the `flat-bug` experiments it is necessary to download the `flat-bug` dataset, which can be found on an `S3` bucket. Here I show how it can be installed using `s5cmd`, but any tool which can access `S3` can be used instead.
```bash
## The following environment variables need to be setup (I suggest in your .bashrc on the login-node, i.e. /home/$USER/.bashrc)
# export AWS_ACCESS_KEY_ID=<ACCESS_KEY_ID>
# export AWS_SECRET_ACCESS_KEY=<SECRET_ACCESS_KEY>
# export AWS_PROFILE=<AWS_USER>
# export AWS_REGION=se-sto-1.linodeobjects.com 

# Install s5cmd
pip install s5cmd

# Setup data directory
FBDIR="$HOME/scratch/fb_data"
mkdir "$FBDIR"
mkdir "$FBDIR/pre-pro"
mkdir "$FBDIR/fb_yolo"

# Download the flat-bug data to the data directory
s5cmd --endpoint-url https://se-sto-1.linodeobjects.com sync -c 10 "s3://flat-bug-data-stable/pre-pro/*" "$FBDIR/pre-pro"
# Format the data for training/eval
fb_prepare_data -i "$FBDIR/pre-pro" -o "$FBDIR/fb_yolo"
# Create a zip-archive of the files for faster transfer to compute nodes
cd $FBDIR
zip -r fb_yolo.zip fb_yolo # Can be very slow and should be done in an interactive job (allocate more than 1 hour to be sure!)
cd $HOME/flat-bug
```

## Create a wheel for offline install
In order to install `flat-bug` in a local virtual environment on the allocated node during a SLURM job, it is necessary to pre-download the dependencies of `flat-bug` as the compute nodes do not have internet access. Probably the easiest way to do this is to use `build` to create a local wheel for `flat-bug` and then use `pip download` to download wheels for the dependencies.
```bash
# Install/update build
pip install build
# Create a wheel for flat-bug
python -m build --wheel
# Create wheels for the dependencies
pip download . -d dist # Can be pretty slow
```

## Install R dependencies
Unfortunately we are currently using an `R` script for our end-to-end evaluation, due to its' much superior plotting capabilities compared to `Python`. This necessitates that the following packages are installed `data.table`, `ggplot2`, `scales`, `optparse` and `magrittr`, which can be done in an interactive `R` shell on the login-node, first simply run the command `R` in the terminal. Then:
```R
install.packages(c("data.table", "ggplot2", "scales", "optparse", "magrittr"))
```
For the prompts simply answer 'yes' for the onees regarding the install path (library) of the packages, and for the CRAN server select one from Canada (probably 12).

## Install offline inside SLURM job
To use the wheels created in the prior section for an offline install of `flat-bug` inside a compute node, the following can be used: 
```bash
# Inside the bash job (or elsewhere) - load the necessary modules
module load python/3.11.5 opencv/4.10.0 gcc scipy-stack/2024a r/4.4.0
# Create a new virtual environment in the local storage of the job and activate it
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
# Install flat-bug from the locally build wheel(s)
pip install --no-index --find-links=$HOME/flat-bug/dist dist/flat_bug-0.3.0-py3-none-any.whl
# (optional) Create a copy of the dataset on the local storage of the job
mkdir "$SLURM_TMPDIR/fb_yolo"
unzip "$HOME/scratch/fb_data/fb_yolo.zip" -d "$SLURM_TMPDIR/fb_yolo" 
```

In reality, we include this:
```bash
# Clean environment
deactivate && module purge
cd $HOME/flat-bug

# Setup virtual environments and local data on all nodes
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash <<EOF
#!/bin/bash
# Install flat-bug in a local virtual environment
module load python/3.11.5 opencv/4.10.0 gcc scipy-stack/2024a r/4.4.0
virtualenv --no-download \$SLURM_TMPDIR/env
source \$SLURM_TMPDIR/env/bin/activate
pip install --no-index --find-links=\$HOME/flat-bug/dist dist/flat_bug-0.3.0-py3-none-any.whl
pip list
echo "Current virtual environment (node \$SLURM_JOB_ID.\$SLURM_ARRAY_TASK_ID%\$SLURM_ARRAY_TASK_COUNT): \$VIRTUAL_ENV"
# Create the output directory for the job
mkdir \$SLURM_TMPDIR/job_output
# Copy flat-bug data to local storage
unzip /home/asgersve/scratch/fb_data/fb_yolo.zip -d \$SLURM_TMPDIR
# Print the state of the local storage
echo "Contents of the temporary \$SLURM_TMPDIR:"
ls -a1 \$SLURM_TMPDIR
EOF

# Activate the environment only on the main node (see https://docs.alliancecan.ca/wiki/Python#Creating_virtual_environments_inside_of_your_jobs_(multi-nodes))
module load python/3.11.5 opencv/4.10.0 gcc scipy-stack/2024a r/4.4.0
source $SLURM_TMPDIR/env/bin/activate;

echo "Running jobs..."
```
inside our SLURM batch scripts before calling `srun`. This template is found in `flat-bug/scripts/experiments/slurm_config/setup.txt` and is included (which is almost a requirement) by specifying:
```bash
python scripts/experiments/.../SOME_SCRIPT.py --some-args some-value --slurm slurm_setup=setup.txt
```

## Using SLURM_TMPDIR with submitit and Python scripts
If you have a Python script which accepts a path as an argument, such as this one (lets call it `script.py`):
```py
import os, argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, dest="path")
    args, _ = args.parse_known_args()
    
    os.system(f"echo {args.path}")
```
This path could for example be the directory of the data or model weights which should used --- and if this path will be inside SLURM_TMPDIR at runtime, then you have a problem; the actual path of the SLURM_TMPDIR depends on the job ID, and in the case of array jobs the array ID as well. Here you should do this:
```bash
# YES
python script.py --path "\$SLURM_TMPDIR/my_path"

# NO!
python script.py --path "$SLURM_TMPDIR/my_path"
```
**OBS:** This can be used for arbitrary code injection, i.e. a security vulnerability, if the arguments can be controlled by an outside user in an unsanitized manner. 

# Example for one of our experiments
```sh
# Ensure that the proper modules and virtual environment (with flat-bug) is active
# This is NOT run with a SLURM script, instead our experiments use `submitit` to submit an array job for all the subtasks in each experiment
# This means that the experiment scripts should be executed on a login-node with the `--slurm` flag followed by the SLURM config arguments. 
# The script should execute relatively quickly (less than a minute mostly), with all subtasks being queued and visible with `squeue -u $USER`
cd $HOME/flat-bug
python scripts/experiments/compare_models.py \
    -i "\$SLURM_TMPDIR/fb_yolo/insects/images/val" \
    -g "\$SLURM_TMPDIR/fb_yolo/insects/labels/val/instances_default.json" \
    --tmp "\$SLURM_TMPDIR/job_output" \
    -o "$HOME/scratch/my_output_folder" \ # Notice that $HOME is NOT escaped here
    -d "~/scratch/output/experiment_*" \ # You can use "$HOME" and "~" interchangeably
    --soft \# Disables check for existence of `i` and `g`
    --slurm \
    slurm_setup=setup.txt \
    gres=gpu:t4:1 \
    cpus_per_task=12 \
    mem=32GB \
    time=01:00:00
# OBS: Comments cannot be included in actual bash code
``` 