#!/bin/bash

#SBATCH --account=CHANGEME          # GenomeDK project; jobs are rejected without one
#SBATCH --partition=gpu             # Verify against `sinfo -s` on the cluster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --job-name=flatbug_m2f
#SBATCH --output=m2f_%j.out

# GenomeDK counterpart of train_m2f_AU_GHPC.sh. Options are documented in,
# and everything is done by, m2f_pipeline.sh.
#
# The header above is NOT verified against GenomeDK - confirm the partition name
# and GPU resource string with `sinfo -s` and `sinfo -o "%P %G"` before relying
# on it, and override at submit time rather than editing:
#
#   sbatch --account=myproject --partition=gpu scripts/training/train_m2f_GenomeDK.sh
#
# GenomeDK home directories are small; keep data and runs on project storage:
#   export ROOT=/faststorage/project/<project>/flatbug-dir

set -euo pipefail

if [[ "${SLURM_JOB_ACCOUNT:-CHANGEME}" == "CHANGEME" ]]; then
    echo "ERROR: set a GenomeDK project account, e.g. sbatch --account=myproject ..." >&2
    exit 1
fi

# GenomeDK uses conda/mamba rather than a bare venv. FB_ENV names the environment.
FB_ENV=${FB_ENV:-flatbug}
if command -v mamba >/dev/null 2>&1 || command -v conda >/dev/null 2>&1; then
    # `conda activate` needs the shell hook inside a non-interactive job.
    eval "$(conda shell.bash hook)"
    conda activate "${FB_ENV}"
elif [[ -f ~/.venv/bin/activate ]]; then
    source ~/.venv/bin/activate
else
    echo "ERROR: no conda/mamba environment and no ~/.venv to activate" >&2
    exit 1
fi

exec "$(dirname "$(readlink -f "$0")")/m2f_pipeline.sh"
