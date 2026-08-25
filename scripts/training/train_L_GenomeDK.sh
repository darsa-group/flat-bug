#!/bin/bash
# Train yolo26l-seg on GenomeDK.
#
#   sbatch --account=<project> scripts/training/train_L_GenomeDK.sh
#
# The SBATCH header below is NOT verified against GenomeDK. Before relying on it:
#   sinfo -s                     # partition names
#   sinfo -o "%P %G %m %c"       # GPU resource strings, memory, cores per node
#   gnodes                       # GenomeDK's own node overview, if available
# Override at submit time rather than editing, e.g.
#   sbatch --account=myproject --partition=gpu --gres=gpu:a100:1 train_L_GenomeDK.sh

#SBATCH --account=CHANGEME
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=168:00:00
#SBATCH --job-name=flatbug_L
#SBATCH --output=flatbug_L_%j.out

set -euo pipefail

if [[ "${SLURM_JOB_ACCOUNT:-CHANGEME}" == "CHANGEME" ]]; then
    echo "ERROR: GenomeDK rejects jobs without a project account." >&2
    echo "       sbatch --account=<project> $0" >&2
    exit 1
fi

# GenomeDK home directories are small and are not meant for datasets or runs.
# ROOT must point at project storage.
ROOT=${ROOT:-/faststorage/project/${SLURM_JOB_ACCOUNT}/flatbug-dir}
if [[ ! -d "${ROOT}/flat-bug-data/yolo" ]]; then
    echo "ERROR: no prepared dataset at ${ROOT}/flat-bug-data/yolo" >&2
    echo "       Set ROOT, or stage the data there first (see notes at the end)." >&2
    exit 1
fi

# GenomeDK uses conda/mamba rather than a bare venv.
FB_ENV=${FB_ENV:-flatbug}
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${FB_ENV}"
elif [[ -f ~/.venv/bin/activate ]]; then
    source ~/.venv/bin/activate
else
    echo "ERROR: no conda environment '${FB_ENV}' and no ~/.venv to activate" >&2
    exit 1
fi

export PYTHONPATH=${FB_SRC:-$HOME/flat-bug-synth/src}

CONFIG=$(dirname "$(readlink -f "$0")")/fb_config_L_GenomeDK.yaml
NAME=fb_L_$(date +"%Y-%m-%d_%H-%M-%S")

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "config: ${CONFIG}"
echo "data:   ${ROOT}/flat-bug-data/yolo"

fb_train -c "${CONFIG}" -d "${ROOT}/flat-bug-data/yolo/" --name "${NAME}"
