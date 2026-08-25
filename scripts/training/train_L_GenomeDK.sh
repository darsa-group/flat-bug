#!/bin/bash
# Train yolo26l-seg on GenomeDK.
#
#   sbatch --account=<project> scripts/training/train_L_GenomeDK.sh
#
# Verified on GenomeDK 2026-08-25: gpu-h200 has 4x H200 (141 GB) per node on
# gn-1003/gn-1004, 7-day limit; gpu-l40s has 7x L40S (48 GB). At batch 16 the
# measured throughput is 50.4 img/s, so 100 epochs over 22,266 samples is ~12 h;
# 48 h leaves ample margin.

#SBATCH --account=CHANGEME
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --job-name=flatbug_L
#SBATCH --output=/faststorage/project/flat-bug/logs/flatbug_L_%j.out

set -euo pipefail

if [[ "${SLURM_JOB_ACCOUNT:-CHANGEME}" == "CHANGEME" ]]; then
    echo "ERROR: GenomeDK rejects jobs without a project account." >&2
    echo "       sbatch --account=<project> $0" >&2
    exit 1
fi

# GenomeDK home directories are small and are not meant for datasets or runs.
# ROOT must point at project storage.
ROOT=${ROOT:-/faststorage/project/flat-bug/flatbug-dir}
if [[ ! -d "${ROOT}/flat-bug-data/yolo" ]]; then
    echo "ERROR: no prepared dataset at ${ROOT}/flat-bug-data/yolo" >&2
    echo "       Set ROOT, or stage the data there first (see notes at the end)." >&2
    exit 1
fi

# GenomeDK uses conda/mamba rather than a bare venv.
# GenomeDK has no module system and no conda; the environment is a uv-built
# venv on project storage (Python 3.11, torch 2.13+cu130).
VENV=${FB_VENV:-/faststorage/project/flat-bug/venv}
if [[ ! -f "${VENV}/bin/activate" ]]; then
    echo "ERROR: no venv at ${VENV}" >&2
    exit 1
fi
source "${VENV}/bin/activate"

export PYTHONPATH=${FB_SRC:-/faststorage/project/flat-bug/code/flat-bug/src}
# GenomeDK nodes reach the internet only through a proxy; ultralytics needs it
# to fetch yolo26l-seg.pt on first use.
export http_proxy=${http_proxy:-http://proxy-default:3128}
export https_proxy=${https_proxy:-http://proxy-default:3128}

CONFIG=$(dirname "$(readlink -f "$0")")/fb_config_L_GenomeDK.yaml
NAME=fb_L_$(date +"%Y-%m-%d_%H-%M-%S")

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "config: ${CONFIG}"
echo "data:   ${ROOT}/flat-bug-data/yolo"

fb_train -c "${CONFIG}" -d "${ROOT}/flat-bug-data/yolo/" --name "${NAME}"
