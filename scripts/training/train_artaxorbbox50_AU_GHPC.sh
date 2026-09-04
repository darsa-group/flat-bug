#!/bin/bash
# Does artaxor-bbox help as a boxes-only dataset? 50-epoch A/B.
#
#   sbatch --job-name=fb_axb_excl --export=ALL,ARM=excl train_artaxorbbox50_AU_GHPC.sh
#   sbatch --job-name=fb_axb_bbox --export=ALL,ARM=bbox train_artaxorbbox50_AU_GHPC.sh
#
# Both arms run the same tree and the same base config; ARM rewrites exactly one line, and the
# diff is echoed into the job log so what each arm trained on is auditable.

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96000
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_axb_%x_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_axb_%x_%j.err

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-axb/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=$HOME/flatbug-dir2
SRC=$HOME/flat-bug-axb

case "$ARM" in
  excl) LINE='fb_exclude_datasets: ["artaxor-bbox"]' ; KEY='^fb_exclude_datasets:' ;;
  bbox) LINE='fb_bbox_only_datasets: ["artaxor-bbox"]' ; KEY='^fb_bbox_only_datasets:' ;;
  *) echo "set ARM=excl or ARM=bbox"; exit 1 ;;
esac

BASE=$SRC/scripts/training/fb_config_artaxorbbox50_GHPC.yaml
CFG=$(mktemp /tmp/fb_config_axb_${ARM}_XXXXXX.yaml)
sed "s|${KEY}.*|${LINE}|" "$BASE" > "$CFG"
grep -q "artaxor-bbox" "$CFG" || { echo "failed to set the arm in $CFG"; exit 1; }
echo "arm=$ARM  config=$CFG"
diff "$BASE" "$CFG" || true

NAME=fb_axb_${ARM}_$(date +"%Y-%m-%d_%H-%M-%S")
cd $SRC/scripts/training
fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
rm -f "$CFG"
