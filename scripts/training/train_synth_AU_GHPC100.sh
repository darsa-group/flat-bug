#!/bin/bash
# Does synthetic touching-instance scenes help? 100-epoch A/B on the halo-fixed code.
#
#   sbatch --export=ALL,ARM=on  train_synth_AU_GHPC100.sh   # fb_synth_prob 0.4
#   sbatch --export=ALL,ARM=off train_synth_AU_GHPC100.sh   # fb_synth_prob 0.0
#
# Both arms run the SAME source tree and the same config file; ARM only overrides
# fb_synth_prob on the command line, so nothing else can drift between them.

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=96000
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:1
#SBATCH -J fb_synth
#SBATCH -o /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_synth_%x_%j.out
#SBATCH -e /usr/home/qgg/qgeiss/flatbug-dir/logs/fb_synth_%x_%j.err

source ~/.venv/bin/activate
export PYTHONPATH=$HOME/flat-bug-synth-test/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=$HOME/flatbug-dir
SRC=$HOME/flat-bug-synth-test

case "$ARM" in
  on)  PROB=0.4 ;;
  off) PROB=0.0 ;;
  *) echo "set ARM=on or ARM=off"; exit 1 ;;
esac

# fb_train rejects fb_-prefixed command-line options outright - they have to come from the
# config file - so the arm is applied by rewriting exactly that one line of the shared base
# config into a per-job copy. One source of truth, one line different, and the diff is echoed
# into the job log so the run records what it actually trained on.
BASE=$SRC/scripts/training/fb_config_synth_GHPC100.yaml
CFG=$(mktemp /tmp/fb_config_synth_${ARM}_XXXXXX.yaml)
sed "s|^fb_synth_prob:.*|fb_synth_prob: ${PROB}|" "$BASE" > "$CFG"
grep -q "^fb_synth_prob: ${PROB}\$" "$CFG" || { echo "failed to set fb_synth_prob in $CFG"; exit 1; }
echo "arm=$ARM  config=$CFG"
diff "$BASE" "$CFG" || true

NAME=fb_synth_${ARM}_$(date +"%Y-%m-%d_%H-%M-%S")

cd $SRC/scripts/training
fb_train -c "$CFG" -d ${ROOT}/flat-bug-data/yolo/ --name ${NAME}
rm -f "$CFG"
