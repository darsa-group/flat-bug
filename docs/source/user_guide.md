# User guide

## Install

flat-bug is a Python package. We recommend installing it with [uv](https://docs.astral.sh/uv/),
which also picks the PyTorch build that matches your GPU:

```bash
uv pip install flat-bug --torch-backend=auto
```

or, to add it to a uv project:

```bash
uv add flat-bug
```

`pip install flat-bug` works too, but then install PyTorch first, following
[pytorch.org](https://pytorch.org/get-started/locally/), so that it matches your CUDA version.
To check that the GPU is used:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If that prints `False` on a machine with an NVIDIA GPU, reinstall PyTorch for your hardware:
`uv pip install torch torchvision --torch-backend=auto --reinstall`.

## Find the animals in your images

From the command line, point `fb_predict` at an image or a folder of images:

```bash
fb_predict -i my_images/ -o results/
```

Useful options:

| Option | What it does |
|---|---|
| `-w flat_bug_M_v2.pt` | which model to use; see [](models.md). A name is downloaded from the model zoo on first use, a path is used as is. |
| `-g cuda:0` / `-g cpu` | which device to run on (default: the GPU if there is one) |
| `-R` | also look for images in sub-folders |
| `--config my_config.yaml` | change the predictor's settings (below) |
| `--no-crops`, `--no-overviews` | skip the crops or the overview images, which saves a lot of disk |

From Python:

```python
from flat_bug.predictor import Predictor

model = Predictor(model="flat_bug_M_v2.pt", device="cuda:0", dtype="float16")
result = model("my_images/trap_042.jpg")

print(len(result), "animals")
result.save("results/", overview=True, crops=True)
```

The [deploy tutorial](https://github.com/darsa-group/flat-bug/blob/main/examples/tutorials/deploy.ipynb)
goes further, and runs on [Google Colab](https://colab.research.google.com/github/darsa-group/flat-bug/blob/master/docs/flat-bug.ipynb).

## What you get

Each image gets its own folder in the output:

```text
results/
└── trap_042/
    ├── metadata_trap_042_UUID_<id>.json   every detection: outline, box, confidence
    ├── overview_trap_042.jpg              the image with every outline drawn on it
    └── crops/                             one image per animal
```

The JSON file holds one entry per animal in parallel lists:

| Field | Content |
|---|---|
| `boxes` | bounding box `[x1, y1, x2, y2]` in image pixels |
| `contours` | the outline, as `[[x0, x1, ...], [y0, y1, ...]]` in mask pixels (scale by `image_width / mask_width`) |
| `confs` | confidence, 0 to 1 |
| `scales` | the pyramid level the animal was found at |
| `image_path`, `image_width`, `image_height` | the image the results belong to |

`fb_predict` also writes a single COCO file for the whole run, which most annotation and
evaluation tools can read.

## Adjust the predictor

The predictor's settings live in a small YAML file passed with `--config` (or `cfg=` in Python).
The ones most worth knowing:

| Setting | Default | Meaning |
|---|---|---|
| `SCORE_THRESHOLD` | 0.2 | drop detections less confident than this |
| `OVERLAP_THRESHOLD` | 0.2 | two detections overlapping more than this are the same animal |
| `OVERLAP_METRIC` | `IoU` | how overlap is measured: `IoU`, or `IoS` (overlap relative to the smaller one) |
| `MIN_MAX_OBJ_SIZE` | `[32, 1e8]` | smallest and largest animal kept, in pixels (square root of the box area) |
| `BATCH_SIZE` | 16 | tiles per GPU batch; lower it if the GPU runs out of memory |

Each model on the [](models.md) page lists the settings it was evaluated with.

## Train a model

Training needs three steps: get the data, turn it into a training corpus, train.

```bash
# 1. Download the annotated datasets (CVAT project + S3 bucket; needs credentials)
fb_clone_data -s secrets.yaml -o data/coco/

# 2. One corpus with a fixed train/validation split
fb_prepare_data -i data/coco/ -o data/yolo/

# 3. Train
fb_train -d data/yolo/ -c my_config.yaml
```

The validation split is decided by each image's content (an md5 hash of its bytes): an image
always lands on the same side, whichever machine prepares the corpus.

### Every model can be traced to how it was made

`fb_train` refuses to start from a flat-bug checkout with uncommitted changes or untracked
files, so that every set of weights can be traced to a commit. If you really need to train
from a modified checkout, pass `--allow-dirty` (or set `fb_allow_dirty: true` in the config):
the changes are then saved next to the training manifest as `code.diff`.

Each training run writes, next to its weights:

- `manifest.train.yaml`: the flat-bug commit, the command line, the config file as written, the
  resolved training settings, the software and hardware, a per-dataset summary of the data, and,
  when training ends, the checksums of `best.pt` and `last.pt`;
- `data_inventory.csv.gz`: every training and validation image with the checksums of the image
  and its labels, so the exact data can be checked later.

## Benchmark a model

`scripts/benchmark/` holds the tools that produce the numbers on the [](models.md) page. A
benchmark run takes three inputs, each identified by its checksum or commit:

- the **benchmark bundle**: the validation split of the published flat-bug dataset;
- a **model bundle**: the weights and the manifest of how they were trained;
- a **flat-bug commit** to run the model with.

```bash
uv run scripts/benchmark/run_benchmark.py \
    --bundle flatbug-bench-v1.zip --model flatbug-M-2026.09.fbmodel.zip --commit main
```

The runner builds a clean environment for the commit from its lockfile, predicts every
benchmark image, scores the result with the benchmark's own fixed scorer, and writes a report.
Everything is cached, so testing a new commit against the same model only costs the prediction
time. See [scripts/benchmark/README.md](https://github.com/darsa-group/flat-bug/tree/main/scripts/benchmark)
for the details.
