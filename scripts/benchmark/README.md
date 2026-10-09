# Benchmark

A fixed, versioned end-to-end benchmark for flat-bug models and code. Three artefacts, each a
zip identified by its sha256:

| Artefact | Made by | Contents |
|---|---|---|
| **Benchmark bundle** `flatbug-bench-v1.zip` | `build_bundle.py` | validation split of the paper dataset: images, COCO ground truth, `BENCHMARK.yaml`, `SHA256SUMS` |
| **Model bundle** `<name>.fbmodel.zip` | `make_model_bundle.py` | `weights.pt`, `manifest.yaml` (how it was trained), optional `inference.yaml` |
| **Run** (folder) | `run_benchmark.py` | `results.json`, `per_image.csv`, `predictions.jsonl.gz`, `packages.txt`, `report.html`, logs |

A run answers one question: how well does **this model**, run by **this flat-bug commit**,
do on **this benchmark**? Changing any of the three gives a different run, so new code can be
tested against a fixed model and a new model against fixed code.

## The benchmark

`flatbug-bench-v1` is the validation split of the dataset published with the flat-bug paper
(Zenodo, concept DOI [10.5281/zenodo.14761446](https://doi.org/10.5281/zenodo.14761446)): 23
datasets, CC-BY-4.0. The split is the one `fb_prepare_data` makes, a pure function of the image
bytes (`int(md5[:4], 16) / 0xffff < 0.15`), so the bundle can be rebuilt from the published
zip and checked byte for byte:

    uv run scripts/benchmark/build_bundle.py flatbug-dataset.zip -o flatbug-bench-v1.zip

The bundle is deterministic, so a rebuild has the same sha256.

## Scoring (scorer v1)

`scoring.py` is the benchmark's own scorer; the code under test never scores itself.

* Instances under 32 px (geometric mean of the box sides) are dropped from both sides.
* Predictions are matched one-to-one to ground truth by mask IoU, greedily, best pair first;
  a pair counts at IoU >= 0.5.
* Precision, recall and F1 are pooled over instances, per dataset and overall. Mean IoU of
  the matched pairs measures mask quality once an animal is found.

Changing any of this means a new `SCORER_VERSION`; scores from different versions are not compared.

## Running

    uv run scripts/benchmark/run_benchmark.py \
        --bundle <url or path to flatbug-bench-v1.zip> --bundle-sha256 <sha256> \
        --model  <url or path to model bundle>        --model-sha256  <sha256> \
        --commit <flat-bug commit> [--repo <url or local clone>] [--baseline <results.json>]

The runner works like a CI job. It builds its own sandbox for the commit: the commit is exported
from a mirror of the repository and installed with `uv sync --frozen` from that commit's
`uv.lock`. Prediction runs that environment's `fb_predict`, with the model's `inference.yaml`
if it ships one. Nothing from the calling environment is used.

Everything is cached under `~/.cache/flatbug-bench` (or `$FLATBUG_BENCH_CACHE`), keyed by sha256 or
commit: downloads, unpacked bundles, the repository mirror, per-commit source trees and
environments, and predictions. uv's package cache is shared by all environments. In practice:

* first run on a machine: download the bundles and build one environment (~5 min plus downloads), then predict;
* a new commit: one more environment, in seconds from uv's cache, then predict;
* the same three inputs again: re-score only.

`--datasets a,b` runs a subset for a quick check; such a run is marked `"valid": false`.

## Model bundles

    python scripts/benchmark/make_model_bundle.py best.pt --name flatbug-M-2026.09 \
        --commit <training commit> --train-config <fb_train yaml> \
        --inference-config <predict yaml> --extra <provenance yaml> -o flatbug-M-2026.09.fbmodel.zip

Run it with torch and ultralytics available: it copies what ultralytics stored in the checkpoint
(train_args, final validation metrics, versions, date) into `manifest.yaml`. Whatever the
checkpoint cannot know goes in `--extra`: the training data, hardware, notes.
`training.commit` is the commit that trained the model; the commit that runs it is chosen per run.
