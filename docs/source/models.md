# Models

A flat-bug **model** is a set of weights plus the record of how it was made: the code commit,
the training settings, the exact images it was trained on. Every released model is listed here,
with how it scores on the [benchmark](#benchmark) and the predictor settings it was evaluated
with.

Use a model by name; it is downloaded from the model zoo on first use:

```bash
fb_predict -i my_images/ -o results/ -w flat_bug_M_v2.pt
```

```python
Predictor(model="flat_bug_M_v2.pt", device="cuda:0")
```

Click a model for its scores on each benchmark dataset, how it was trained, and a comparison
with any other benchmarked model.

```{raw} html
:file: _generated/models_table.html
```

(benchmark)=
## Benchmark

Scores on this page come from one fixed benchmark, so that any two models, or two versions of
the code, can be compared directly. It is the validation split of the
[dataset published with the paper](https://doi.org/10.5281/zenodo.14761446): 23 datasets
from traps, scanners and collections. Images are predicted whole, as you would use flat-bug,
not tile by tile.

A detection counts when its outline overlaps a hand-drawn one by at least 50% (intersection
over union), each animal matched at most once; animals under 32 pixels across are left out on
both sides. **Precision** is the share of detections that are real animals, **recall** the
share of animals that are found, and **F1** combines the two. **Mask IoU** is how closely the
outlines of the animals that were found follow the hand-drawn ones.

**What it does not measure.** The benchmark covers the 23 datasets of the paper, and the paper
models (`flat_bug_N`, `S`, `M`, `L`) were trained on exactly those datasets, with these images held
out. Later models were trained on a wider corpus (40 datasets in 2026: new traps, scanners and
field cameras, and box-only data) that this benchmark does not test. Compare models on it for
what it covers; a gain on the newer kinds of images will not show here.

The benchmark, the code that scores it, and the result of every run are versioned and
checksummed: see [Benchmark a model](user_guide.md#benchmark-a-model).

```{include} _generated/models_toc.md
```
