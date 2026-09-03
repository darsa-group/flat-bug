# Known issues in the flat-bug stack

Defects found in `src/flat_bug/` that affect the main pipeline, with how they were measured
so a fix can be verified rather than assumed. Prototype-only problems belong in the
prototype's own docstrings.

---

## 1. Inpainting drew an outline of the ground truth into every crop

**Status:** fixed on `fix/inpaint-halo` (`telea_inpaint_polys`, 2026-09-03). Every model and
every metric produced before this date is affected.

**What happened.** The inpaint mask is built from the polygons to erase *and* the polygons to
keep — the latter deliberately, so `cv2.inpaint` cannot rebuild a deleted animal out of its
neighbour's body. The mask is then dilated so a removed instance's edge cannot bleed into the
fill. Both steps are correct. The defect is that only the *undilated* excluded polygons were
subtracted afterwards, leaving the dilation ring — about `downscale_factor` px at full
resolution — still marked for inpainting. Every instance that SURVIVED the crop therefore
acquired a smeared band tracing its own outline.

**Why it mattered.** `FixInstances` runs in the validation pipeline as well as the training
one, so the marker sat on both sides of the train/val split. A ring hugging every labelled
instance is a cue a model can learn instead of the animal, and then be rewarded for at
validation time.

**Measured.** 0.2-1.1% of each crop altered, concentrated on instance boundaries: on a
regenerated 126 Mpx tile, 12.06% of pixels differed inside a 25px ring outside kept instances
against 0.008% elsewhere, on a 0.000% JPEG re-encode noise floor. Nothing on disk was ever
affected - the function mutates the in-memory crop.

**Fix.** Track the exclusions in their own mask, dilate it with the same kernel, and subtract
that. A 25px band outside kept instances goes from 43.9% of pixels altered to 0.0%, while
instances being removed stay 100% inpainted.

**Verified by a controlled A/B** (jobs 1724716 / 1724717, yolo26m-seg, 50 epochs, stock
settings, source trees differing in one file):

| measured on                   | before (halo) | after (fix) |
|-------------------------------|---------------|-------------|
| box mAP50-95, val CROPS       | 0.859         | 0.781       |
| mask mAP50-95, val CROPS      | 0.750         | 0.655       |
| F1, end-to-end whole images   | 0.849         | **0.878**   |
| recall, end-to-end            | 0.850         | **0.895**   |
| touching-instance recall      | 0.749         | **0.819**   |
| merge rate (lower better)     | 0.156         | **0.142**   |
| GT instances matched          | 23,440        | **24,684**  |

The fix looks like an 8-point mAP regression on the cropped benchmark and is a decisive
improvement in real inference, because the cropped benchmark carried the artefact. F1 rises
on 26 of 31 sub-datasets; the largest gains are on the weakest ones (broto2025 +0.187,
BugNet +0.099, AMI-traps +0.081). PeMaToEuroPep (-0.024) and Diopsis (-0.014) regress on
precision.

**Consequences.** No checkpoint trained before this is a valid baseline, and no metric
computed before it is comparable with one computed after.

---

## 2. The scale pyramid silently skipped native resolution

**Status:** NOT fixed on this branch, deliberately - it is an inference-time change and was
kept out so the halo fix could be attributed alone. Fixed on
`fix/inpaint-halo-and-pyramid-scale`.

**What happened.** `pyramid_predictions` decides whether to add native scale by testing the
ladder's exit value rather than the ladder itself:

```python
while s <= 0.9:
    scales.append(s)
    s /= scale_increment     # s grows by 1.5x
if s != 1:                   # tests `s`, not `scales`
    scales.append(1.0)
```

The loop can only append values <= 0.9, so 1.0 is never already present and should always be
added. Where `max_dim == TILE_SIZE * 1.5**k` the exit value lands exactly on 1.0 and full
resolution was never run - **1024, 1536, 2304, 3456 and 5184 px** at the default 1024 tile,
and 5184x3456 is a stock DSLR frame.

**Consequence.** The pyramid is the only way to find objects larger than a tile and helps in
every size bucket (recall 0.864 vs 0.847 single-scale). For affected sizes the finest level
was missing, so those benchmarks are pessimistic by an unknown amount.

**Fix.** `if 1.0 not in scales: scales.append(1.0)`. Verified across image sizes: 1536, 2304,
3456 and 5184 px go from missing native scale to including it, all others unchanged.

**History.** Already fixed in the M2F prototype's predictor but never ported back.
