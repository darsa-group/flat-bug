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

---

## 3. Synthetic touching scenes do not work — do not enable `fb_synth_prob` by default

**Status:** measured and rejected, 2026-09-06. The code on this branch is correct and stays
here unmerged so it can be revived if the compositor is improved. This section exists so the
next person measures something new instead of rebuilding this.

**The experiment.** 100-epoch A/B, `fb_synth_off` vs `fb_synth_on`, both on `flatbug-dir`,
same `yolo26m-seg.pt`, batch 8, imgsz 1024, seed 0, lr0 0.01. One config line differed:
`fb_synth_prob` 0.0 vs 0.4. Both arms ran to epoch 100; both best at epoch 96. Judged
end-to-end on one identical held-out set (whole validation images, no crops, no inpainting,
instances >=32 px, IoU 0.5) — see the note on why crop metrics cannot settle this.

**On the hypothesis it was built for, it fails.** Synthetic scenes target merge and split
errors on adjacent instances, so the metrics that matter are touch recall and merge rate on
instances whose gap to the nearest neighbour is <= 0 px:

|                              | control | synth  | delta   |
|------------------------------|---------|--------|---------|
| touch recall (n=5452)        | 0.8199  | 0.8085 | -0.0114 |
| merge rate                   | 0.1462  | 0.1381 | -0.0081 |
| touch recall, ex broto2025   | 0.8877  | 0.8904 | +0.0027 |
| merge rate,   ex broto2025   | 0.1351  | 0.1401 | **+0.0050** |

The pooled merge-rate "improvement" is entirely `broto2025`, where the synth arm detects far
less (touch recall 0.306 -> 0.191) and therefore merges less. Excluding it — 4819 touching
instances across 15 datasets — touch recall is unchanged and **merging is worse**.

**Overall it is a null.** Pooled end-to-end F1 0.8863 -> 0.8852 (-0.0011), 11 datasets
improved and 17 worsened of 31. Crop mask mAP50-95 at the matched epoch 96: 0.66405 ->
0.66336.

**The cost is real.** 29.3 min/epoch against 15.9, so **1.8x wall clock** — 42.5 h vs 26.5 h
for the same result.

**The one effect that survived every cut:** thin-appendage recall **+0.0123** (0.5151 ->
0.5274), stable with and without `broto2025` and across both the epoch-86 and epoch-96
checkpoints. Core-body recall +0.0028. Compositing crops does seem to teach legs and wings,
plausibly because pasted specimens have crisp complete outlines the model must reproduce.
That is worth chasing separately and far more cheaply than scene composition.

**Why it does not transfer.** Train seg loss ran ~0.07 lower for the synth arm from epoch 20
on while **val seg loss was identical**. Lower training loss with unchanged validation loss
means the composited scenes are *easier* to segment than real crowded ones — sharp paste
boundaries, cleaner separation — rather than regularising. The limiting factor is the
compositor's realism, not the idea.

**For scale, what else bought on the same benchmark:** the inpainting-halo fix (issue 1)
+0.028 F1 for a bug fix; doubling training 50 -> 100 epochs +0.009 F1 for 2x compute.
