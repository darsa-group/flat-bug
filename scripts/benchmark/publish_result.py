# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Put a benchmark run on the website.

    uv run scripts/benchmark/publish_result.py <run folder> [--model-bundle <bundle.zip>] [--name <model name>]

Copies into docs/source/_static/models/:

    results/<name>.json     the run's scores and provenance - overall and per dataset - without
                            the per-image rows or polygons, so it stays a few KB
    manifests/<name>.yaml   the model bundle's manifest, when the bundle is given

The model must already be in models.json (the registry): this publishes a RESULT for it. A run
that is not valid (a subset of datasets, missing predictions, or benchmark images found in the
model's training data) is refused unless --force is given, and then marked as such.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outline import simplify  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs" / "source" / "_static" / "models"


def leak_text(lk: dict) -> str:
    if not lk.get("checked"):
        return f"not checked: {lk.get('reason', 'no training inventory')}"
    return (f"{lk['train_overlap']} benchmark images in the model's training data, "
            f"{lk['val_overlap']} in its validation split")


def export_tiles(run: Path, name: str, bench: dict) -> Path | None:
    """This model's outlines on the benchmark's example tiles (see make_tiles.py).

    For each tile: every prediction that reaches into it, in tile pixels, with its confidence and
    the IoU of its match (None for a false detection), and for each animal in view the IoU at
    which it was found (None if missed). Predictions are clipped by the page, not here.
    """
    spec = SITE / "tiles" / bench["name"] / "tiles.json"
    if not spec.exists():
        return None
    meta = json.loads(spec.read_text())
    if meta["benchmark_sha256"] != bench["sha256"]:
        raise SystemExit(f"{spec} was made for another version of {bench['name']}")
    by_image = {(t["dataset"], t["image"]): t for t in meta["tiles"]}
    out = {}
    with gzip.open(run / "predictions.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            t = by_image.get((rec["dataset"], rec["image"]))
            if t is None:
                continue
            x0, y0, w, h = t["window"]
            s = t["scale"]
            preds = []
            for p in rec["pred"]:
                xs, ys = p["xy"][0::2], p["xy"][1::2]
                if max(xs) < x0 or min(xs) > x0 + w or max(ys) < y0 or min(ys) > y0 + h:
                    continue
                xy = simplify([(v - (x0 if k % 2 == 0 else y0)) * s for k, v in enumerate(p["xy"])])
                preds.append({"xy": xy, "conf": p["conf"], "iou": p["iou"]})
            out[t["id"]] = {"pred": preds, "gt_iou": [rec["gt"][g["i"]]["iou"] for g in t["gt"]]}
    path = SITE / "tiles" / bench["name"] / f"{name}.json"
    path.write_text(json.dumps({"schema": "flatbug-site-tile-predictions/1", "model": name, "tiles": out},
                               separators=(",", ":")))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a run folder written by run_benchmark.py")
    ap.add_argument("--model-bundle", help="the model bundle, to publish its manifest too")
    ap.add_argument("--name", help="model name on the site (default: the name in the run)")
    ap.add_argument("--force", action="store_true", help="publish a run that is not valid")
    ap.add_argument("--leak-note", help="how leakage was checked by other means, when the bundle has no inventory")
    a = ap.parse_args()

    r = json.loads((Path(a.run) / "results.json").read_text())
    name = a.name or r["model"]["name"]
    reg = json.loads((SITE / "models.json").read_text())
    if name not in {m["name"] for m in reg["models"]}:
        raise SystemExit(f"{name} is not in {SITE / 'models.json'}; add it to the registry first")
    if not r["valid"] and not a.force:
        raise SystemExit("this run is not a valid benchmark score (see results.json 'valid'); --force to publish anyway")

    slim = {
        "schema": "flatbug-site-result/1",
        "name": name,
        "valid": r["valid"],
        "benchmark": {k: r["benchmark"][k] for k in ("name", "version", "sha256")},
        "scorer": r["scorer"],
        "model": {k: r["model"][k] for k in ("name", "bundle_sha256", "weights_sha256", "training_commit", "inference_config")},
        "code": {k: v for k, v in r["code"].items() if k in ("commit", "subject", "author_date", "environment")},
        "run": {k: r["run"][k] for k in ("started", "finished", "device")},
        "leakage_text": a.leak_note or leak_text(r.get("leakage", {})),
        "overall": r["overall"],
        "datasets": r["datasets"],
    }
    if r.get("timing"):
        t = r["timing"]
        slim["timing"] = {k: t[k] for k in ("device", "complete", "seconds", "images", "megapixels",
                                            "seconds_per_image", "seconds_per_megapixel", "measure")}
        for d, x in t["datasets"].items():
            if d in slim["datasets"] and x.get("seconds"):
                slim["datasets"][d]["seconds_per_image"] = round(x["seconds"] / x["images"], 3)
    slim["code"]["environment"] = {k: v for k, v in slim["code"]["environment"].items() if k != "flat_bug_file"}
    (SITE / "results").mkdir(exist_ok=True)
    out = SITE / "results" / f"{name}.json"
    out.write_text(json.dumps(slim, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}  (F1 {r['overall']['f1']:.4f})")

    tiles = export_tiles(Path(a.run), name, r["benchmark"])
    if tiles:
        print(f"wrote {tiles.relative_to(ROOT)}")

    if a.model_bundle:
        with zipfile.ZipFile(a.model_bundle) as z:
            text = z.read("manifest.yaml").decode()
        (SITE / "manifests").mkdir(exist_ok=True)
        mf = SITE / "manifests" / f"{name}.yaml"
        mf.write_text(text)
        print(f"wrote {mf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
