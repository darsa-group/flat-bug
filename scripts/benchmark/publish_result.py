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
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs" / "source" / "_static" / "models"


def leak_text(lk: dict) -> str:
    if not lk.get("checked"):
        return f"not checked: {lk.get('reason', 'no training inventory')}"
    return (f"{lk['train_overlap']} benchmark images in the model's training data, "
            f"{lk['val_overlap']} in its validation split")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a run folder written by run_benchmark.py")
    ap.add_argument("--model-bundle", help="the model bundle, to publish its manifest too")
    ap.add_argument("--name", help="model name on the site (default: the name in the run)")
    ap.add_argument("--force", action="store_true", help="publish a run that is not valid")
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
        "leakage_text": leak_text(r.get("leakage", {})),
        "overall": r["overall"],
        "datasets": r["datasets"],
    }
    slim["code"]["environment"] = {k: v for k, v in slim["code"]["environment"].items() if k != "flat_bug_file"}
    (SITE / "results").mkdir(exist_ok=True)
    out = SITE / "results" / f"{name}.json"
    out.write_text(json.dumps(slim, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}  (F1 {r['overall']['f1']:.4f})")

    if a.model_bundle:
        with zipfile.ZipFile(a.model_bundle) as z:
            text = z.read("manifest.yaml").decode()
        (SITE / "manifests").mkdir(exist_ok=True)
        mf = SITE / "manifests" / f"{name}.yaml"
        mf.write_text(text)
        print(f"wrote {mf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
