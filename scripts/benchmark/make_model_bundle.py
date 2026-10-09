"""Package flat-bug weights and how they were made into a model bundle.

    python scripts/benchmark/make_model_bundle.py best.pt --name flatbug-M-2026.09 \
        --commit 0a75d69 --train-config scripts/training/fb_config_500_M_power.yaml \
        --inference-config refine_on.yaml --extra provenance.yaml -o flatbug-M-2026.09.fbmodel.zip

Run it in an environment with torch and ultralytics: it opens the checkpoint to copy out what
ultralytics stored there (train_args, final validation metrics, ultralytics version, date).

A model bundle is a zip holding:

    manifest.yaml       everything about the model: identity, weights checksum, how it was
                        trained (commit, config, data), its final validation metrics
    weights.pt          the checkpoint, unchanged
    inference.yaml      (optional) the flat-bug predict config the model is meant to be run with;
                        the benchmark passes it to fb_predict --config

The manifest's ``training.commit`` is the flat-bug commit that TRAINED the model. Which commit
is used to RUN it is a separate question, answered per benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from datetime import datetime, timezone

import yaml

SCHEMA = "flatbug-model/1"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**24), b""):
            h.update(b)
    return h.hexdigest()


def plain(x):
    """Checkpoint values as YAML-safe builtins."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def from_checkpoint(path: str) -> dict:
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = plain(ck.get("train_args") or {})
    res = ck.get("train_results") or {}
    out = {
        "checkpoint_date": ck.get("date"),
        "ultralytics_version": ck.get("version"),
        "license": ck.get("license"),
        "train_args": args,
        "final_val_metrics": plain(ck.get("train_metrics") or {}),
        "epochs_completed": len(res.get("epoch", [])) or None,
        "git_recorded_in_checkpoint": plain(ck.get("git")),
    }
    return out


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("weights")
    ap.add_argument("--name", required=True, help="model name, e.g. flatbug-M-2026.09")
    ap.add_argument("--training-manifest",
                    help="the run's manifest.train.yaml (fb_train writes it); supplies commit, config, data "
                         "and environment, and its data_inventory.csv.gz is shipped in the bundle")
    ap.add_argument("--commit", help="flat-bug commit that trained the model (when there is no training manifest)")
    ap.add_argument("--train-config", help="the fb_train config YAML used (when there is no training manifest)")
    ap.add_argument("--inference-config", help="flat-bug predict config to ship with the model")
    ap.add_argument("--extra", help="YAML merged into the manifest last (data, hardware, notes, ...)")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    tm = yaml.safe_load(open(a.training_manifest)) if a.training_manifest else None
    if tm is None and not a.commit:
        raise SystemExit("give --training-manifest, or --commit for weights trained before manifests existed")
    if tm is not None:
        a.commit = tm["code"]["commit"]
        if tm["code"].get("dirty"):
            print(f"WARNING: trained from a modified checkout of {a.commit[:12]}; "
                  f"see training.code.diff_file in the manifest")
    inventory = None
    if tm is not None and tm.get("data", {}).get("inventory_file"):
        inventory = os.path.join(os.path.dirname(a.training_manifest), tm["data"]["inventory_file"])

    ck = from_checkpoint(a.weights)
    args = ck.pop("train_args")
    manifest = {
        "schema": SCHEMA,
        "name": a.name,
        "bundled": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": {"file": "weights.pt", "sha256": sha256(a.weights), "bytes": os.path.getsize(a.weights)},
        "model": {
            "framework": "ultralytics",
            "ultralytics_version": ck["ultralytics_version"],
            "task": args.get("task"),
            "base_model": args.get("model"),
            "image_size": args.get("imgsz"),
            "classes": ["insect"],
        },
        "training": {
            "commit": a.commit,
            "checkpoint_date": ck["checkpoint_date"],
            "epochs": args.get("epochs"),
            "epochs_completed": ck["epochs_completed"],
            "run_name": args.get("name"),
            "data": args.get("data"),
            "config": yaml.safe_load(open(a.train_config)) if a.train_config else None,
            "train_args": args,
            "git_recorded_in_checkpoint": ck["git_recorded_in_checkpoint"],
        },
        "final_val_metrics": ck["final_val_metrics"],
        "inference": {"config_file": "inference.yaml" if a.inference_config else None},
        "license": ck["license"],
    }
    if tm is not None:
        manifest["training"].update({
            "config": (tm.get("config") or {}).get("contents"),
            "code": tm["code"],
            "environment": tm.get("environment"),
            "run": tm.get("run"),
            "data": tm.get("data"),
            "result": tm.get("result"),
            "resumes": tm.get("resumes"),
        })
        if inventory:
            manifest["training"]["data"]["inventory_file"] = "data_inventory.csv.gz"
    if a.extra:
        manifest = deep_merge(manifest, yaml.safe_load(open(a.extra)) or {})

    files = {"manifest.yaml": yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).encode()}
    if a.inference_config:
        files["inference.yaml"] = open(a.inference_config, "rb").read()
    if inventory:
        files["data_inventory.csv.gz"] = open(inventory, "rb").read()
    if tm is not None and tm["code"].get("diff_file"):
        files["code.diff"] = open(os.path.join(os.path.dirname(a.training_manifest), tm["code"]["diff_file"]), "rb").read()
    tmp = a.output + ".tmp"
    with zipfile.ZipFile(tmp, "w") as z:
        for n in sorted(files):
            i = zipfile.ZipInfo(n, FIXED_TIME)
            i.compress_type, i.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            z.writestr(i, files[n])
        i = zipfile.ZipInfo("weights.pt", FIXED_TIME)
        i.compress_type, i.external_attr = zipfile.ZIP_STORED, 0o644 << 16
        with open(a.weights, "rb") as src, z.open(i, "w") as dst:
            for b in iter(lambda: src.read(2**24), b""):
                dst.write(b)
    os.replace(tmp, a.output)
    h = sha256(a.output)
    with open(a.output + ".sha256", "w") as f:
        f.write(f"{h}  {os.path.basename(a.output)}\n")
    print(f"{a.output}\nsha256 {h}\nweights sha256 {manifest['weights']['sha256']}")


if __name__ == "__main__":
    main()
