# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Build the flat-bug benchmark bundle from the published flat-bug dataset.

    uv run scripts/benchmark/build_bundle.py flatbug-dataset.zip -o flatbug-bench-v1.zip

The benchmark is the VALIDATION split of the dataset published with the flat-bug paper
(Zenodo, concept DOI 10.5281/zenodo.14761446). The split is the one fb_prepare_data has always
made, and it is a pure function of the image bytes:

    validation  <=>  int(md5(image bytes)[:4], 16) / 0xffff < 0.15

so anyone holding the published zip can rebuild this bundle and check it. As in fb_prepare_data,
an image takes part only if it is listed in its dataset's COCO file AND present in the folder.

The output zip is deterministic: entries in sorted order, fixed timestamps, JPEGs stored and
text deflated at a fixed level. Building it twice from the same source gives the same bytes,
hence the same sha256, which is what the benchmark runner pins and caches by.

Layout of the bundle:

    flatbug-bench-<version>/
        BENCHMARK.yaml          what this is, where it came from, the split and scoring rules,
                                per-dataset counts
        metadata.csv            the source dataset's table, verbatim
        SHA256SUMS              every other file
        <dataset>/instances.json   COCO, restricted to the validation images
        <dataset>/images/<file>.jpg
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from datetime import date

import yaml

SOURCE = {
    "title": "flatbug-dataset: a compilation of datasets of terrestrial arthropods on various surfaces",
    "concept_doi": "10.5281/zenodo.14761446",
    "record": "https://zenodo.org/records/14761447",
    "file": "flatbug-dataset.zip",
    "md5": "c10b7438f93eae8ba855babfa6bae751",
    "license": "CC-BY-4.0",
}
SCORING = {
    "scorer_version": "1",
    "iou_threshold": 0.5,
    "min_size_px": 32,
    "size_definition": "geometric mean of the bounding-box width and height, in full-image pixels",
    "matching": "one-to-one, greedy by mask IoU, best pair first",
    "metrics": "precision, recall and F1 pooled over instances (micro-average), overall and per dataset; "
               "mean IoU of matched pairs",
    "reference_implementation": "scripts/benchmark/scoring.py",
}
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
IMAGE_EXT = (".jpg", ".jpeg")


def is_validation(data: bytes, proportion: float) -> bool:
    """fb_prepare_data's split rule."""
    return int(hashlib.md5(data).hexdigest()[0:4], 16) / int("ffff", 16) < proportion


def add(z: zipfile.ZipFile, name: str, data: bytes, sums: dict[str, str]):
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.external_attr = 0o644 << 16
    if name.lower().endswith(IMAGE_EXT):
        info.compress_type = zipfile.ZIP_STORED
    else:
        info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, data, compresslevel=9 if info.compress_type == zipfile.ZIP_DEFLATED else None)
    sums[name] = hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", help="the published flatbug-dataset.zip")
    ap.add_argument("-o", "--output", required=True, help="output zip, e.g. flatbug-bench-v1.zip")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--proportion", type=float, default=0.15, help="fb_prepare_data's validation proportion")
    ap.add_argument("--skip-md5", action="store_true", help="do not verify the source zip's md5")
    a = ap.parse_args()

    if not a.skip_md5:
        h = hashlib.md5()
        with open(a.source, "rb") as f:
            for b in iter(lambda: f.read(2**24), b""):
                h.update(b)
        if h.hexdigest() != SOURCE["md5"]:
            raise SystemExit(f"{a.source}: md5 {h.hexdigest()} is not the published {SOURCE['md5']}")

    src = zipfile.ZipFile(a.source)
    names = [n for n in src.namelist() if not n.endswith("/")]
    root = names[0].split("/")[0]
    datasets = sorted({n.split("/")[1] for n in names if n.count("/") == 2})
    meta_csv = src.read(f"{root}/metadata.csv")
    meta = {r["dataset"]: r for r in csv.DictReader(io.StringIO(meta_csv.decode()))}

    top = f"flatbug-bench-{a.version}"
    tmp = a.output + ".tmp"
    sums: dict[str, str] = {}
    table = []
    with zipfile.ZipFile(tmp, "w") as z:
        add(z, f"{top}/metadata.csv", meta_csv, sums)
        for d in datasets:
            jsons = [n for n in names if n.startswith(f"{root}/{d}/") and n.endswith(".json")]
            if len(jsons) != 1:
                raise SystemExit(f"{d}: expected one COCO file, found {jsons}")
            coco = json.loads(src.read(jsons[0]))
            files = {os.path.basename(n): n for n in names
                     if n.startswith(f"{root}/{d}/") and n.lower().endswith(IMAGE_EXT)}
            keep_ids, images = set(), []
            for im in sorted(coco["images"], key=lambda im: os.path.basename(im["file_name"])):
                base = os.path.basename(im["file_name"])
                if base not in files:
                    continue
                data = src.read(files[base])
                if not is_validation(data, a.proportion):
                    continue
                add(z, f"{top}/{d}/images/{base}", data, sums)
                keep_ids.add(im["id"])
                images.append({**im, "file_name": base})
            anns = [an for an in coco["annotations"] if an["image_id"] in keep_ids]
            n_with = len({an["image_id"] for an in anns})
            out = {k: v for k, v in coco.items() if k not in ("images", "annotations")}
            out["images"], out["annotations"] = images, anns
            add(z, f"{top}/{d}/instances.json", json.dumps(out, separators=(",", ":")).encode(), sums)
            m = meta.get(d, {})
            table.append({
                "name": d,
                "short_name": m.get("short_name", ""),
                "doi": m.get("DOI_data_new", ""),
                "images": len(images),
                "images_without_annotations": len(images) - n_with,
                "instances": len(anns),
                "source_images": len(files),
            })
            print(f"{d:28s} {len(images):5d} / {len(files):5d} images  {len(anns):6d} instances", flush=True)

        spec = {
            "name": top,
            "version": a.version,
            "built": date.today().isoformat(),
            "description": "Validation split of the dataset published with the flat-bug paper, "
                           "used as a fixed end-to-end benchmark for flat-bug models and code.",
            "source": SOURCE,
            "split": {
                "rule": "int(md5(image bytes)[:4], 16) / 0xffff < proportion",
                "proportion": a.proportion,
                "membership": "image listed in its dataset's COCO file and present in the dataset folder",
                "reference_implementation": "src/flat_bug/cli/fb_prepare_data.py",
            },
            "scoring": SCORING,
            "totals": {
                "datasets": len(table),
                "images": sum(t["images"] for t in table),
                "instances": sum(t["instances"] for t in table),
            },
            "datasets": table,
            "license": "CC-BY-4.0, as the source; cite the source dataset and each dataset's own DOI",
        }
        add(z, f"{top}/BENCHMARK.yaml", yaml.safe_dump(spec, sort_keys=False, allow_unicode=True).encode(), sums)
        lines = "".join(f"{h}  {n[len(top) + 1:]}\n" for n, h in sorted(sums.items()))
        add(z, f"{top}/SHA256SUMS", lines.encode(), {})
    os.replace(tmp, a.output)

    h = hashlib.sha256()
    with open(a.output, "rb") as f:
        for b in iter(lambda: f.read(2**24), b""):
            h.update(b)
    with open(a.output + ".sha256", "w") as f:
        f.write(f"{h.hexdigest()}  {os.path.basename(a.output)}\n")
    t = spec["totals"]
    print(f"\n{a.output}: {t['datasets']} datasets, {t['images']} images, {t['instances']} instances")
    print(f"sha256 {h.hexdigest()}")


if __name__ == "__main__":
    main()
