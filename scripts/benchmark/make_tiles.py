# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pillow", "pyyaml"]
# ///
"""Choose the example tiles of a benchmark, once, and put them on the website.

    uv run scripts/benchmark/make_tiles.py <benchmark folder> <reference run folder>

For every dataset, one EASY and one HARD image as the reference model sees them - its best and
worst F1 among images with at least --min-animals animals - so a page shows both what flat-bug
does well and where it fails. The reference should be the released default model at the time
the tiles are chosen. The choice is then frozen: every model is shown on the same tiles, which
is what makes them comparable, and it is recorded in tiles.json with the reference it came from.

An image larger than --window pixels is cropped to the window holding the most of what matters:
the most errors (missed animals and false detections) for a hard case, the most found animals
for an easy one. Tiles are saved as JPEGs no wider than --size pixels.

Writes docs/source/_static/models/tiles/<benchmark>/:

    tiles.json        each tile's dataset, image, window, scale, kind, and its ground truth
                      outlines in tile pixels
    <id>.jpg          the tile
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outline import simplify  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # benchmark scans are trusted files, some over 100 MP
ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "docs" / "source" / "_static" / "models" / "tiles"


def centre(xy):
    a = np.asarray(xy, float).reshape(-1, 2)
    return a.mean(0)


def best_window(points: np.ndarray, weights: np.ndarray, W: int, H: int, win: int) -> tuple[int, int, int, int]:
    """The win x win window (clipped to the image) containing the largest total weight."""
    if W <= win and H <= win:
        return 0, 0, W, H
    w, h = min(win, W), min(win, H)
    best, at = -1.0, (0, 0)
    step = max(32, win // 8)
    for y0 in range(0, max(1, H - h) + 1, step):
        for x0 in range(0, max(1, W - w) + 1, step):
            inside = ((points[:, 0] >= x0) & (points[:, 0] < x0 + w) &
                      (points[:, 1] >= y0) & (points[:, 1] < y0 + h)) if len(points) else np.zeros(0, bool)
            s = float(weights[inside].sum()) if len(points) else 0.0
            if s > best:
                best, at = s, (x0, y0)
    return at[0], at[1], w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("benchmark", help="the unpacked benchmark folder (flatbug-bench-<version>/)")
    ap.add_argument("reference_run", help="run_benchmark.py output of the reference model")
    ap.add_argument("--min-animals", type=int, default=3)
    ap.add_argument("--window", type=int, default=1024, help="crop window in image pixels")
    ap.add_argument("--size", type=int, default=900, help="largest tile side in the saved JPEG")
    a = ap.parse_args()

    bench = Path(a.benchmark)
    spec = yaml.safe_load((bench / "BENCHMARK.yaml").read_text())
    run = Path(a.reference_run)
    ref = json.loads((run / "results.json").read_text())
    if ref["benchmark"]["name"] != spec["name"]:
        raise SystemExit(f"the run is on {ref['benchmark']['name']}, not {spec['name']}")

    every = defaultdict(list)
    with open(run / "per_image.csv") as f:
        for r in csv.DictReader(f):
            if int(r["gt"]) >= 1 and r["f1"] not in ("", "nan"):
                every[r["dataset"]].append((float(r["f1"]), int(r["gt"]), r["image"]))
    # Prefer images with a few animals; datasets of one animal per image (specimen photos) have none.
    rows = {ds: ([t for t in rs if t[1] >= a.min_animals] or rs) for ds, rs in every.items()}
    picks = {}
    for ds, rs in rows.items():
        rs.sort(key=lambda t: (t[0], -t[1]))  # worst F1 first; among ties, the busiest image
        hard = rs[0]
        picks[(ds, hard[2])] = "hard"
        rest = rs[1:]  # when every image scores the same, easy is still a different picture
        if rest:
            picks[(ds, max(rest, key=lambda t: (t[0], t[1]))[2])] = "easy"

    out = SITE / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    with gzip.open(run / "predictions.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            kind = picks.get((rec["dataset"], rec["image"]))
            if kind is None:
                continue
            im = Image.open(bench / rec["dataset"] / "images" / rec["image"])
            W, H = im.size
            # What should be in view: errors for a hard case, successes for an easy one.
            pts, wts = [], []
            for g in rec["gt"]:
                pts.append(centre(g["xy"]))
                wts.append(1.0 if (g["match"] < 0) == (kind == "hard") else 0.1)
            for p in rec["pred"]:
                if p["match"] < 0:
                    pts.append(centre(p["xy"]))
                    wts.append(1.0 if kind == "hard" else 0.0)
            x0, y0, w, h = best_window(np.array(pts).reshape(-1, 2), np.array(wts), W, H, a.window)
            s = min(1.0, a.size / max(w, h))
            tid = f"{rec['dataset']}__{kind}"
            crop = im.convert("RGB").crop((x0, y0, x0 + w, y0 + h))
            if s < 1:
                crop = crop.resize((round(w * s), round(h * s)), Image.LANCZOS)
            crop.save(out / f"{tid}.jpg", quality=80, optimize=True)

            def to_tile(xy):
                q = np.asarray(xy, float).reshape(-1, 2)
                return simplify(((q - [x0, y0]) * s).ravel().tolist())

            # Ground truth in view, with its index in the image so each model can mark it found or missed.
            gt = [{"i": i, "xy": to_tile(g["xy"])} for i, g in enumerate(rec["gt"])
                  if x0 - 50 <= centre(g["xy"])[0] <= x0 + w + 50 and y0 - 50 <= centre(g["xy"])[1] <= y0 + h + 50]
            tiles.append({"id": tid, "dataset": rec["dataset"], "image": rec["image"], "kind": kind,
                          "window": [x0, y0, w, h], "scale": s, "size": list(crop.size),
                          "reference_f1": next(t[0] for t in rows[rec["dataset"]] if t[2] == rec["image"]),
                          "gt": gt})
            print(f"{tid:40s} {rec['image'][:40]:40s} window {w}x{h}  {len(gt)} animals")

    tiles.sort(key=lambda t: (t["dataset"].lower(), t["kind"] != "easy"))
    meta = {"schema": "flatbug-site-tiles/1", "benchmark": spec["name"],
            "benchmark_sha256": ref["benchmark"]["sha256"],
            "chosen_with": {"model": ref["model"]["name"], "commit": ref["code"]["commit"],
                            "rule": f"per dataset, the reference model's best and worst F1 among images with "
                                    f">= {a.min_animals} animals; window {a.window} px around the errors "
                                    f"(hard) or the found animals (easy)"},
            "tiles": tiles}
    (out / "tiles.json").write_text(json.dumps(meta, separators=(",", ":")))
    size = sum(p.stat().st_size for p in out.glob("*.jpg"))
    print(f"\n{len(tiles)} tiles, {size / 1e6:.1f} MB of JPEG, in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
