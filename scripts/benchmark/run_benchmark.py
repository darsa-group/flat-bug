# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "shapely>=2", "pyyaml", "pillow"]
# ///
"""Benchmark one flat-bug model, run by one flat-bug commit, against the benchmark bundle.

    uv run scripts/benchmark/run_benchmark.py \
        --bundle https://zenodo.org/records/<id>/files/flatbug-bench-v1.zip --bundle-sha256 <sha> \
        --model  flatbug-M-2026.09.fbmodel.zip \
        --commit 6a95fd2

It works like a CI job, in a sandbox it builds itself:

1. BUNDLE and MODEL are fetched (URL or path), checked against their sha256 when one is given,
   and unpacked; the benchmark's own SHA256SUMS is verified once.
2. The COMMIT is exported from a mirror of the repository (``--repo``, a URL or a local clone)
   into a clean source tree, and installed into its own environment with
   ``uv sync --frozen --no-dev --no-editable``: the exact versions in that commit's uv.lock.
   Nothing from the calling environment leaks in.
3. That environment's ``fb_predict`` runs over every benchmark image, with the model's own
   inference.yaml when the bundle ships one.
4. Predictions are scored by scoring.py - the BENCHMARK's scorer, never the commit's own
   evaluation code, so the code under test cannot move its own score.
5. Results go to ``--out``: results.json (scores plus full provenance), per_image.csv,
   the environment's package list, prediction logs and report.html.

Everything expensive is cached under ``--cache`` (default ~/.cache/flatbug-bench, or
$FLATBUG_BENCH_CACHE): downloads and unpacked bundles by sha256, the repository mirror, one
source tree and environment per commit, and predictions per (benchmark, model, commit). uv's
own package cache is shared by all environments, so a second commit installs in seconds.
A re-run with the same three inputs only re-scores.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report  # noqa: E402
import scoring  # noqa: E402

DEFAULT_REPO = "https://github.com/darsa-group/flat-bug.git"


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**24), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- inputs: fetch, verify, unpack
def fetch(src: str, expected: str | None, cache: Path) -> tuple[str, Path]:
    """A local copy of src, verified. Returns (sha256, path)."""
    if expected:
        hit = cache / "downloads" / f"{expected}.zip"
        if hit.exists():
            return expected, hit
    if src.startswith(("http://", "https://")):
        (cache / "downloads").mkdir(parents=True, exist_ok=True)
        tmp = cache / "downloads" / f"partial-{os.getpid()}"
        log(f"downloading {src}")
        with urllib.request.urlopen(src, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 2**24)
        sha = sha256_file(tmp)
        if expected and sha != expected:
            tmp.unlink()
            raise SystemExit(f"{src}: sha256 {sha}, expected {expected}")
        path = cache / "downloads" / f"{sha}.zip"
        os.replace(tmp, path)
        return sha, path
    path = Path(src).expanduser().resolve()
    sha = sha256_file(path)
    if expected and sha != expected:
        raise SystemExit(f"{path}: sha256 {sha}, expected {expected}")
    return sha, path


def unpack(zip_path: Path, dest: Path) -> Path:
    if (dest / ".complete").exists():
        return dest
    tmp = dest.with_name(dest.name + f".partial-{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    (tmp / ".complete").touch()
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return dest


def load_benchmark(src: str, expected: str | None, cache: Path):
    sha, zp = fetch(src, expected, cache)
    root = unpack(zp, cache / "benchmarks" / sha)
    tops = [p for p in root.iterdir() if p.is_dir()]
    if len(tops) != 1:
        raise SystemExit(f"{src}: expected one top-level folder, found {[p.name for p in tops]}")
    top = tops[0]
    if not (root / ".verified").exists():
        log("verifying benchmark SHA256SUMS")
        for line in (top / "SHA256SUMS").read_text().splitlines():
            h, name = line.split("  ", 1)
            if sha256_file(top / name) != h:
                raise SystemExit(f"benchmark file corrupted: {name}")
        (root / ".verified").touch()
    spec = yaml.safe_load((top / "BENCHMARK.yaml").read_text())
    if str(spec["scoring"]["scorer_version"]) != scoring.SCORER_VERSION:
        raise SystemExit(f"benchmark expects scorer v{spec['scoring']['scorer_version']}, "
                         f"this runner has v{scoring.SCORER_VERSION}")
    return sha, top, spec


def load_model(src: str, expected: str | None, cache: Path):
    sha, zp = fetch(src, expected, cache)
    root = unpack(zp, cache / "models" / sha)
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    w = root / manifest["weights"]["file"]
    if not (root / ".verified").exists():
        if sha256_file(w) != manifest["weights"]["sha256"]:
            raise SystemExit(f"{src}: weights do not match the manifest's sha256")
        (root / ".verified").touch()
    inf = manifest.get("inference", {}).get("config_file")
    return sha, root, manifest, w, (root / inf if inf else None)


# ---------------------------------------------------------------- code: mirror, export, install
def checkout(repo: str, rev: str, cache: Path) -> tuple[str, Path, dict]:
    """Export `rev` of `repo` into a clean source tree. Returns (full sha, path, commit info)."""
    key = hashlib.sha256(repo.encode()).hexdigest()[:12]
    mirror = cache / "repos" / f"{key}.git"
    if not mirror.exists():
        log(f"mirroring {repo}")
        mirror.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", "--mirror", repo, str(mirror)])

    def resolve():
        r = subprocess.run(["git", "-C", str(mirror), "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    sha = resolve()
    if sha is None:
        log("commit not in mirror; fetching")
        run(["git", "-C", str(mirror), "fetch", "--quiet", "--prune", "origin", "+refs/*:refs/*"])
        sha = resolve()
    if sha is None:
        raise SystemExit(f"{rev}: no such commit in {repo}")
    info = dict(zip(("subject", "author_date", "author"),
                    run(["git", "-C", str(mirror), "log", "-1", "--format=%s%n%aI%n%an", sha]).splitlines()))
    src = cache / "src" / sha
    if not (src / ".complete").exists():
        tmp = src.with_name(sha + f".partial-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        archive = subprocess.Popen(["git", "-C", str(mirror), "archive", sha], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", str(tmp)], stdin=archive.stdout, check=True)
        if archive.wait() != 0:
            raise SystemExit(f"git archive {sha} failed")
        (tmp / ".complete").touch()
        shutil.rmtree(src, ignore_errors=True)
        os.replace(tmp, src)
    return sha, src, info


def install(sha: str, src: Path, cache: Path, uv: str, python: str | None) -> Path:
    """One environment per commit, from the commit's own lockfile."""
    env = cache / "envs" / sha
    if (env / ".complete").exists():
        return env
    shutil.rmtree(env, ignore_errors=True)
    t0 = time.time()
    e = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(env)}
    e.pop("VIRTUAL_ENV", None)
    py = ["--python", python] if python else []
    if (src / "uv.lock").exists():
        log(f"installing {sha[:12]} from its uv.lock")
        run([uv, "sync", "--frozen", "--no-dev", "--no-editable", *py], cwd=src, env=e)
    else:
        log(f"installing {sha[:12]} (no uv.lock in this commit: resolving fresh)")
        run([uv, "venv", "--quiet", *py, str(env)], env=e)
        run([uv, "pip", "install", "--quiet", "--python", str(env / "bin" / "python"), str(src)], env=e)
    (env / ".complete").touch()
    log(f"environment ready in {time.time() - t0:.0f} s")
    return env


def describe_env(env: Path, uv: str) -> dict:
    py = env / "bin" / "python"
    probe = ("import json, sys, torch, ultralytics, flat_bug\n"
             "print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__,"
             " 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),"
             " 'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
             " 'ultralytics': ultralytics.__version__, 'flat_bug_file': flat_bug.__file__}))")
    d = json.loads(run([str(py), "-c", probe], env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"}))
    if not d["flat_bug_file"].startswith(str(env)):
        raise SystemExit(f"flat_bug imported from {d['flat_bug_file']}, not from the sandbox {env}")
    d["packages"] = run([uv, "pip", "freeze", "--python", str(py)])
    return d


# ---------------------------------------------------------------- predict
def predict(env: Path, weights: Path, inference: Path | None, bench: Path, datasets: list[str],
            pred_root: Path, device: str, logs: Path) -> float:
    fb = env / "bin" / "fb_predict"
    clean = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    logs.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for i, d in enumerate(datasets, 1):
        out = pred_root / d
        if (out / ".complete").exists():
            continue
        shutil.rmtree(out, ignore_errors=True)
        n = len(list((bench / d / "images").iterdir()))
        log(f"predict {i}/{len(datasets)} {d} ({n} images)")
        cmd = [str(fb), "-i", str(bench / d / "images"), "-o", str(out), "-w", str(weights),
               "--no-crops", "--no-overviews", "-C", "-g", device]
        if inference:
            cmd += ["--config", str(inference)]
        t0 = time.time()
        with open(logs / f"{d}.log", "w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=clean)
        spent += time.time() - t0
        if r.returncode != 0:
            raise SystemExit(f"fb_predict failed on {d}; see {logs / f'{d}.log'}")
        (out / ".complete").touch()
    return spent


def leak_check(bench: Path, model_root: Path) -> dict:
    """Which benchmark images the model saw in training, by sha256 of the image bytes.

    Needs the model bundle to ship the training inventory (fb_train writes it since manifests
    exist). An image that sits in the model's VALIDATION split is not a leak - it was held out
    exactly as the benchmark holds it out - but is reported too.
    """
    inv = model_root / "data_inventory.csv.gz"
    if not inv.exists():
        return {"checked": False, "reason": "the model bundle has no training data inventory"}
    with gzip.open(inv, "rt") as f:
        seen = {r["sha256"]: r["split"] for r in csv.DictReader(f)}
    out = {"checked": True, "train_overlap": 0, "val_overlap": 0, "per_dataset": {}, "examples": []}
    for line in (bench / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        if "/images/" not in name or h not in seen:
            continue
        split, ds = seen[h], name.split("/", 1)[0]
        out[f"{split}_overlap"] = out.get(f"{split}_overlap", 0) + 1
        d = out["per_dataset"].setdefault(ds, {"train": 0, "val": 0})
        d[split] = d.get(split, 0) + 1
        if split == "train" and len(out["examples"]) < 20:
            out["examples"].append(name)
    return out


def ring(p) -> list[int]:
    """A polygon's outline as a flat [x0, y0, x1, y1, ...] list of whole pixels."""
    q = p.simplify(0.5, preserve_topology=True)
    return [round(v) for xy in list(q.exterior.coords)[:-1] for v in xy]


def instances_record(dataset: str, image: str, gt, pred, confs, pairs) -> dict:
    """Every scored instance of one image, so views can be rebuilt without re-running anything.

    ``gt[i].match`` / ``pred[j].match`` give the index of the partner on the other side (-1 if
    none), and ``iou`` the pair's IoU. Only instances that passed the scorer's size filter are
    listed: these are exactly what the scores count.
    """
    g_m, p_m = [-1] * len(gt), [-1] * len(pred)
    iou = {}
    for i, j, v in pairs:
        g_m[i], p_m[j], iou[i] = j, i, round(v, 4)
    return {
        "dataset": dataset, "image": image,
        "gt": [{"xy": ring(p), "match": g_m[i], "iou": iou.get(i)} for i, p in enumerate(gt)],
        "pred": [{"xy": ring(p), "conf": round(c, 4), "match": p_m[j],
                  "iou": iou.get(p_m[j]) if p_m[j] >= 0 else None} for j, (p, c) in enumerate(zip(pred, confs))],
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True, help="benchmark bundle zip: URL or path")
    ap.add_argument("--bundle-sha256")
    ap.add_argument("--model", required=True, help="model bundle zip: URL or path")
    ap.add_argument("--model-sha256")
    ap.add_argument("--commit", required=True, help="flat-bug commit (any rev git understands) to run the model with")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"repository to take the commit from (default {DEFAULT_REPO})")
    ap.add_argument("--out", help="results folder (default <cache>/runs/<bench>/<model>/<commit>)")
    ap.add_argument("--cache", default=os.environ.get("FLATBUG_BENCH_CACHE", "~/.cache/flatbug-bench"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--python", help="Python version for the sandbox environment (uv decides by default)")
    ap.add_argument("--uv", default=shutil.which("uv"), help="path to uv")
    ap.add_argument("--datasets", help="comma-separated subset, for a quick check (not a valid score)")
    ap.add_argument("--baseline", help="a previous results.json to compare against in the report")
    a = ap.parse_args()
    if not a.uv:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/getting-started/installation/")

    cache = Path(a.cache).expanduser()
    started = datetime.now(timezone.utc)
    repo = a.repo if "://" in a.repo or a.repo.startswith("git@") else str(Path(a.repo).expanduser().resolve())

    bench_sha, bench, spec = load_benchmark(a.bundle, a.bundle_sha256, cache)
    model_sha, model_root, manifest, weights, inference = load_model(a.model, a.model_sha256, cache)
    leaks = leak_check(bench, model_root)
    if leaks["checked"] and leaks["train_overlap"]:
        log(f"WARNING: {leaks['train_overlap']} benchmark images were in this model's TRAINING data")
    commit, src, info = checkout(repo, a.commit, cache)
    log(f"benchmark {spec['name']} ({bench_sha[:12]})  model {manifest['name']} ({model_sha[:12]})  "
        f"commit {commit[:12]} {info.get('subject', '')}")
    env = install(commit, src, cache, a.uv, a.python)
    envinfo = describe_env(env, a.uv)
    log(f"sandbox: python {envinfo['python']}  torch {envinfo['torch']}  ultralytics {envinfo['ultralytics']}  "
        f"device {envinfo['device_name'] or 'cpu'}")

    datasets = [d["name"] for d in spec["datasets"]]
    if a.datasets:
        want = a.datasets.split(",")
        unknown = set(want) - set(datasets)
        if unknown:
            raise SystemExit(f"not in the benchmark: {sorted(unknown)}")
        datasets = [d for d in datasets if d in want]
    partial = len(datasets) < len(spec["datasets"])

    out = Path(a.out).expanduser() if a.out else (
        cache / "runs" / spec["name"] / manifest["name"] / commit[:12])
    out.mkdir(parents=True, exist_ok=True)
    pred_root = cache / "predictions" / f"{bench_sha[:16]}-{model_sha[:16]}-{commit[:16]}-{a.device.replace(':', '')}"
    seconds = predict(env, weights, inference, bench, datasets, pred_root, a.device, out / "logs")

    log("scoring")
    per_ds, rows, pooled = {}, [], {"gt": 0, "pred": 0, "tp_gt": 0, "tp_pred": 0, "iou_sum": 0.0}
    polys = {}
    with gzip.open(out / "predictions.jsonl.gz", "wt") as pj:
        for d in datasets:
            gt = scoring.gt_polygons(str(bench / d / "instances.json"))
            pr, confs = scoring.pred_polygons(str(pred_root / d))
            summary, r, tot, matches = scoring.score_dataset(gt, pr)
            per_ds[d] = summary
            for k in pooled:
                pooled[k] += tot[k]
            rows += [{"dataset": d, **x} for x in r]
            polys[d] = (gt, pr)
            for name in sorted(gt):
                pj.write(json.dumps(instances_record(d, name, gt[name], pr.get(name, []),
                                                     confs.get(name, []), matches[name])) + "\n")
    missing = sum(1 for r in rows if not r["predicted"])
    overall = scoring.summarise(pooled)

    results = {
        "schema": "flatbug-benchmark-result/1",
        "valid": not partial and missing == 0 and not (leaks["checked"] and leaks["train_overlap"]),
        "benchmark": {"name": spec["name"], "version": spec["version"], "sha256": bench_sha,
                      "source": spec["source"], "datasets_scored": datasets if partial else "all"},
        "scorer": {"version": scoring.SCORER_VERSION, "iou_threshold": scoring.IOU_THRESHOLD,
                   "min_size_px": scoring.MIN_SIZE_PX},
        "model": {"name": manifest["name"], "bundle_sha256": model_sha,
                  "weights_sha256": manifest["weights"]["sha256"],
                  "training_commit": manifest.get("training", {}).get("commit"),
                  "inference_config": yaml.safe_load(inference.read_text()) if inference else None},
        "code": {"commit": commit, "repo": repo, **info,
                 "environment": {k: v for k, v in envinfo.items() if k != "packages"},
                 "uv_lock_sha256": sha256_file(src / "uv.lock") if (src / "uv.lock").exists() else None},
        "run": {"started": started.isoformat(timespec="seconds"),
                "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": socket.gethostname(), "platform": platform.platform(), "device": a.device,
                "predict_seconds_this_run": round(seconds, 1), "predictions": str(pred_root),
                "runner": str(Path(__file__).resolve())},
        "missing_predictions": missing,
        "leakage": leaks,
        "overall": overall,
        "datasets": per_ds,
    }
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "packages.txt").write_text(envinfo["packages"])
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    baseline = json.loads(Path(a.baseline).read_text()) if a.baseline else None
    report.write(out, results, rows, polys, bench, baseline)

    o = overall
    log(f"{'PARTIAL ' if partial else ''}F1 {o['f1']:.4f}  P {o['precision']:.4f}  R {o['recall']:.4f}  "
        f"mIoU {o['mean_iou']:.3f}  (GT {o['gt']}, pred {o['pred']}, missing {missing})")
    log(f"results: {out}")


if __name__ == "__main__":
    main()
