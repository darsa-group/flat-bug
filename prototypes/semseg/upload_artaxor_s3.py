#!/usr/bin/env python3
"""PROTOTYPE - upload the ArTaxOr agree/review image sets to the flat-bug S3 bucket.

CVAT tasks in project `flatbug-cloud` are backed by cloud storage rather than copies, so the
bucket is the single source of truth: an object deleted or renamed later breaks the task.
The upload therefore runs first and is verified before any task is created.

Images are routed by the rule that any instance sent to review sends its whole image to
review, so `ArTaxOr-more` contains only images whose every instance was corroborated by
flat-bug. Two basenames collide across taxonomic orders (the same image filed twice, verified
byte-identical), so upload is deduplicated by basename to match the flat one-prefix-per-task
layout the existing tasks use.

Existing keys are listed first and skipped, so an interrupted run resumes.

Usage:
    upload_artaxor_s3.py --dry-run
    upload_artaxor_s3.py --go [--workers 8]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
import yaml
from botocore.config import Config

BAGS = "/home/quentin/Desktop/ArTaxOr_masks"
SRC = "/home/quentin/Desktop/ArTaxOr_full/ArTaxOr"
SECRETS = "/home/quentin/repos/flat-bug-git/scripts/training/.secrets.yaml"
# The key in SECRETS is read-only - correct for fb_clone_data, which only downloads, and
# confirmed by a 403 on PUT. Writing needs a separate credential, kept in the s3cmd config
# rather than copied into the secrets file so the download path stays read-only.
S3CFG = "/home/quentin/.s3cfg-linode-flatbug-cloud"
PREFIXES = {"more": "data/ArTaxOr-more/", "review": "data/ArTaxOr-review/"}


def route_images() -> dict[str, list[str]]:
    """Split images between the two tasks by the any-review-wins rule.

    Returns:
        {"more": [paths], "review": [paths]} with basenames deduplicated.
    """
    has_review: dict[str, bool] = {}
    for side in ("agree", "review"):
        for f in glob.glob(os.path.join(BAGS, side, "*", "instances.json")):
            j = json.load(open(f))
            names = {im["id"]: im["file_name"] for im in j["images"]}
            for a in j["annotations"]:
                n = names[a["image_id"]]
                has_review[n] = has_review.get(n, False) or (side == "review")
    on_disk = {os.path.basename(p): p for p in glob.glob(os.path.join(SRC, "*", "*.jpg"))}
    out: dict[str, list[str]] = {"more": [], "review": []}
    missing = 0
    for name, rev in has_review.items():
        p = on_disk.get(name)
        if p is None:
            missing += 1
            continue
        out["review" if rev else "more"].append(p)
    if missing:
        print(f"warning: {missing} annotated images not found on disk", file=sys.stderr)
    return out


def _read_s3cfg(path: str) -> dict:
    """Parse an s3cmd config, which is headerless ``key = value`` rather than INI."""
    out = {}
    for line in open(path):
        if "=" in line and not line.strip().startswith(("[", "#")):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def client():
    """Build an S3 client: bucket and endpoint from the secrets file, write key from s3cmd."""
    d = yaml.safe_load(open(SECRETS))["s3"]
    w = _read_s3cfg(S3CFG)
    host = w.get("host_base", "")
    endpoint = host if host.startswith("http") else "https://" + host
    s = boto3.session.Session(aws_access_key_id=w["access_key"],
                              aws_secret_access_key=w["secret_key"],
                              region_name=d.get("region"))
    return s.client("s3", endpoint_url=endpoint,
                    config=Config(s3={"addressing_style": "virtual"},
                                  connect_timeout=30, read_timeout=120,
                                  retries={"max_attempts": 5, "mode": "adaptive"},
                                  max_pool_connections=32)), d["bucket"]


def existing_keys(s3, bucket: str, prefix: str) -> set[str]:
    """List the keys already present under a prefix, so the upload can resume."""
    out, tok = set(), None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if tok:
            kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out |= {o["Key"] for o in r.get("Contents", [])}
        if not r.get("IsTruncated"):
            return out
        tok = r["NextContinuationToken"]


def main() -> None:
    """Route, then optionally upload."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="actually upload; without it, dry run only")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    routed = route_images()
    s3, bucket = client()
    print(f"bucket {bucket}\n")
    plan = {}
    for side, paths in routed.items():
        pre = PREFIXES[side]
        have = existing_keys(s3, bucket, pre)
        todo = [p for p in paths if pre + os.path.basename(p) not in have]
        size = sum(os.path.getsize(p) for p in todo)
        plan[side] = todo
        print(f"{pre:26s} {len(paths):6d} images   already there {len(have):6d}   "
              f"to upload {len(todo):6d}   {size / 1e9:5.2f} GB")
    if not a.go:
        print("\ndry run - nothing uploaded. re-run with --go")
        return

    lock = threading.Lock()
    done = {"n": 0}
    total = sum(len(v) for v in plan.values())
    t0 = time.time()

    def put(args):
        side, p = args
        key = PREFIXES[side] + os.path.basename(p)
        s3.upload_file(p, bucket, key, ExtraArgs={"ContentType": "image/jpeg"})
        with lock:
            done["n"] += 1
            if done["n"] % 250 == 0:
                el = time.time() - t0
                print(f"  {done['n']}/{total}  {el / 60:.1f} min  "
                      f"~{el / done['n'] * (total - done['n']) / 60:.0f} min left", flush=True)

    jobs = [(side, p) for side, paths in plan.items() for p in paths]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(put, jobs))
    print(f"\nuploaded {done['n']} objects in {(time.time() - t0) / 60:.1f} min")
    for side in plan:
        pre = PREFIXES[side]
        print(f"  {pre:26s} now {len(existing_keys(s3, bucket, pre)):6d} objects")


if __name__ == "__main__":
    main()
