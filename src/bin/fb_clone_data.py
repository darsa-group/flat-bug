from zipfile import ZipFile
import hashlib
import boto3
from cvat_sdk import make_client
from botocore.config import Config
import yaml
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os, json, shutil
from botocore.exceptions import ClientError

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    _HAVE_TQDM = False


secrets_structure = """
cvat:
  host: https://app.cvat.ai
  username: 
  password: 
  project_id: 
s3:
  endpoint_url: 
  region: 
  bucket: 
  prefix: 
  access_key: 
  secret_key: 
"""




# TARGET_DIR = Path("/home/quentin/Desktop/flat-bug/flat-bug-data/pre-pro")
FORMAT_NAME = "COCO 1.0"
# ------------------ Secrets ------------------

# ------------------ Load secrets from YAML ------------------
def load_secrets_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------ Helpers ------------------
def safe_segment(name: str) -> str:
    """
    Make a filesystem-safe folder name (keep common chars; replace others with underscore).
    """
    allowed = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    cleaned = "".join(c if c in allowed else "_" for c in name).strip()
    # avoid empty folder names
    return cleaned or "unnamed_task"

def md5_file(path: Path, chunk=1024 * 1024) -> str:
    """Compute MD5 hex digest of a file (for ETag comparison if single-part)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def task_is_completed(task) -> bool:
    """
    Consider a task completed if either:
      - task.status == 'completed', OR
      - all of its jobs are in state == 'completed'
    """
    try:
        if getattr(task, "status", None) == "completed":
            return True
    except Exception:
        pass

    try:
        jobs = list(task.get_jobs())
        return len(jobs) > 0 and all(getattr(j, "state", None) == "completed" for j in jobs)
    except Exception:
        return False

def build_s3_client(s3_access_key, s3_secret_key, s3_region, s3_endpoint):
    session = boto3.session.Session(
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
        region_name=s3_region,
    )
    return session.client(
        "s3",
        endpoint_url=s3_endpoint,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "virtual"},  # or "path" if your bucket has dots
        ),
    )

def list_s3_objects_with_prefix(s3, bucket: str, prefix: str):
    """
    Yield dicts with 'Key', 'Size', 'ETag' (no quotes), and 'LastModified'.
    """
    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation:
            kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue  # skip directory placeholders
            etag = item.get("ETag", "").strip('"')
            yield {
                "Key": key,
                "Size": item["Size"],
                "ETag": etag,
                "LastModified": item["LastModified"],
            }
        if resp.get("IsTruncated"):
            continuation = resp["NextContinuationToken"]
        else:
            break

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
#
# ------------------ COCO Export ------------------
def export_coco_annotations_for_task(task, output_json_path: Path, s3_prefix):
    """
    Export task dataset (COCO 1.0, annotations only) to a temp zip,
    then extract the COCO annotations json to output_json_path.
    """
    tmp_zip = output_json_path.parent / f"task-{task.id}-export.tmp.zip"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    task.export_dataset(
        FORMAT_NAME,
        filename=str(tmp_zip),
        include_images=False,

    )

    # Find the annotations JSON inside the zip (usually 'annotations/instances_default.json')
    with ZipFile(tmp_zip, "r") as zf:
        # Prefer the standard path; fallback to the first *instances*.json we find.
        preferred = "annotations/instances_default.json"
        info = None
        try:
            info = zf.getinfo(preferred)
        except KeyError:
            for zi in zf.infolist():
                name = zi.filename.replace("\\", "/")
                if name.lower().startswith("annotations/") and name.lower().endswith(".json") and "instance" in name.lower():
                    info = zi
                    break
        if info is None:
            # last resort: any annotations json
            for zi in zf.infolist():
                name = zi.filename.replace("\\", "/")
                if name.lower().startswith("annotations/") and name.lower().endswith(".json"):
                    info = zi
                    break

        if info is None:
            zf.close()
            tmp_zip.unlink(missing_ok=True)
            raise RuntimeError("Could not locate COCO annotations JSON inside the export zip.")

        with zf.open(info, "r") as src:
            coco = json.load(src)


        for im in coco.get("images", []):
            orig = im.get("file_name", "")
            # Make sure we don't accidentally duplicate prefixes
            im["file_name"] = os.path.relpath(orig, os.path.join(s3_prefix, safe_segment(task.name)))

        # Write back modified JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)

        # Cleanup temp zip
        tmp_zip.unlink(missing_ok=True)


def _iter_s3_keys(s3, bucket, prefix):
    """Yield all keys under a prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def sync_s3_prefix_to_local(
    s3,
    bucket: str,
    prefix: str,
    local_root: Path,
    delete_extraneous: bool = True,
) -> bool:
    """
    Incremental sync of s3://bucket/prefix -> local_root.

    - Skips files whose cached ETag matches upstream ETag.
    - If no sidecar exists: when upstream ETag looks like a single-part MD5
      (no '-'), compares to local MD5 and creates sidecar if they match.
    - Otherwise downloads and then stores upstream ETag in <file>.etag.
    - Optionally deletes local files that are not present upstream.
    """
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    # 1) Index upstream
    upstream = {}
    for obj in list_s3_objects_with_prefix(s3, bucket, prefix):
        rel = obj["Key"][len(prefix):].lstrip("/")
        if not rel:
            continue
        upstream[rel] = obj

    if not upstream:
        return False

    # 2) Decide work & download as needed
    keys = list(upstream.keys())
    prog = tqdm(total=len(keys), desc=f"s3 sync {prefix}", unit="file") if _HAVE_TQDM else None
    try:
        for rel in keys:
            meta = upstream[rel]
            dest = local_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            etag_sidecar = dest.with_suffix(dest.suffix + ".etag")
            remote_etag = meta.get("ETag", "")
            remote_size = meta.get("Size", None)

            need_dl = False
            if not dest.exists():
                need_dl = True
            else:
                # If we have a cached ETag, trust it.
                if etag_sidecar.exists():
                    on_disk_etag = etag_sidecar.read_text().strip()
                    # quick size check helps catch stale sidecars due to manual edits
                    if on_disk_etag == remote_etag and (remote_size is None or dest.stat().st_size == remote_size):
                        need_dl = False
                    else:
                        need_dl = True
                else:
                    # No sidecar: try to avoid re-download for single-part objects
                    if remote_size is not None and dest.stat().st_size != remote_size:
                        need_dl = True
                    else:
                        # Single-part ETag is MD5 (no '-')
                        if remote_etag and "-" not in remote_etag:
                            try:
                                need_dl = (md5_file(dest) != remote_etag)
                            except Exception:
                                need_dl = True
                        else:
                            # Multipart: cannot reproduce ETag; refresh once
                            need_dl = True

            if need_dl:
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                s3.download_file(bucket, meta["Key"], str(tmp))
                tmp.replace(dest)
                # write/refresh etag sidecar
                if remote_etag:
                    etag_sidecar.write_text(remote_etag)
                elif etag_sidecar.exists():
                    etag_sidecar.unlink(missing_ok=True)
            else:
                # ensure we have a sidecar when we can
                if remote_etag and not etag_sidecar.exists():
                    etag_sidecar.write_text(remote_etag)

            if prog:
                prog.update(1)
    finally:
        if prog:
            prog.close()

    # 3) Delete locals that aren't upstream (and their .etag)
    if delete_extraneous:
        for local_path in local_root.rglob("*"):
            if local_path.is_dir():
                continue
            if local_path.suffix == ".etag":
                # keep sidecars only for files that still exist
                data_path = local_path.with_suffix("")
                if not data_path.exists():
                    local_path.unlink(missing_ok=True)
                continue
            rel = str(local_path.relative_to(local_root))
            if rel not in upstream:
                local_path.unlink(missing_ok=True)
                side = local_path.with_suffix(local_path.suffix + ".etag")
                side.unlink(missing_ok=True)

    return True
def _process_one_task(task_id: int,
                      task_name: str,
                      cfg_cvat: dict,
                      s3,
                      s3_bucket: str,
                      s3_prefix_root: str,
                      target_dir: Path):
    """Runs in a thread. Returns (task_id, ok, msg)."""

    CVAT_HOST = cfg_cvat.get("host", "https://app.cvat.ai")
    USERNAME = cfg_cvat["username"]
    PASSWORD = cfg_cvat["password"]
    ORG_SLUG = cfg_cvat.get("org_slug")

    # Build local folder for this task
    task_dir_name = safe_segment(task_name)
    task_dir = target_dir / task_dir_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Build S3 prefix: <root>/<task_name>/
    prefix = f"{s3_prefix_root}/{task_name}/" if s3_prefix_root else f"{task_name}/"

    try:
        # Check S3 existence (lightweight)
        has_upstream = False
        try:
            resp = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix, MaxKeys=1)
            has_upstream = resp.get("KeyCount", 0) > 0
        except ClientError as e:
            # Non-fatal; continue to annotation export
            print(f"[Task {task_id}] S3 error: {e}")

        if has_upstream:
            print(f"[Task {task_id}] Syncing s3://{s3_bucket}/{prefix} -> {task_dir}")
            synced = sync_s3_prefix_to_local(s3, s3_bucket, prefix, task_dir)
            if not synced:
                print(f"[Task {task_id}] Nothing to sync.")

        # Now do CVAT export (thread-local client for safety)
        with make_client(host=CVAT_HOST, credentials=(USERNAME, PASSWORD)) as client:
            if ORG_SLUG:
                client.organization_slug = ORG_SLUG
            t = client.tasks.retrieve(task_id)   # get fresh task handle
            t.fetch()
            if not task_is_completed(t):
                return task_id, False, "Skipping: not completed"

            coco_json_path = task_dir / "instances_default.json"
            print(f"[Task {task_id}] Exporting COCO -> {coco_json_path}")
            export_coco_annotations_for_task(t, coco_json_path, s3_prefix_root)

        return task_id, True, "ok"
    except Exception as e:
        return task_id, False, f"error: {e}"

# ------------------ Main ------------------
def main():
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-s", "--secrets-file", dest="secrets_file",
                            help="A YAML files containing credentials for s3 and cvat. It has the following structure"
                                 f"{secrets_structure}"
                            )

    args_parse.add_argument("-o", "--output-dir", dest="output_dir",
                            help="The output directory where all subdatasets are stored. Each subdirectory is a coco dataset, with a JSON file and a list of images")

    args_parse.add_argument("-f", "--force", dest="delete_target_before",
                            help="Delete output directory before, this avoids duplicating data etc",
                            action="store_true")
    args = args_parse.parse_args()
    option_dict = vars(args)

    TARGET_DIR = Path(option_dict["output_dir"])
    SECRETS = load_secrets_yaml(option_dict["secrets_file"])
    assert SECRETS, f"Failed to load secrets {option_dict['secrets_file']}"
    CVAT_CFG = SECRETS["cvat"]
    S3_CFG = SECRETS["s3"]

    CVAT_HOST = CVAT_CFG.get("host", "https://app.cvat.ai")
    USERNAME = CVAT_CFG["username"]
    PASSWORD = CVAT_CFG["password"]
    PROJECT_ID = int(CVAT_CFG["project_id"])
    ORG_SLUG = CVAT_CFG.get("org_slug")

    AWS_ACCESS_KEY_ID = S3_CFG["access_key"]
    AWS_SECRET_ACCESS_KEY = S3_CFG["secret_key"]
    AWS_REGION = S3_CFG["region"]
    S3_ENDPOINT_URL = S3_CFG["endpoint_url"]
    S3_BUCKET = S3_CFG["bucket"]
    S3_PREFIX = S3_CFG.get("prefix", "").rstrip("/")

    # if option_dict["delete_target_before"] and TARGET_DIR.exists():
    #     shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    s3 = build_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_ENDPOINT_URL)

    # List tasks once (single client), then do the heavy work in parallel per task
    with make_client(host=CVAT_HOST, credentials=(USERNAME, PASSWORD)) as client:
        if ORG_SLUG:
            client.organization_slug = ORG_SLUG

        tasks = [t for t in client.tasks.list() if t.project_id == PROJECT_ID]
        print(f"Found {len(tasks)} tasks in project {PROJECT_ID}")

        # Prepare summaries to avoid sharing CVAT objects across threads
        items = []
        for t in tasks:
            t.fetch()
            items.append((t.id, t.name, task_is_completed(t)))

    # Filter to completed tasks (we'll still re-check in threads for safety)
    items = [(tid, tname) for (tid, tname, ok) in items if ok]
    if not items:
        print("No completed tasks to process.")
        print("✅ Done.")
        return

    max_workers = min(8, max(2, os.cpu_count() or 4))
    print(f"Running in parallel with {max_workers} workers...")

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for tid, tname in items:
            futures.append(
                ex.submit(
                    _process_one_task,
                    tid,
                    tname,
                    CVAT_CFG,
                    s3,
                    S3_BUCKET,
                    S3_PREFIX,
                    TARGET_DIR,
                )
            )

        for f in as_completed(futures):
            tid, ok, msg = f.result()
            status = "✅" if ok else "❌"
            print(f"{status} Task {tid}: {msg}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
