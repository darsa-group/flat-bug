"""Training provenance: what code, environment and data produced a set of weights.

Training writes two files into its run folder (``runs/segment/<name>/``):

``manifest.train.yaml``
    The flat-bug commit (and whether the working tree had uncommitted changes, saved alongside
    as ``code.diff``), the command line, the config file as written, the resolved training
    arguments, the software and hardware, the data summary, and - once training ends - the
    checksums of the weights it produced.

``data_inventory.csv.gz``
    One row per training and validation image: split, dataset, path relative to the corpus,
    size, md5 and sha256 of the image, sha256 of its label file and its instance count. md5 is
    kept on purpose: it is what flat-bug's train/validation split is computed from and what S3
    reports as an ETag, so a benchmark can check that none of its images were trained on.

Ultralytics' checkpoints carry a ``git`` field, but it describes the ultralytics installation,
not flat-bug; it is not a substitute for this.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import platform
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from flat_bug import logger

SCHEMA = "flatbug-training/1"
MANIFEST = "manifest.train.yaml"
INVENTORY = "data_inventory.csv.gz"
MAX_UNTRACKED_DIFF_BYTES = 1_000_000
INVENTORY_FIELDS = ["split", "dataset", "image", "bytes", "md5", "sha256", "label_sha256", "instances"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.rstrip() if r.returncode == 0 else None


def code_info(diff_path: Path | None = None) -> dict:
    """The flat-bug code that is running: commit, branch, and uncommitted changes.

    If flat-bug is not running from a git checkout (e.g. installed from a wheel), only the package
    version is known, and ``commit`` is None.

    Args:
        diff_path: Where to save the uncommitted changes, if there are any.
    """
    import tomllib  # noqa: PLC0415
    from importlib.metadata import PackageNotFoundError  # noqa: PLC0415
    from importlib.metadata import version as pkg_version

    here = Path(__file__).resolve().parent
    root = _git(here, "rev-parse", "--show-toplevel")
    if root is None:
        try:
            version = pkg_version("flat-bug")
        except PackageNotFoundError:
            version = None
        return {"package_version": version, "source": str(here), "commit": None}

    root = Path(root)
    # In a checkout, the installed metadata can be stale (an editable install keeps the version it
    # was installed with); the checkout's own pyproject.toml is what is running.
    with open(root / "pyproject.toml", "rb") as f:
        version = tomllib.load(f).get("project", {}).get("version")
    tracked = (_git(root, "status", "--porcelain", "--untracked-files=no") or "").splitlines()
    # Untracked files count too: a new module, or a config that exists only on this machine,
    # changes what trains as much as an edit does. Ignored files (.gitignore) do not count.
    new = (_git(root, "ls-files", "--others", "--exclude-standard") or "").splitlines()
    dirty = tracked + [f"?? {p}" for p in new]
    info: dict[str, Any] = {
        "package_version": version,
        "source": str(here),
        "repository": _git(root, "remote", "get-url", "origin"),
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "subject": _git(root, "log", "-1", "--format=%s"),
        "commit_date": _git(root, "log", "-1", "--format=%aI"),
        "dirty": bool(dirty),
        "dirty_files": dirty or None,
    }
    if dirty and diff_path is not None:
        parts, skipped = [_git(root, "diff", "HEAD") or ""], []
        for p in new:
            if (root / p).stat().st_size > MAX_UNTRACKED_DIFF_BYTES:
                skipped.append(p)  # listed in dirty_files, but too large to inline
                continue
            # `git diff --no-index` exits 1 when files differ, so not via _git
            r = subprocess.run(["git", "-C", str(root), "diff", "--no-index", "--", "/dev/null", p],
                               capture_output=True, text=True)
            parts.append(r.stdout.rstrip("\n"))
        diff = "\n".join(x for x in parts if x) + "\n"
        diff_path.write_text(diff)
        info["diff_file"] = diff_path.name
        info["diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
        if skipped:
            info["untracked_not_in_diff"] = skipped
    return info


class DirtyCheckoutError(RuntimeError):
    """Training was asked to run from a checkout with uncommitted changes or untracked files."""


def check_clean(allow_dirty: bool = False) -> dict:
    """Refuse to go on from a checkout with uncommitted changes or untracked files.

    Weights are only reproducible from a commit. With ``allow_dirty`` the check only warns, and the
    changes are saved next to the training manifest as ``code.diff``.

    Returns:
        The code_info() of the checkout (without saving a diff).
    """
    info = code_info()
    if info.get("dirty"):
        files = "\n  ".join(info["dirty_files"])
        msg = (f"The flat-bug checkout has uncommitted changes or untracked files:\n  {files}\n"
               "Commit them (or add them to .gitignore) so the weights can be traced to a commit, or "
               "override with `fb_train --allow-dirty` / `fb_allow_dirty: true` in the config; the "
               "changes are then saved next to the training manifest as code.diff.")
        if not allow_dirty:
            raise DirtyCheckoutError(msg)
        logger.warning(msg)
    return info


def environment_info() -> dict:
    """Software and hardware the training ran with."""
    import torch  # noqa: PLC0415
    import ultralytics  # noqa: PLC0415

    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpus = [torch.cuda.get_device_name(i) for i in range(n)]
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "gpus": gpus,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _hash_file(path: str) -> tuple[int, str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**22), b""):
            md5.update(b)
            sha.update(b)
            n += len(b)
    return n, md5.hexdigest(), sha.hexdigest()


def _label_path(image: str) -> str:
    """Ultralytics' convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return sb.join(image.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def _row(split: str, image: str, root: Path) -> dict:
    n, md5, sha = _hash_file(image)
    label = _label_path(image)
    if os.path.exists(label):
        with open(label, "rb") as f:
            text = f.read()
        label_sha = hashlib.sha256(text).hexdigest()
        instances = sum(1 for line in text.splitlines() if line.strip())
    else:
        label_sha, instances = None, 0
    try:
        rel = str(Path(image).resolve().relative_to(root))
    except ValueError:
        rel = image
    base = os.path.basename(image)
    return {"split": split, "dataset": base.split("_", 1)[0] if "_" in base else "", "image": rel, "bytes": n,
            "md5": md5, "sha256": sha, "label_sha256": label_sha, "instances": instances}


def data_inventory(splits: dict[str, list[str]], root: Path, dest: Path, workers: int = 16) -> dict:
    """Checksum every image and label of every split, write the inventory, return its summary.

    Args:
        splits: Image paths per split, e.g. ``{"train": [...], "val": [...]}``.
        root: Corpus root; paths in the inventory are relative to it.
        dest: The ``data_inventory.csv.gz`` to write.
        workers: Hashing threads (hashing is I/O-bound).
    """
    root = root.resolve()
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for split, images in splits.items():
            rows += ex.map(lambda p, s=split: _row(s, p, root), sorted(images))
    rows.sort(key=lambda r: (r["split"], r["image"]))

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue().encode()
    with open(dest, "wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(text)

    summary: dict[str, Any] = {"inventory_file": dest.name, "inventory_sha256": hashlib.sha256(text).hexdigest()}
    for split in splits:
        r = [x for x in rows if x["split"] == split]
        per: dict[str, dict] = {}
        for x in r:
            d = per.setdefault(x["dataset"], {"images": 0, "instances": 0})
            d["images"] += 1
            d["instances"] += x["instances"]
        summary[split] = {
            "images": len(r),
            "instances": sum(x["instances"] for x in r),
            "without_labels": sum(1 for x in r if x["label_sha256"] is None),
            "bytes": sum(x["bytes"] for x in r),
            "datasets": dict(sorted(per.items(), key=lambda kv: kv[0].lower())),
        }
    return summary


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, str):
        return str(x)  # str subclasses (e.g. torch's TorchVersion) are not safe for yaml.safe_dump
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    return str(x)


def _dump(m: dict) -> str:
    return yaml.safe_dump(_plain(m), sort_keys=False, allow_unicode=True)


def _sha256(path: Path) -> str:
    return _hash_file(str(path))[2]


def write_start(trainer, config_file: str | None = None, allow_dirty: bool = False) -> Path:
    """Write the manifest when training starts (or append a resume to an existing one)."""
    save_dir = Path(trainer.save_dir)
    path = save_dir / MANIFEST
    if path.exists():  # a resumed run: keep the original, record the resume
        m = yaml.safe_load(path.read_text())
        m.setdefault("resumes", []).append({
            "at": _now(), "epoch": int(getattr(trainer, "start_epoch", 0)), "command": sys.argv,
            "code": code_info(save_dir / f"code.resume{len(m.get('resumes', [])) + 1}.diff"),
            "environment": environment_info(),
        })
        path.write_text(_dump(m))
        return path

    check_clean(allow_dirty)  # fb_train checks first; this covers trainers started any other way
    code = code_info(save_dir / "code.diff")
    args = _plain(vars(trainer.args))
    data_yaml = args.get("data")
    root = Path(trainer.data.get("path") or Path(data_yaml).parent)
    m = {
        "schema": SCHEMA,
        "run": {"name": save_dir.name, "save_dir": str(save_dir), "started": _now(), "command": sys.argv},
        "code": code,
        "environment": environment_info(),
        "config": {
            "file": os.path.abspath(config_file) if config_file else None,
            "contents": yaml.safe_load(open(config_file)) if config_file else None,
        },
        "train_args": args,
        "data": {
            "data_yaml": data_yaml,
            "data_yaml_contents": yaml.safe_load(open(data_yaml)) if data_yaml and os.path.exists(data_yaml) else None,
            "root": str(root),
            **data_inventory({"train": trainer.training_image_paths, "val": trainer.val_image_paths},
                             root, save_dir / INVENTORY),
        },
    }
    path.write_text(_dump(m))
    return path


def write_end(trainer) -> Path:
    """Add how training ended, and the checksums of the weights it left behind."""
    save_dir = Path(trainer.save_dir)
    path = save_dir / MANIFEST
    m = yaml.safe_load(path.read_text()) if path.exists() else {"schema": SCHEMA}
    weights = {}
    for name in ("best.pt", "last.pt"):
        p = save_dir / "weights" / name
        if p.exists():
            weights[name] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    m["result"] = {
        "finished": _now(),
        "epochs_completed": int(getattr(trainer, "epoch", -1)) + 1,
        "best_fitness": _plain(getattr(trainer, "best_fitness", None)),
        "final_metrics": _plain(getattr(trainer, "metrics", None)),
        "weights": weights,
    }
    path.write_text(_dump(m))
    return path
