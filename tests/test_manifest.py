import csv
import gzip
import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from flat_bug import manifest

ASSETS = Path(__file__).parent / "assets"
STEM = "ALUS_Non-miteArachnids_Unknown_2020_11_03_4545"


@pytest.fixture
def corpus(tmp_path):
    """A two-image corpus in ultralytics' layout: one train image with a label, one val image without."""
    root = tmp_path / "insects"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    shutil.copy(ASSETS / f"{STEM}.jpg", root / "images" / "train" / f"{STEM}.jpg")
    shutil.copy(ASSETS / f"{STEM}.txt", root / "labels" / "train" / f"{STEM}.txt")
    shutil.copy(ASSETS / f"{STEM}.jpg", root / "images" / "val" / "other_unlabelled.jpg")
    data = {"path": str(root), "train": "images/train", "val": "images/val"}
    (tmp_path / "data.yaml").write_text(yaml.safe_dump(data))
    return tmp_path, root


def read_inventory(path):  # noqa: D103
    with gzip.open(path, "rt") as f:
        return list(csv.DictReader(f))


def test_inventory_checksums_images_and_labels(corpus, tmp_path):
    """Images AND labels are checksummed: a relabelled image is different training data."""
    _, root = corpus
    img = root / "images" / "train" / f"{STEM}.jpg"
    dest = tmp_path / manifest.INVENTORY
    summary = manifest.data_inventory(
        {"train": [str(img)], "val": [str(root / "images" / "val" / "other_unlabelled.jpg")]}, root, dest)

    rows = read_inventory(dest)
    train = next(r for r in rows if r["split"] == "train")
    data = img.read_bytes()
    assert train["md5"] == hashlib.md5(data).hexdigest()
    assert train["sha256"] == hashlib.sha256(data).hexdigest()
    label = (root / "labels" / "train" / f"{STEM}.txt").read_bytes()
    assert train["label_sha256"] == hashlib.sha256(label).hexdigest()
    assert int(train["instances"]) == sum(1 for line in label.splitlines() if line.strip())
    assert train["dataset"] == "ALUS"
    assert train["image"] == f"images/train/{STEM}.jpg"  # relative to the corpus root

    val = next(r for r in rows if r["split"] == "val")
    assert val["label_sha256"] == "" and val["instances"] == "0"
    assert summary["val"]["without_labels"] == 1
    assert summary["train"]["datasets"]["ALUS"]["images"] == 1


def test_inventory_is_deterministic(corpus, tmp_path):
    """Same corpus, same bytes: the inventory hash identifies the data."""
    _, root = corpus
    imgs = {"train": [str(root / "images" / "train" / f"{STEM}.jpg")]}
    a = manifest.data_inventory(imgs, root, tmp_path / "a.csv.gz")
    b = manifest.data_inventory(imgs, root, tmp_path / "b.csv.gz")
    assert a["inventory_sha256"] == b["inventory_sha256"]
    assert (tmp_path / "a.csv.gz").read_bytes() == (tmp_path / "b.csv.gz").read_bytes()


def fake_trainer(tmp_path, root, data_yaml):
    """The parts of a FlatBugSegmentationTrainer the manifest reads."""
    save = tmp_path / "run"
    (save / "weights").mkdir(parents=True)
    return SimpleNamespace(
        save_dir=save,
        args=SimpleNamespace(data=str(data_yaml), epochs=1, fb_require_clean=False),
        data={"path": str(root)},
        training_image_paths=[str(root / "images" / "train" / f"{STEM}.jpg")],
        val_image_paths=[str(root / "images" / "val" / "other_unlabelled.jpg")],
        epoch=0, best_fitness=0.5, metrics={"metrics/mAP50(M)": 0.5},
    )


def test_manifest_start_and_end(corpus, tmp_path):
    """The start section is kept when the end section is added."""
    base, root = corpus
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("epochs: 1\nfb_zoom_prob: 0.2\n")
    t = fake_trainer(tmp_path, root, base / "data.yaml")

    # allow_dirty: this test must not depend on the state of the checkout running it
    path = manifest.write_start(t, config_file=str(cfg), allow_dirty=True)
    m = yaml.safe_load(path.read_text())
    assert m["schema"] == manifest.SCHEMA
    assert m["config"]["contents"] == {"epochs": 1, "fb_zoom_prob": 0.2}
    assert m["data"]["train"]["images"] == 1 and m["data"]["val"]["images"] == 1
    assert (t.save_dir / manifest.INVENTORY).exists()
    assert "torch" in m["environment"]

    (t.save_dir / "weights" / "best.pt").write_bytes(b"weights")
    manifest.write_end(t)
    m = yaml.safe_load(path.read_text())
    assert m["result"]["weights"]["best.pt"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert m["result"]["epochs_completed"] == 1
    assert m["data"]["train"]["images"] == 1  # the start section survives


def test_code_info_records_commit_and_changes(tmp_path):
    """The commit is the checkout's HEAD, and uncommitted changes are saved as a diff."""
    info = manifest.code_info(tmp_path / "code.diff")
    if info["commit"] is None:
        pytest.skip("flat-bug is not running from a git checkout")
    git = ["git", "-C", info["source"], "rev-parse", "HEAD"]
    head = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    assert info["commit"] == head
    assert info["dirty"] == (tmp_path / "code.diff").exists()


def dirty_checkout(monkeypatch, line):
    """Make code_info() report a checkout with one uncommitted or untracked file."""
    state = {"commit": "x", "dirty": True, "dirty_files": [line]}
    monkeypatch.setattr(manifest, "code_info", lambda diff_path=None: state)


@pytest.mark.parametrize("line", [" M src/flat_bug/trainers.py", "?? scripts/training/new_config.yaml"])
def test_dirty_checkout_is_refused_by_default(corpus, tmp_path, monkeypatch, line):
    """Modified AND untracked files both stop training, before anything is written."""
    base, root = corpus
    dirty_checkout(monkeypatch, line)
    t = fake_trainer(tmp_path, root, base / "data.yaml")
    with pytest.raises(manifest.DirtyCheckoutError, match="--allow-dirty"):
        manifest.write_start(t)
    assert not (t.save_dir / manifest.MANIFEST).exists()


def test_allow_dirty_overrides(corpus, tmp_path, monkeypatch):
    """With the override, training goes on and the manifest says the tree was dirty."""
    base, root = corpus
    dirty_checkout(monkeypatch, "?? src/flat_bug/new_module.py")
    path = manifest.write_start(fake_trainer(tmp_path, root, base / "data.yaml"), allow_dirty=True)
    m = yaml.safe_load(path.read_text())
    assert m["code"]["dirty"] and m["code"]["dirty_files"] == ["?? src/flat_bug/new_module.py"]


def test_untracked_files_are_in_the_diff(tmp_path):
    """A real checkout: untracked files are listed and, when small, inlined in code.diff."""
    info = manifest.code_info()
    if info["commit"] is None:
        pytest.skip("flat-bug is not running from a git checkout")
    root = Path(subprocess.run(["git", "-C", info["source"], "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True).stdout.strip())
    probe = root / "untracked_manifest_probe.txt"
    probe.write_text("probe\n")
    try:
        info = manifest.code_info(tmp_path / "code.diff")
        assert "?? untracked_manifest_probe.txt" in info["dirty_files"]
        assert "+probe" in (tmp_path / "code.diff").read_text()
        with pytest.raises(manifest.DirtyCheckoutError):
            manifest.check_clean()
    finally:
        probe.unlink()
