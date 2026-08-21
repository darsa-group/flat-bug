"""Tests for the incremental CVAT/S3 clone used to assemble the training data.

`fb_clone_data` is what feeds `fb_prepare_data` and, in turn, training. The bits
that make it cheap to re-run - only completed tasks, and only files whose ETag
changed upstream - are exercised here against a stub S3 client, so no
credentials or network are needed.
"""

import os
import tempfile
from pathlib import Path

import pytest

from flat_bug.cli.fb_clone_data import md5_file, sync_s3_prefix_to_local, task_is_completed


class _StubS3:
    """Minimal stand-in for a boto3 S3 client over an in-memory bucket."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = dict(objects)
        self.downloads: list[str] = []

    def _etag(self, key: str) -> str:
        import hashlib

        return hashlib.md5(self.objects[key]).hexdigest()

    def list_objects_v2(self, Bucket, Prefix, **kwargs):  # noqa: N803
        contents = [
            {"Key": k, "Size": len(v), "ETag": f'"{self._etag(k)}"', "LastModified": 0}
            for k, v in sorted(self.objects.items())
            if k.startswith(Prefix)
        ]
        return {"Contents": contents, "KeyCount": len(contents), "IsTruncated": False}

    def download_file(self, bucket, key, dest):  # noqa: D102
        self.downloads.append(key)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.objects[key])


class _StubJob:
    def __init__(self, state: str):  # noqa: D107
        self.state = state


class _StubTask:
    """A CVAT task exposing just what `task_is_completed` looks at."""

    def __init__(self, status=None, jobs=()):  # noqa: D107
        self.status = status
        self._jobs = list(jobs)

    def get_jobs(self):  # noqa: D102
        return self._jobs


def _sync(s3, tmp, prefix="proj/task_a/"):
    return sync_s3_prefix_to_local(s3, "bucket", prefix, Path(tmp))


def test_sync_downloads_then_skips_unchanged_files():
    """Re-running the clone must not re-download anything that did not change upstream."""
    s3 = _StubS3({"proj/task_a/a.jpg": b"image-a", "proj/task_a/instances_default.json": b"{}"})
    with tempfile.TemporaryDirectory() as tmp:
        assert _sync(s3, tmp) is True
        assert sorted(s3.downloads) == ["proj/task_a/a.jpg", "proj/task_a/instances_default.json"]
        assert (Path(tmp) / "a.jpg").read_bytes() == b"image-a"
        # The ETag sidecar is what makes the second pass cheap.
        assert (Path(tmp) / "a.jpg.etag").read_text() == md5_file(Path(tmp) / "a.jpg")

        s3.downloads.clear()
        assert _sync(s3, tmp) is True
        assert s3.downloads == []


def test_sync_redownloads_changed_files_only():  # noqa: D103
    s3 = _StubS3({"proj/task_a/a.jpg": b"image-a", "proj/task_a/b.jpg": b"image-b"})
    with tempfile.TemporaryDirectory() as tmp:
        _sync(s3, tmp)
        s3.downloads.clear()

        s3.objects["proj/task_a/b.jpg"] = b"image-b-reannotated"
        _sync(s3, tmp)
        assert s3.downloads == ["proj/task_a/b.jpg"]
        assert (Path(tmp) / "b.jpg").read_bytes() == b"image-b-reannotated"


def test_sync_recovers_when_the_sidecar_is_missing():
    """A file already on disk from an older clone must not be re-fetched needlessly."""
    s3 = _StubS3({"proj/task_a/a.jpg": b"image-a"})
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "a.jpg").write_bytes(b"image-a")
        _sync(s3, tmp)
        assert s3.downloads == []
        assert (Path(tmp) / "a.jpg.etag").exists()


def test_sync_deletes_files_dropped_upstream():  # noqa: D103
    s3 = _StubS3({"proj/task_a/a.jpg": b"image-a", "proj/task_a/gone.jpg": b"image-gone"})
    with tempfile.TemporaryDirectory() as tmp:
        _sync(s3, tmp)
        assert (Path(tmp) / "gone.jpg").exists()

        del s3.objects["proj/task_a/gone.jpg"]
        _sync(s3, tmp)
        assert not (Path(tmp) / "gone.jpg").exists()
        assert not (Path(tmp) / "gone.jpg.etag").exists()
        assert (Path(tmp) / "a.jpg").exists()


def test_sync_reports_an_empty_prefix():  # noqa: D103
    s3 = _StubS3({"proj/other_task/a.jpg": b"image-a"})
    with tempfile.TemporaryDirectory() as tmp:
        assert _sync(s3, tmp) is False
        assert os.listdir(tmp) == []


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (_StubTask(status="completed"), True),
        (_StubTask(status="annotation", jobs=[_StubJob("completed"), _StubJob("completed")]), True),
        (_StubTask(status="annotation", jobs=[_StubJob("completed"), _StubJob("in progress")]), False),
        (_StubTask(status="annotation", jobs=[]), False),
    ],
)
def test_only_completed_tasks_are_pulled(task, expected):
    """Half-annotated tasks must stay out of the training set."""
    assert task_is_completed(task) is expected
