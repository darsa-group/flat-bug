"""Where a task's images live is a property of CVAT, not of the task's name.

Four of the project's tasks read from a bucket folder named differently from the task -
MAMBOcrops-bbox from data/MAMBOcrops, UrbanInsects-bbox from data/UrbanInsects,
AMI-traps-2024-bbox from data/AMI-traps-2024, bugbox-bulk-bbox-downscaled from
data/bugbox-bulk-cvat. Deriving the prefix from the name syncs zero images for those, which
surfaces much later as an assertion inside fb_prepare_data.
"""

from types import SimpleNamespace

from flat_bug.cli.fb_clone_data import storage_prefix_for_task


def _task(name, frame_names, raises=False):
    def get_meta():
        if raises:
            raise RuntimeError("meta unavailable")
        return SimpleNamespace(frames=[SimpleNamespace(name=n) for n in frame_names])

    return SimpleNamespace(name=name, get_meta=get_meta)


def test_prefix_comes_from_the_frames_not_the_name():
    t = _task("MAMBOcrops-bbox", ["data/MAMBOcrops/a.jpg", "data/MAMBOcrops/b.jpg"])
    assert storage_prefix_for_task(t, "data") == "data/MAMBOcrops"


def test_matching_name_is_unchanged():
    """The 39 tasks whose folder does match their name must behave exactly as before."""
    t = _task("BugNet", ["data/BugNet/a.jpg", "data/BugNet/b (1).jpg"])
    assert storage_prefix_for_task(t, "data") == "data/BugNet"


def test_falls_back_when_frames_disagree():
    t = _task("mixed", ["data/one/a.jpg", "data/two/b.jpg"])
    assert storage_prefix_for_task(t, "data") == "data/mixed"


def test_falls_back_when_meta_unavailable():
    t = _task("offline", [], raises=True)
    assert storage_prefix_for_task(t, "data") == "data/offline"


def test_falls_back_when_frames_are_bucket_root():
    """A frame with no directory part must not yield an empty prefix that syncs the bucket."""
    t = _task("rooted", ["a.jpg", "b.jpg"])
    assert storage_prefix_for_task(t, "data") == "data/rooted"


def test_name_is_sanitised_in_the_fallback():
    t = _task("odd/name", [], raises=True)
    assert "/" not in storage_prefix_for_task(t, "data").removeprefix("data/")
