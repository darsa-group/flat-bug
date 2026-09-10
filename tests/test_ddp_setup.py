"""The DDP wrapper must tolerate a data-dependent unused-parameter set.

flat-bug's loss leaves some parameters gradient-less on *some* batches (a batch with no
positive anchors never reaches the box branch; a wholly bbox-only batch never reaches the
mask branch). Plain DDP raises "Expected to have finished reduction in the prior iteration"
the first time that happens - mid-epoch, so it reads as a transient rather than a structural
problem. These tests pin the two properties that keep it from coming back.
"""

import torch

from flat_bug.trainers import _ddp_allows_unused_parameters

DDP = torch.nn.parallel.DistributedDataParallel


def _capture():
    """Stand in for DDP.__init__ so the patch can be exercised without a process group."""
    seen = {}

    def fake(self, module, *args, **kwargs):
        seen.clear()
        seen.update(kwargs)

    return seen, fake


def test_find_unused_parameters_is_forced(monkeypatch):
    seen, fake = _capture()
    monkeypatch.setattr(DDP, "__init__", fake)
    with _ddp_allows_unused_parameters():
        DDP(torch.nn.Linear(2, 2), device_ids=[0])
    assert seen["find_unused_parameters"] is True


def test_static_graph_is_cleared(monkeypatch):
    """static_graph=True assumes the unused set never changes, which is exactly what breaks."""
    seen, fake = _capture()
    monkeypatch.setattr(DDP, "__init__", fake)
    with _ddp_allows_unused_parameters():
        DDP(torch.nn.Linear(2, 2), device_ids=[0], static_graph=True)
    assert "static_graph" not in seen
    assert seen["find_unused_parameters"] is True


def test_constructor_is_restored(monkeypatch):
    """A leaked patch would silently apply to every later DDP build in the process."""
    seen, fake = _capture()
    monkeypatch.setattr(DDP, "__init__", fake)
    with _ddp_allows_unused_parameters():
        pass
    assert DDP.__init__ is fake


def test_restored_even_if_setup_raises(monkeypatch):
    seen, fake = _capture()
    monkeypatch.setattr(DDP, "__init__", fake)
    try:
        with _ddp_allows_unused_parameters():
            raise RuntimeError("setup blew up")
    except RuntimeError:
        pass
    assert DDP.__init__ is fake


def test_setup_train_wraps_only_under_ddp(monkeypatch):
    """The override must delegate upstream, and patch DDP only when world_size > 1."""
    from ultralytics.models.yolo.segment import SegmentationTrainer

    from flat_bug.trainers import FlatBugSegmentationTrainer

    calls = []
    monkeypatch.setattr(
        SegmentationTrainer, "_setup_train", lambda self, *a, **k: calls.append(DDP.__init__)
    )

    trainer = object.__new__(FlatBugSegmentationTrainer)
    seen, fake = _capture()
    monkeypatch.setattr(DDP, "__init__", fake)

    trainer.world_size = 1
    trainer._setup_train()
    assert calls[-1] is fake, "single GPU must not patch DDP"

    trainer.world_size = 2
    trainer._setup_train()
    assert calls[-1] is not fake, "DDP must be patched while upstream sets up"
    assert DDP.__init__ is fake, "and unpatched afterwards"
