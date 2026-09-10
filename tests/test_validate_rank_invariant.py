"""Every rank must make the SAME decision about whether to validate.

Upstream validate() begins with `dist.broadcast` over the EMA buffers when world_size > 1,
and _do_train calls it unguarded on all ranks. A rank that skips never enters that collective,
so the others block in it forever - and the skipping rank then hangs in the next epoch's
gradient all-reduce. That is a silent deadlock, not an error: both GPUs sit at 100% because
NCCL busy-waits.

The trap is that upstream validate() returns (None, None) on non-zero ranks, so any condition
reading back its own stored metrics is permanently true there.
"""

import pytest

from flat_bug.trainers import FlatBugSegmentationTrainer


class _FakeTrainer(FlatBugSegmentationTrainer):
    """Just enough trainer to exercise validate()'s branch, with no CUDA and no process group."""

    def __init__(self, rank0: bool, save_period: int = 5):
        self._rank0 = rank0
        self.save_period = save_period
        self.epoch = 0
        self._val_metrics = None
        self._val_fitness = None
        self._have_validated = False
        self.custom_eval = False
        self._do_custom_eval = False
        self.model = None
        self.ema = None
        self.validated_epochs = []

    def _super_validate(self):
        # Mirrors upstream: real metrics on rank 0, (None, None) everywhere else.
        return ({"fitness": 0.5}, 0.5) if self._rank0 else (None, None)


def _run(trainer, monkeypatch, epochs):
    import flat_bug.trainers as T

    monkeypatch.setattr(T.torch.cuda, "empty_cache", lambda: None)

    def fake_super_validate(self):
        self.validated_epochs.append(self.epoch)
        return self._super_validate()

    # Stand in for super().validate() without needing a real ultralytics trainer.
    monkeypatch.setattr(T.SegmentationTrainer, "validate", fake_super_validate)
    for e in range(epochs):
        trainer.epoch = e
        trainer.validate()
    return trainer.validated_epochs


def test_ranks_agree_on_which_epochs_validate(monkeypatch):
    r0 = _run(_FakeTrainer(rank0=True), monkeypatch, 12)
    r1 = _run(_FakeTrainer(rank0=False), monkeypatch, 12)
    assert r0 == r1, f"ranks disagree: rank0 validated {r0}, rank1 validated {r1}"


def test_validation_actually_follows_save_period(monkeypatch):
    got = _run(_FakeTrainer(rank0=True, save_period=25), monkeypatch, 60)
    assert got == [0, 25, 50], got


def test_non_zero_rank_does_not_validate_every_epoch(monkeypatch):
    """The regression itself: rank 1 stores None, and must not read that back as 'never ran'."""
    got = _run(_FakeTrainer(rank0=False, save_period=5), monkeypatch, 10)
    assert got == [0, 5], got
