"""The resume path must work before it is needed, not after a 70-hour run dies.

Two independent faults, each fatal on its own:

  * `custom_fb_args["fb_custom_eval"]` was a hard index. On resume, fb_train rebuilds
    `overrides` from the checkpoint rather than from DEFAULT_CONF, so a config that never
    mentions fb_custom_eval (none of the 300-epoch ones do) raises KeyError before training
    starts.
  * `torch.load(resume_model)` omitted weights_only=False. A training checkpoint carries the
    pickled model and optimiser state, and torch >= 2.6 defaults weights_only to True, so the
    load raises UnpicklingError on every flat-bug checkpoint.

`last.pt` is written every epoch regardless of save_period, so the checkpoint was always
there; only the tooling to continue from it was missing.
"""

import inspect

import torch

import flat_bug.trainers as T


def test_resume_load_is_not_weights_only():
    src = inspect.getsource(T.apply_overrides_to_checkpoint)
    assert "weights_only=False" in src, (
        "torch >= 2.6 defaults weights_only=True, which cannot load a training checkpoint"
    )


def test_resume_load_maps_to_cpu():
    """The checkpoint records the device its tensors lived on; loading it elsewhere raises."""
    src = inspect.getsource(T.apply_overrides_to_checkpoint)
    assert 'map_location="cpu"' in src, (
        "without map_location a checkpoint saved on cuda:0 cannot be read on a CPU-only host"
    )


def test_custom_eval_keys_are_optional():
    """A config that omits the fb_custom_eval keys must not raise, which is the resume case."""
    src = inspect.getsource(T.FlatBugSegmentationTrainer.__init__)
    assert 'custom_fb_args["fb_custom_eval"]' not in src, "hard index breaks resume"
    assert 'custom_fb_args["fb_custom_eval_num_images"]' not in src, "hard index breaks resume"
    assert 'custom_fb_args.get("fb_custom_eval"' in src
    assert 'custom_fb_args.get("fb_custom_eval_num_images"' in src


def test_defaults_match_fb_train(tmp_path):
    """The .get defaults must agree with DEFAULT_CONF, or resume silently changes behaviour."""
    from flat_bug.cli.fb_train import main  # noqa: F401
    import flat_bug.cli.fb_train as ft

    src = inspect.getsource(ft)
    assert '"fb_custom_eval": False' in src
    assert '"fb_custom_eval_num_images": -1' in src
    tsrc = inspect.getsource(T.FlatBugSegmentationTrainer.__init__)
    assert 'get("fb_custom_eval", False)' in tsrc
    assert 'get("fb_custom_eval_num_images", -1)' in tsrc


def test_real_checkpoint_round_trips(tmp_path):
    """A pickled trainer-style checkpoint must load back under the fixed call."""
    ckpt = {"epoch": 7, "model": torch.nn.Linear(2, 2), "optimizer": {"lr": 0.01},
            "train_args": {"epochs": 300}}
    p = tmp_path / "last.pt"
    torch.save(ckpt, p)
    try:
        torch.load(p)
        stock_ok = True
    except Exception:
        stock_ok = False
    got = torch.load(p, weights_only=False)
    assert got["epoch"] == 7 and isinstance(got["model"], torch.nn.Linear)
    if stock_ok:
        import pytest
        pytest.skip("this torch still defaults weights_only=False; the fix is future-proofing")


# ---------------------------------------------------------------------------
# Faults four and five, both found by the first DDP resume on GenomeDK.
#
#   * `-r` was `action="store_true"`, so `-r last.pt` left the path in `extra`, where the
#     pairwise `--key value` loop dropped it without a word. The run then "resumed" from the
#     config's `model: yolo26m-seg.pt` - a real checkpoint with real train_results - so it
#     started over from the COCO pretrain while every log line said resume.
#   * `__init__` overwrote `self.args.resume` with the bare `True`. `generate_ddp_file`
#     serialises `vars(self.args)` into the file each rank executes, so the ranks got a bool:
#     `os.path.splitext(True)` raised TypeError, and had it not, ultralytics' `check_resume`
#     would have fallen back to `get_latest_run()` and resumed somebody else's run.
# ---------------------------------------------------------------------------

import os

import pytest

from flat_bug.cli.fb_train import _make_parser


def _ckpt(tmp_path, name="last.pt", epochs=500, done=12):
    """A checkpoint shaped like the ones the trainer writes."""
    ckpt = {
        "epoch": done,
        "model": torch.nn.Linear(2, 2),
        "train_args": {"epochs": epochs, "model": str(tmp_path / "src.pt")},
        "train_results": {"epoch": list(range(1, done + 1))},
    }
    p = tmp_path / name
    torch.save(ckpt, p)
    return p


def test_resume_flag_keeps_its_path():
    args, extra = _make_parser().parse_known_args(["-r", "/runs/x/weights/last.pt"])
    assert args.resume == "/runs/x/weights/last.pt", "the checkpoint path must not be swallowed"
    assert extra == [], "a swallowed path lands here, where main() silently discards it"


def test_bare_resume_still_means_true():
    args, _ = _make_parser().parse_known_args(["-r"])
    assert args.resume is True
    assert _make_parser().parse_known_args([])[0].resume is False


def test_resume_rejects_a_bool_checkpoint(tmp_path):
    """What a DDP rank used to receive. A clear error beats TypeError from posixpath."""
    with pytest.raises(NotImplementedError, match="resume=<checkpoint>.pt"):
        T.apply_overrides_to_checkpoint({"resume": True, "project": str(tmp_path), "name": "r"})


def test_resume_rewrites_the_checkpoint(tmp_path):
    src = _ckpt(tmp_path)
    overrides = {"resume": str(src), "project": str(tmp_path / "proj"), "name": "run", "epochs": 500}
    T.apply_overrides_to_checkpoint(overrides)
    assert overrides["resume"] != str(src), "must point at the patched copy, not the original"
    patched = torch.load(overrides["resume"], weights_only=False, map_location="cpu")
    assert patched["epoch"] == 12, "the epoch to continue from"
    assert patched["train_args"]["epochs"] == 500


def test_ddp_ranks_do_not_rewrite_the_checkpoint(tmp_path, monkeypatch):
    """Rank-local rewrites would fork one save_dir per rank via increment_path."""
    src = _ckpt(tmp_path)
    monkeypatch.setenv("LOCAL_RANK", "3")
    overrides = {"resume": str(src), "project": str(tmp_path / "proj"), "name": "run"}
    T.apply_overrides_to_checkpoint(overrides)
    assert overrides["resume"] == str(src), "a rank must use the parent's patched checkpoint as-is"
    assert not os.path.exists(tmp_path / "proj" / "resume_weights")


def test_args_resume_stays_a_path_for_ddp():
    """`vars(self.args)` is what the DDP ranks are handed; a bool there is unrecoverable."""
    src = inspect.getsource(T.FlatBugSegmentationTrainer.__init__)
    assert "self.args.resume = True" not in src, "a bare True reaches every rank through generate_ddp_file"
    assert 'self.args.resume = os.fspath(updated_overrides["resume"])' in src


def test_resume_does_not_skip_an_epoch(tmp_path):
    """results.csv counts from 1 and ckpt["epoch"] from 0; conflating them lost an epoch."""
    src = _ckpt(tmp_path, done=12)  # 12 epochs finished -> ckpt["epoch"] == 11
    torch.save({**torch.load(src, weights_only=False, map_location="cpu"), "epoch": 11}, src)
    overrides = {"resume": str(src), "project": str(tmp_path / "p"), "name": "run"}
    T.apply_overrides_to_checkpoint(overrides)
    patched = torch.load(overrides["resume"], weights_only=False, map_location="cpu")
    # ultralytics does start_epoch = ckpt["epoch"] + 1, so 11 resumes at the 13th epoch.
    assert patched["epoch"] == 11, "one epoch silently skipped per resume, 15 over a chained run"


def test_stripped_checkpoint_falls_back_to_the_csv(tmp_path):
    """`strip_optimizer` stamps epoch = -1 on a finished run's weights."""
    src = _ckpt(tmp_path, done=12)
    torch.save({**torch.load(src, weights_only=False, map_location="cpu"), "epoch": -1}, src)
    overrides = {"resume": str(src), "project": str(tmp_path / "p"), "name": "run"}
    T.apply_overrides_to_checkpoint(overrides)
    patched = torch.load(overrides["resume"], weights_only=False, map_location="cpu")
    assert patched["epoch"] == 11
