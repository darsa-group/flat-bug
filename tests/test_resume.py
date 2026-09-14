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
