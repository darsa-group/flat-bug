"""The LR anneal must be able to end lower than lr0/epochs.

ultralytics' linear schedule is lf(x) = (1 - x/epochs) * (1 - lrf) + lrf, so the last trained
epoch (x = epochs - 1) sits at roughly lr0/epochs regardless of lrf: with lr0 0.01 over 500
epochs that is 2.0e-5, while lrf's own floor is 1.0e-7. Dividing lrf by five moves the final
learning rate by 0.4%, which is why `fb_lr_power` exists - it bends the same curve down while
keeping lr0 and the epoch count fixed.
"""

import inspect
import math

import pytest

import flat_bug.trainers as T


def lf_linear(x, epochs, lrf):
    return max(1 - x / epochs, 0) * (1.0 - lrf) + lrf


def lf_power(x, epochs, lrf, power):
    return max(1 - x / epochs, 0) ** power * (1.0 - lrf) + lrf


def test_lrf_cannot_lower_the_final_lr():
    """The premise: this is why a power is needed at all."""
    a = lf_linear(499, 500, 1e-5)
    b = lf_linear(499, 500, 2e-6)  # lrf / 5
    assert abs(a - b) / a < 0.01, "if lrf moved the endpoint, fb_lr_power would be unnecessary"


# The exponent the 500-epoch run uses. Solving (1/E)**p * (1 - lrf) + lrf = lf_linear(E-1)/5
# rather than ignoring the lrf term, which is 2.5% of the target and would leave the endpoint
# high by that much.
POWER_DIV5 = 1.262227


def test_power_divides_the_final_lr_by_five():
    lin = lf_linear(499, 500, 1e-5)
    pw = lf_power(499, 500, 1e-5, POWER_DIV5)
    assert lin / pw == pytest.approx(5.0, rel=1e-4)


def test_power_starts_at_lr0_and_never_exceeds_linear():
    """Same starting point, and no epoch spends longer at high LR than the schedule we trust."""
    assert lf_power(0, 500, 1e-5, POWER_DIV5) == pytest.approx(1.0, rel=1e-6)
    for x in range(0, 500, 7):
        assert lf_power(x, 500, 1e-5, POWER_DIV5) <= lf_linear(x, 500, 1e-5) + 1e-12


def test_power_one_is_the_stock_schedule():
    for x in (0, 1, 250, 499):
        assert lf_power(x, 500, 1e-5, 1.0) == pytest.approx(lf_linear(x, 500, 1e-5))


def test_trainer_overrides_the_scheduler():
    src = inspect.getsource(T.FlatBugSegmentationTrainer)
    assert "def _setup_scheduler" in src, "must override, not patch self.lf after the fact"
    assert "self.args.cos_lr or self._lr_power == 1.0" in src, "cos_lr and power cannot both apply"
    assert 'custom_fb_args.get("fb_lr_power", 1.0)' in src


def test_power_is_exposed_and_defaults_to_stock_behaviour():
    import flat_bug.cli.fb_train as ft

    assert '"fb_lr_power": 1.0' in inspect.getsource(ft), "absent from DEFAULT_CONF breaks resume"


def test_power_must_be_positive():
    src = inspect.getsource(T.FlatBugSegmentationTrainer.__init__)
    assert "fb_lr_power must be > 0" in src


def test_the_exponent_we_intend_to_run():
    """Guard the constant itself, so a future edit to the config cannot silently drift."""
    E, lrf = 500, 1e-5
    target = lf_linear(E - 1, E, lrf) / 5
    p = math.log((target - lrf) / (1 - lrf)) / math.log(1 / E)
    assert p == pytest.approx(POWER_DIV5, abs=1e-5)
