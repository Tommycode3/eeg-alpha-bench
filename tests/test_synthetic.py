"""Ground-truth tests on synthetic EEG: the verifier for the method itself."""

import numpy as np
import pytest

from alphabench import FAIL, PASS, Config, analyze
from alphabench.synthetic import make_pair


@pytest.mark.parametrize("seed", range(3))
def test_injected_alpha_is_detected(seed):
    eo, ec = make_pair(alpha_amp_uv=6.0, alpha_freq=10.0, seed=seed)
    r = analyze(eo, ec)
    assert r.verdict == PASS
    assert set(r.passing_channels) <= {"O1", "O2"}
    for ch in ("O1", "O2"):
        assert r.channels.at[ch, "iaf_hz"] == pytest.approx(10.0, abs=0.3)
        assert r.channels.at[ch, "reactivity_abs"] > 2.0


@pytest.mark.parametrize("freq", [8.5, 11.5])
def test_iaf_tracks_true_frequency(freq):
    eo, ec = make_pair(alpha_amp_uv=8.0, alpha_freq=freq, seed=1)
    r = analyze(eo, ec)
    assert r.channels.at["O1", "iaf_hz"] == pytest.approx(freq, abs=0.3)


@pytest.mark.parametrize("seed", range(8))
def test_pure_noise_never_passes(seed):
    eo, ec = make_pair(seed=seed)
    assert analyze(eo, ec).verdict != PASS


@pytest.mark.parametrize("seed", range(5))
def test_eog_does_not_create_false_pass(seed):
    eo, ec = make_pair(eog_amp_uv=80.0, seed=seed)
    r = analyze(eo, ec)
    assert r.verdict != PASS
    # the pitfall is real: relative alpha on Fp1 rises EO->EC with no alpha present ...
    assert r.channels.at["Fp1", "reactivity_rel"] > 1.5
    # ... absolute alpha does not, and the channel is flagged
    assert r.channels.at["Fp1", "reactivity_abs"] < 1.2
    assert "relative_only" in r.channels.at["Fp1", "flags"]
    assert "eog_suspect" in r.channels.at["Fp1", "flags"]


def test_frontal_only_alpha_cannot_pass():
    eo, ec = make_pair(alpha_amp_uv=10.0, channels=["Fp1", "AF7", "F3"], seed=0)
    r = analyze(eo, ec)
    assert r.verdict != PASS
    assert not r.channels["eligible"].any()


def test_too_little_data_is_inconclusive_not_fail():
    eo, ec = make_pair(duration_s=20.0, alpha_amp_uv=10.0, seed=0)
    assert analyze(eo, ec).verdict == "INCONCLUSIVE"


def test_flat_channel_is_rejected():
    eo, ec = make_pair(alpha_amp_uv=6.0, seed=0)
    eo.data[3] = 0.0
    ec.data[3] = 0.0
    r = analyze(eo, ec)
    assert r.channels.at["O2", "n_eo"] == 0 and r.channels.at["O2", "n_ec"] == 0
    assert r.verdict == PASS  # O1 still carries it


def test_noise_level_false_positive_rate_is_low():
    """Across many null pairs, PASS must essentially never happen."""
    n_pass = sum(analyze(*make_pair(duration_s=60, seed=100 + s)).verdict == PASS for s in range(20))
    assert n_pass == 0


@pytest.mark.parametrize("seed", range(3))
def test_broadband_gain_change_is_not_alpha(seed):
    """EC recorded with 2x the amplitude everywhere (e.g. better contact): every band
    'reacts', but nothing is specific to alpha, so the verdict must not be PASS."""
    eo, ec = make_pair(gain_ec=2.0, seed=seed)
    r = analyze(eo, ec)
    assert r.channels.loc[["O1", "O2"], "reactivity_abs"].min() > 3.0
    assert r.verdict != PASS
    assert r.channels.loc[["O1", "O2"], "specificity"].max() < 1.3


def test_strong_alpha_with_eog_still_passes():
    eo, ec = make_pair(alpha_amp_uv=6.0, eog_amp_uv=80.0, seed=3)
    assert analyze(eo, ec).verdict == PASS


def test_channel_selection_is_case_insensitive():
    eo, ec = make_pair(alpha_amp_uv=6.0, seed=0)
    r = analyze(eo, ec, channels=["o1", "O2", "Xx9"])
    assert list(r.channels.index) == ["O1", "O2"]
    assert r.meta["missing_channels"] == ["Xx9"]
