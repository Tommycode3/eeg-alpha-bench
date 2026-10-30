import numpy as np
import pytest

from alphabench import Config
from alphabench import signal_proc as sp
from alphabench.stats import cliffs_delta, mann_whitney_greater, permutation_test_mean, wilson_ci


def test_band_power_of_sinusoid_equals_half_amplitude_squared():
    fs, A = 256.0, 20.0
    t = np.arange(int(8 * fs)) / fs
    x = A * np.sin(2 * np.pi * 10 * t)[None, None, :]
    f, p = sp.welch_psd(x, fs, 2.0)
    assert sp.band_power(f, p, (8, 12))[0, 0] == pytest.approx(A**2 / 2, rel=0.02)


def test_preprocess_removes_line_noise_and_drift_keeps_alpha():
    fs = 250.0
    t = np.arange(int(30 * fs)) / fs
    alpha = 10 * np.sin(2 * np.pi * 10 * t)
    x = (alpha + 50 * np.sin(2 * np.pi * 50 * t) + 300 * t / t[-1])[None]
    y = sp.preprocess(x, fs, Config(line_freq=50.0))
    f, p = sp.welch_psd(y[None], fs, 2.0)
    assert sp.band_power(f, p, (49, 51))[0, 0] < 1e-3 * sp.band_power(f, p, (8, 12))[0, 0]
    assert sp.band_power(f, p, (8, 12))[0, 0] == pytest.approx(50.0, rel=0.05)


def test_epochs_and_rejection():
    x = np.zeros((2, 1000)); x[0] = np.random.default_rng(0).normal(0, 5, 1000)
    x[0, 100] = 500.0
    ep = sp.make_epochs(x, 100.0, 2.0)
    assert ep.shape == (2, 5, 200)
    good = sp.clean_mask(ep, Config())
    assert not good[0, 0] and good[0, 1:].all()  # spike epoch rejected
    assert not good[1].any()                     # flat channel rejected


def test_alpha_peak_requires_a_local_maximum():
    f = np.arange(0, 40.5, 0.5)
    f[0] = 0.25
    psd = 1 / f
    assert np.isnan(sp.alpha_peak(f, psd, Config())["paf"])
    psd2 = psd * (1 + 3 * np.exp(-0.5 * ((f - 9.7) / 0.8) ** 2))
    pk = sp.alpha_peak(f, psd2, Config())
    assert pk["paf"] == pytest.approx(9.7, abs=0.2) and pk["peak_db"] > 3


def test_stats():
    rng = np.random.default_rng(0)
    x, y = rng.normal(1, 1, 40), rng.normal(0, 1, 40)
    u, p = mann_whitney_greater(x, y)
    assert p < 0.01
    assert cliffs_delta(np.array([2, 3]), np.array([0, 1])) == 1.0
    assert cliffs_delta(x, y, u) == pytest.approx(cliffs_delta(x, y))
    _, pp = permutation_test_mean(x, y, 2000, 0)
    assert 1 / 2001 <= pp < 0.01
    lo, hi = wilson_ci(80, 100)
    assert lo < 0.8 < hi
