"""Synthetic EEG with known ground truth, for tests and for the EOG-pitfall figure.

Background: 1/f^chi ("pink-ish") noise, shaped in the frequency domain.
Alpha: a 10 Hz carrier with slowly varying amplitude (bursts), added to
posterior channels in eyes-closed. EOG: blinks (Gaussian bumps, ~300 ms)
and saccade steps, strongest at Fp sites, added in eyes-open.
"""

from __future__ import annotations

import numpy as np

from .io import Recording

DEFAULT_CHANNELS = ["Fp1", "Fp2", "O1", "O2"]


def colored_noise(n: int, sfreq: float, chi: float, rms_uv: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with PSD ~ 1/f^chi and *expected* RMS ``rms_uv``.

    The scale is set from the filter, not from the realised sample, so two
    independent draws have the same expected spectrum (no hidden gain change).
    """
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / sfreq)
    f[0] = f[1]
    h = f ** (-chi / 2)
    x = np.fft.irfft(spec * h, n)
    expected_var = 4.0 * np.sum(h[1:] ** 2) / n**2
    return x / np.sqrt(expected_var) * rms_uv


def alpha_bursts(n: int, sfreq: float, freq: float, amp_uv: float, rng: np.random.Generator) -> np.ndarray:
    """Amplitude-modulated alpha: carrier at ``freq`` with a smooth positive envelope."""
    t = np.arange(n) / sfreq
    env = np.abs(colored_noise(n, sfreq, 4.0, 1.0, rng)) + 0.5
    env /= env.mean()
    jitter = np.cumsum(rng.standard_normal(n)) * 0.002  # slow phase wander
    return amp_uv * env * np.sin(2 * np.pi * freq * t + jitter)


def eog(n: int, sfreq: float, amp_uv: float, rate_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Blinks (Gaussian, sigma 80 ms) at ~``rate_hz`` plus low-passed saccade steps."""
    x = np.zeros(n)
    t = np.arange(n) / sfreq
    n_blinks = rng.poisson(rate_hz * n / sfreq)
    for t0 in rng.uniform(0, n / sfreq, n_blinks):
        x += amp_uv * np.exp(-0.5 * ((t - t0) / 0.08) ** 2)
    steps = np.cumsum(rng.standard_normal(n) * (rng.random(n) < 0.5 / sfreq)) * amp_uv * 0.3
    k = int(0.05 * sfreq) or 1
    x += np.convolve(steps, np.ones(k) / k, mode="same")
    return x


def make_pair(
    duration_s: float = 120.0,
    sfreq: float = 256.0,
    alpha_amp_uv: float = 0.0,
    alpha_freq: float = 10.0,
    eog_amp_uv: float = 0.0,
    noise_rms_uv: float = 10.0,
    gain_ec: float = 1.0,
    channels: list[str] | None = None,
    seed: int = 0,
) -> tuple[Recording, Recording]:
    """Return (EO, EC) recordings.

    * EC gets ``alpha_amp_uv`` alpha on O1/O2 (and 30 % of it on frontal sites).
    * EO gets ``eog_amp_uv`` ocular artifact (100 % at Fp, 15 % at O).
    * Both get independent 1/f noise of the same statistics.
    * ``gain_ec`` multiplies the whole EC recording (a broadband amplitude
      change between sessions, e.g. impedance or amplifier gain): not alpha.
    """
    rng = np.random.default_rng(seed)
    channels = channels or DEFAULT_CHANNELS
    n = int(duration_s * sfreq)

    def base() -> np.ndarray:
        return np.stack([colored_noise(n, sfreq, 1.0, noise_rms_uv, rng) for _ in channels])

    eo, ec = base(), base()
    eog_src = eog(n, sfreq, eog_amp_uv, 0.3, rng) if eog_amp_uv else None
    for i, ch in enumerate(channels):
        frontal = ch.upper().startswith(("FP", "AF", "F"))
        if alpha_amp_uv:
            ec[i] += alpha_bursts(n, sfreq, alpha_freq, alpha_amp_uv * (0.3 if frontal else 1.0), rng)
        if eog_src is not None:  # one ocular source, volume-conducted with distance
            eo[i] += eog_src * (1.0 if frontal else 0.15)
    ec *= gain_ec
    return (Recording(eo, sfreq, list(channels), "synthetic:EO"),
            Recording(ec, sfreq, list(channels), "synthetic:EC"))
