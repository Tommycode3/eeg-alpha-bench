"""Preprocessing, epoching, artifact rejection and spectral estimation.

Pipeline (per channel, per recording)::

    x  ->  linear detrend  ->  notch(f_line, 2 f_line, ...)  ->  band-pass [l, h]
       ->  trim edges  ->  non-overlapping epochs  ->  reject (ptp, flat)
       ->  Welch PSD per epoch  ->  band powers

All filters are applied forward-backward (zero phase), so the alpha peak is
not shifted in frequency and its amplitude is not distorted inside the pass band.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

from .config import Config


def preprocess(x: np.ndarray, sfreq: float, cfg: Config) -> np.ndarray:
    """Detrend, notch and band-pass ``x`` (n_channels, n_samples, uV); trims edges."""
    x = signal.detrend(np.asarray(x, dtype=np.float64), axis=-1, type="linear")
    nyq = sfreq / 2.0
    if cfg.line_freq:
        k = 1
        while cfg.line_freq * k < nyq - 1.0:
            b, a = signal.iirnotch(cfg.line_freq * k, Q=cfg.notch_q, fs=sfreq)
            x = signal.filtfilt(b, a, x, axis=-1)
            k += 1
    h = min(cfg.h_freq, 0.45 * sfreq)
    sos = signal.butter(cfg.filter_order, [cfg.l_freq, h], btype="bandpass", fs=sfreq, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=-1)
    edge = int(round(cfg.edge_s * sfreq))
    if x.shape[-1] > 4 * edge > 0:
        x = x[:, edge:-edge]
    return x


def make_epochs(x: np.ndarray, sfreq: float, epoch_s: float) -> np.ndarray:
    """Cut into non-overlapping epochs: (n_channels, n_epochs, n_samples_per_epoch)."""
    n = int(round(epoch_s * sfreq))
    k = x.shape[-1] // n
    if k == 0:
        return np.empty((x.shape[0], 0, n))
    return x[:, : k * n].reshape(x.shape[0], k, n)


def clean_mask(epochs: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean (n_channels, n_epochs): True where the channel-epoch is kept.

    Three rules, per channel, applied to one condition at a time:

    1. flat: peak-to-peak (ptp) < ``flat_ptp_uv`` (disconnected / railed electrode);
    2. hard ceiling: ptp > ``reject_ptp_uv`` (saturation, electrode pop);
    3. robust outlier: modified z-score of log(ptp) > ``reject_z``, where
       z = 0.6745 (x - median(x)) / MAD(x) over the non-flat epochs.

    Rule 3 is relative to the condition's own amplitude distribution, so the
    large-amplitude alpha that defines eyes-closed is not itself rejected.
    Rejection is per channel, so one bad electrode does not discard the others.
    """
    if epochs.shape[1] == 0:
        return np.zeros(epochs.shape[:2], dtype=bool)
    ptp = np.ptp(epochs, axis=-1)
    good = (ptp >= cfg.flat_ptp_uv) & (ptp <= cfg.reject_ptp_uv)
    for c in range(ptp.shape[0]):
        ok = ptp[c] >= cfg.flat_ptp_uv
        if ok.sum() < 3:
            continue
        lx = np.log(ptp[c, ok])
        med = np.median(lx)
        mad = np.median(np.abs(lx - med))
        if mad > 0:
            z = np.zeros_like(ptp[c])
            z[ok] = 0.6745 * (lx - med) / mad
            good[c] &= z <= cfg.reject_z
    return good


def welch_psd(epochs: np.ndarray, sfreq: float, welch_s: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD (uV^2/Hz) along the last axis. Hann window, 50 % overlap."""
    n = int(round(welch_s * sfreq))
    n = min(n, epochs.shape[-1])
    f, p = signal.welch(
        epochs, fs=sfreq, window="hann", nperseg=n, noverlap=n // 2,
        detrend="constant", scaling="density", axis=-1,
    )
    return f, p


def band_power(f: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Integrate the PSD over ``band`` (inclusive) with the trapezoidal rule -> uV^2."""
    m = (f >= band[0]) & (f <= band[1])
    return trapezoid(psd[..., m], f[m], axis=-1)


def aperiodic_fit(
    f: np.ndarray, psd: np.ndarray, ranges: tuple[tuple[float, float], ...]
) -> tuple[float, float]:
    """Least-squares line in log-log space, ``log10 P = b - chi * log10 f``,
    fitted on ``ranges`` (which must exclude the alpha band). Returns (b, chi)."""
    m = np.zeros_like(f, dtype=bool)
    for lo, hi in ranges:
        m |= (f >= lo) & (f <= hi)
    m &= (f > 0) & (psd > 0)
    slope, intercept = np.polyfit(np.log10(f[m]), np.log10(psd[m]), 1)
    return float(intercept), float(-slope)


def alpha_peak(f: np.ndarray, psd: np.ndarray, cfg: Config) -> dict:
    """Individual alpha frequency (IAF) of a 1-D PSD.

    Returns a dict with
      * ``paf``  - peak alpha frequency (Hz): among the *local* maxima of the PSD
        strictly inside ``cfg.iaf_search``, the one standing highest above the
        1/f fit, refined by parabolic interpolation of log10 P
        over the three bins around the maximum; NaN if there is no local maximum.
      * ``peak_db`` - height of that maximum above the 1/f fit, 10 log10(P / P_fit).
      * ``cog`` - centre of gravity of the PSD above the 1/f fit within the
        search window, sum f (P - P_fit)^+ / sum (P - P_fit)^+ (Klimesch, 1999).
    """
    out = {"paf": np.nan, "peak_db": np.nan, "cog": np.nan}
    if not np.all(np.isfinite(psd)) or np.all(psd <= 0):
        return out
    b, chi = aperiodic_fit(f, psd, cfg.aperiodic_ranges)
    lo, hi = cfg.iaf_search
    idx = np.flatnonzero((f >= lo) & (f <= hi))
    fit = 10 ** (b - chi * np.log10(np.maximum(f, 1e-12)))
    resid = np.clip(psd[idx] - fit[idx], 0, None)
    if resid.sum() > 0:
        out["cog"] = float(np.sum(f[idx] * resid) / resid.sum())
    # local maxima strictly inside the search window
    cand = [j for j in idx[1:-1] if psd[j] > psd[j - 1] and psd[j] > psd[j + 1]]
    if not cand:
        return out
    j = max(cand, key=lambda k: psd[k] / fit[k])
    la, lb, lc = np.log10(psd[j - 1 : j + 2])
    denom = la - 2 * lb + lc
    delta = 0.5 * (la - lc) / denom if denom != 0 else 0.0
    out["paf"] = float(f[j] + delta * (f[1] - f[0]))
    out["peak_db"] = float(10 * np.log10(psd[j] / fit[j]))
    return out
