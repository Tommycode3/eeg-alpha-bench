"""Analysis parameters and pre-registered verdict thresholds.

Every number that influences the PASS / FAIL / INCONCLUSIVE verdict lives here,
so that it can be cited, changed deliberately, and diffed in version control.

The verdict thresholds below were fixed *before* the PhysioNet validation was
run (see the "Validation" section of the README). If you change them
after looking at results, say so in your report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    # ---- preprocessing -------------------------------------------------
    line_freq: float | None = 50.0
    """Mains frequency in Hz (50 EU/Asia, 60 Americas). ``None`` disables the notch."""
    notch_q: float = 30.0
    """Quality factor of the IIR notch (bandwidth ~ f0/Q)."""
    l_freq: float = 1.0
    """Band-pass lower edge (Hz)."""
    h_freq: float = 40.0
    """Band-pass upper edge (Hz); clipped to 0.45 * fs for low sampling rates."""
    filter_order: int = 4
    """Butterworth prototype order (applied forward-backward, zero phase)."""
    edge_s: float = 1.0
    """Seconds trimmed at both ends after filtering (filter transients)."""

    # ---- epoching and artifact rejection ------------------------------
    epoch_s: float = 4.0
    """Non-overlapping epoch length (s). Non-overlap keeps epochs ~independent."""
    reject_ptp_uv: float = 1000.0
    """Hard ceiling: reject a channel-epoch whose peak-to-peak amplitude exceeds this (uV)
    (saturation, electrode pops). Deliberately loose: see ``reject_z``."""
    reject_z: float = 3.5
    """Robust outlier rule, per channel and per condition: reject an epoch whose
    modified z-score of log peak-to-peak amplitude, 0.6745 (x - median) / MAD,
    exceeds this (Iglewicz & Hoaglin, 1993). Adapts to each device's amplitude
    scale and never rejects an epoch merely for containing strong alpha."""
    flat_ptp_uv: float = 1.0
    """Reject a channel-epoch whose peak-to-peak amplitude is below this (uV)."""

    # ---- spectral estimation ------------------------------------------
    welch_s: float = 2.0
    """Welch segment length (s) -> frequency resolution 1/welch_s = 0.5 Hz."""
    alpha_band: tuple[float, float] = (8.0, 12.0)
    total_band: tuple[float, float] = (1.0, 40.0)
    low_band: tuple[float, float] = (1.0, 4.0)
    """Band used as an ocular / low-frequency artifact index."""
    flank_bands: tuple[tuple[float, float], ...] = ((4.0, 7.0), (14.0, 30.0))
    """Bands next to alpha, used to separate an alpha-specific change from a broadband one."""
    iaf_search: tuple[float, float] = (7.0, 13.0)
    aperiodic_ranges: tuple[tuple[float, float], ...] = ((2.0, 7.0), (14.0, 30.0))
    """Frequency ranges (alpha excluded) used to fit the 1/f background."""

    # ---- statistics ---------------------------------------------------
    n_permutations: int = 5000
    seed: int = 0

    # ---- verdict thresholds (pre-registered) ---------------------------
    min_epochs: int = 10
    """Minimum clean epochs per condition for a channel to be eligible."""
    alpha_level: float = 0.01
    """Family-wise error rate; Bonferroni-divided by the number of eligible channels."""
    min_reactivity: float = 1.5
    """Minimum absolute-alpha reactivity EC/EO (ratio of medians) for PASS."""
    min_specificity: float = 1.3
    """Minimum alpha reactivity divided by flank-band reactivity for PASS: the change
    must be specific to alpha, not a broadband gain/impedance change between sessions."""
    min_cliffs_delta: float = 0.33
    """Minimum Cliff's delta (EC > EO) for PASS."""
    min_peak_db: float = 1.0
    """Minimum height of the EC alpha peak above the 1/f fit (dB) for PASS."""
    nominal_p: float = 0.05
    """Uncorrected p below which a channel counts as 'some evidence' (FAIL vs INCONCLUSIVE)."""
    nominal_reactivity: float = 1.2
    """Reactivity above which a channel counts as 'some evidence'."""
    eog_low_ratio: float = 2.0
    """EO/EC low-band power ratio above which a channel is flagged for ocular artifact."""

    def to_dict(self) -> dict:
        return asdict(self)
