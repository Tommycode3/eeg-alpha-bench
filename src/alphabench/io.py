"""Readers that turn device files into a common :class:`Recording`.

All readers return data in microvolts, shape ``(n_channels, n_samples)``.

Supported formats
-----------------
``edf`` / ``bdf`` / ``fif``
    Anything :func:`mne.io.read_raw` understands. EEG channels only.
``csv``
    Generic table: one optional time column plus one column per channel.
    The sampling rate comes from ``sfreq=`` or is estimated from the time column.
``muse``
    Muse headsets, as written by ``muselsl record`` (``timestamps,TP9,AF7,AF8,TP10,...``)
    or by the Mind Monitor app (``TimeStamp,...,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10,...``).
``openbci``
    OpenBCI GUI ``.txt`` exports (``%``-prefixed header, ``EXG Channel N`` columns).

To add a device, write a function ``read_<device>(path, **kw) -> Recording`` and
register it in :data:`READERS` (see the README section "Adding a device adapter").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

UNIT_SCALE = {"uv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


@dataclass
class Recording:
    data: np.ndarray
    """EEG in microvolts, shape (n_channels, n_samples)."""
    sfreq: float
    ch_names: list[str]
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError("data must be 2-D (n_channels, n_samples)")
        if self.data.shape[0] != len(self.ch_names):
            raise ValueError(
                f"{self.data.shape[0]} data rows but {len(self.ch_names)} channel names"
            )
        if not np.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError(f"invalid sampling rate {self.sfreq!r}")

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / self.sfreq

    def pick(self, channels: list[str]) -> "Recording":
        idx = [self.ch_names.index(c) for c in channels]
        return Recording(self.data[idx], self.sfreq, list(channels), self.source, dict(self.meta))


# --------------------------------------------------------------------------- helpers


def _estimate_sfreq(t: np.ndarray) -> float:
    """Sampling rate from a time column in seconds (median inter-sample interval)."""
    dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("cannot estimate sampling rate from time column; pass sfreq=")
    return float(1.0 / np.median(dt))


def _drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0, how="any").reset_index(drop=True)


# --------------------------------------------------------------------------- readers


def read_mne(path: str | Path, **_: object) -> Recording:
    """EDF, BDF, FIF, ... via MNE-Python. Keeps EEG channels only."""
    import mne

    raw = mne.io.read_raw(str(path), preload=True, verbose="ERROR")
    raw.pick("eeg")
    data = raw.get_data(units="uV")
    names = [n.strip().rstrip(".") for n in raw.ch_names]
    rec = Recording(data, float(raw.info["sfreq"]), names, str(path))
    del raw
    return rec


def from_mne_raw(raw, source: str = "") -> Recording:  # noqa: ANN001 - mne type optional
    """Convert an in-memory :class:`mne.io.Raw` (EEG channels only)."""
    raw = raw.copy().pick("eeg")
    return Recording(raw.get_data(units="uV"), float(raw.info["sfreq"]), list(raw.ch_names), source)


def read_csv(
    path: str | Path,
    sfreq: float | None = None,
    channels: list[str] | None = None,
    time_col: str | None = None,
    units: str = "uV",
    **_: object,
) -> Recording:
    """Generic CSV: optional time column (seconds) + one numeric column per channel."""
    df = pd.read_csv(path)
    if time_col is None:
        for c in df.columns:
            if str(c).strip().lower() in {"time", "t", "timestamp", "timestamps", "time_s"}:
                time_col = c
                break
    if sfreq is None:
        if time_col is None:
            raise ValueError("no time column found; pass sfreq=")
        sfreq = _estimate_sfreq(pd.to_numeric(df[time_col], errors="coerce").to_numpy())
    cols = channels or [
        c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        raise ValueError("no numeric channel columns found")
    sub = _drop_nan_rows(df[list(cols)].apply(pd.to_numeric, errors="coerce"))
    scale = UNIT_SCALE[units.lower()]
    return Recording(sub.to_numpy().T * scale, float(sfreq), [str(c) for c in cols], str(path))


MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


def read_muse(path: str | Path, sfreq: float | None = None, **_: object) -> Recording:
    """Muse CSV from ``muselsl record`` or Mind Monitor (raw channels, uV).

    Mind Monitor interleaves marker/"Elements" rows that have no raw sample;
    those rows are dropped. Muse 2 / Muse S stream EEG at 256 Hz.
    """
    df = pd.read_csv(path)
    cols = {c.strip(): c for c in df.columns}
    if all(f"RAW_{c}" in cols for c in MUSE_CHANNELS):  # Mind Monitor
        raw_cols = [cols[f"RAW_{c}"] for c in MUSE_CHANNELS]
        flavour = "mindmonitor"
        if "Elements" in cols:
            df = df[df[cols["Elements"]].isna()]
    elif all(c in cols for c in MUSE_CHANNELS):  # muselsl
        raw_cols = [cols[c] for c in MUSE_CHANNELS]
        flavour = "muselsl"
    else:
        raise ValueError(f"{path}: not a Muse CSV (expected TP9/AF7/AF8/TP10 columns)")
    sub = _drop_nan_rows(df[raw_cols].apply(pd.to_numeric, errors="coerce"))
    if sfreq is None:
        t = cols.get("timestamps")
        if t is not None and flavour == "muselsl":
            sfreq = _estimate_sfreq(pd.to_numeric(df[t], errors="coerce").to_numpy())
        else:
            sfreq = 256.0
    rec = Recording(sub.to_numpy().T, float(sfreq), list(MUSE_CHANNELS), str(path))
    rec.meta["flavour"] = flavour
    return rec


_SR_RE = re.compile(r"sample rate\s*=\s*([\d.]+)", re.IGNORECASE)


def read_openbci(path: str | Path, sfreq: float | None = None, **_: object) -> Recording:
    """OpenBCI GUI ``.txt`` export. Reads ``%Sample Rate = N Hz`` from the header."""
    header_lines = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("%"):
                break
            header_lines += 1
            m = _SR_RE.search(line)
            if m and sfreq is None:
                sfreq = float(m.group(1))
    if sfreq is None:
        raise ValueError("sample rate not found in OpenBCI header; pass sfreq=")
    df = pd.read_csv(path, skiprows=header_lines, skipinitialspace=True)
    exg = [c for c in df.columns if str(c).strip().lower().startswith("exg channel")]
    if exg:
        names = [f"EXG{int(str(c).split()[-1])}" for c in exg]
    else:  # legacy export without a column header row: index, ch1..chN, ...
        df = pd.read_csv(path, skiprows=header_lines, header=None, skipinitialspace=True)
        n_ch = 16 if df.shape[1] >= 20 else 8
        exg = list(df.columns[1 : 1 + n_ch])
        names = [f"EXG{i}" for i in range(n_ch)]
    sub = _drop_nan_rows(df[exg].apply(pd.to_numeric, errors="coerce"))
    return Recording(sub.to_numpy().T, float(sfreq), names, str(path))


READERS = {
    "mne": read_mne,
    "csv": read_csv,
    "muse": read_muse,
    "openbci": read_openbci,
}


def detect_format(path: str | Path) -> str:
    """Guess the reader from extension and, for text files, the first lines."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in {".edf", ".bdf", ".fif", ".gdf", ".vhdr", ".set", ".cnt"}:
        return "mne"
    with open(p, encoding="utf-8", errors="replace") as fh:
        head = "".join(fh.readline() for _ in range(5))
    if head.startswith("%") and "openbci" in head.lower():
        return "openbci"
    if "TP9" in head and "AF7" in head:
        return "muse"
    return "csv"


def load(path: str | Path, fmt: str = "auto", **kw: object) -> Recording:
    """Load any supported file. ``fmt`` is one of ``auto`` or a key of :data:`READERS`
    (``edf``/``bdf``/``fif`` are aliases of ``mne``)."""
    fmt = fmt.lower()
    if fmt in {"edf", "bdf", "fif"}:
        fmt = "mne"
    if fmt == "auto":
        fmt = detect_format(path)
    if fmt not in READERS:
        raise ValueError(f"unknown format {fmt!r}; choose from auto, {', '.join(READERS)}")
    return READERS[fmt](path, **kw)
