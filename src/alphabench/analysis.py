"""Eyes-open vs eyes-closed alpha analysis and the PASS / FAIL / INCONCLUSIVE verdict."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import signal_proc as sp
from .config import Config
from .io import Recording
from .stats import cliffs_delta, mann_whitney_greater, permutation_test_mean

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

_FRONTAL = re.compile(r"^(FP|AF|F(?=[0-9]|Z$|Z\b))", re.IGNORECASE)


def is_frontal(name: str) -> bool:
    """10-20 / 10-10 frontal sites (Fp*, AF*, F1..F10, Fz), most exposed to EOG.

    FC*/FT* are not counted as frontal. Unknown labels (e.g. ``EXG0``) return
    False; the low-frequency EOG index still guards those channels.
    """
    return bool(_FRONTAL.match(name.strip()))


@dataclass
class ConditionSpectra:
    f: np.ndarray
    psd: np.ndarray  # (n_channels, n_epochs, n_freqs), uV^2/Hz
    good: np.ndarray  # (n_channels, n_epochs) bool


@dataclass
class Result:
    verdict: str
    reasons: list[str]
    channels: pd.DataFrame
    f: np.ndarray
    mean_psd_eo: np.ndarray  # (n_channels, n_freqs), mean over clean epochs
    mean_psd_ec: np.ndarray
    config: Config
    meta: dict = field(default_factory=dict)

    @property
    def passing_channels(self) -> list[str]:
        return list(self.channels.index[self.channels["pass"]])


def _spectra(rec: Recording, cfg: Config) -> ConditionSpectra:
    x = sp.preprocess(rec.data, rec.sfreq, cfg)
    ep = sp.make_epochs(x, rec.sfreq, cfg.epoch_s)
    good = sp.clean_mask(ep, cfg)
    f, psd = sp.welch_psd(ep, rec.sfreq, cfg.welch_s)
    return ConditionSpectra(f, psd, good)


def _common_grid(a: ConditionSpectra, b: ConditionSpectra) -> np.ndarray:
    if a.f.shape == b.f.shape and np.allclose(a.f, b.f):
        return a.f
    n = min(a.f.size, b.f.size)
    if not np.allclose(a.f[:n], b.f[:n]):
        raise ValueError("EO and EC frequency grids differ; use the same welch_s")
    return a.f[:n]


def analyze(
    eo: Recording,
    ec: Recording,
    cfg: Config | None = None,
    channels: list[str] | None = None,
    verdict_channels: list[str] | None = None,
) -> Result:
    """Compare eyes-closed (EC) with eyes-open (EO) alpha, channel by channel.

    Parameters
    ----------
    eo, ec
        Paired recordings of the same montage.
    channels
        Channels to analyse (default: all channels present in both).
    verdict_channels
        Subset allowed to produce the verdict (default: every analysed channel
        that is not frontal). Frontal channels are always excluded from the
        verdict, because ocular artifact (EOG) dominates them in EO.
    """
    cfg = cfg or Config()
    common = [c for c in eo.ch_names if c in ec.ch_names]
    lookup = {c.lower(): c for c in common}  # labels differ in case across devices
    requested = channels or common
    missing = [c for c in requested if c.lower() not in lookup]
    channels = [lookup[c.lower()] for c in requested if c.lower() in lookup]
    if verdict_channels is not None:
        verdict_channels = [lookup[c.lower()] for c in verdict_channels if c.lower() in lookup]
    if not channels:
        raise ValueError("no channels in common between EO and EC recordings")
    eo, ec = eo.pick(channels), ec.pick(channels)
    s_eo, s_ec = _spectra(eo, cfg), _spectra(ec, cfg)
    f = _common_grid(s_eo, s_ec)
    nf = f.size

    rows = []
    mean_eo = np.full((len(channels), nf), np.nan)
    mean_ec = np.full((len(channels), nf), np.nan)
    for i, ch in enumerate(channels):
        g_eo, g_ec = s_eo.good[i], s_ec.good[i]
        p_eo, p_ec = s_eo.psd[i, g_eo, :nf], s_ec.psd[i, g_ec, :nf]
        row = {"channel": ch, "frontal": is_frontal(ch),
               "n_eo": int(g_eo.sum()), "n_ec": int(g_ec.sum()),
               "n_eo_total": int(g_eo.size), "n_ec_total": int(g_ec.size)}
        if p_eo.shape[0] >= 2 and p_ec.shape[0] >= 2:
            mean_eo[i], mean_ec[i] = p_eo.mean(0), p_ec.mean(0)
            a_eo, a_ec = sp.band_power(f, p_eo, cfg.alpha_band), sp.band_power(f, p_ec, cfg.alpha_band)
            t_eo, t_ec = sp.band_power(f, p_eo, cfg.total_band), sp.band_power(f, p_ec, cfg.total_band)
            l_eo, l_ec = sp.band_power(f, p_eo, cfg.low_band), sp.band_power(f, p_ec, cfg.low_band)
            fl_eo = sum(sp.band_power(f, p_eo, b) for b in cfg.flank_bands)
            fl_ec = sum(sp.band_power(f, p_ec, b) for b in cfg.flank_bands)
            r_eo, r_ec = a_eo / t_eo, a_ec / t_ec
            la_eo, la_ec = np.log10(a_eo), np.log10(a_ec)
            u, p = mann_whitney_greater(la_ec, la_eo)
            d_log, p_perm = permutation_test_mean(la_ec, la_eo, cfg.n_permutations, cfg.seed + i)
            pk = sp.alpha_peak(f, mean_ec[i], cfg)
            row.update(
                alpha_eo_uv2=float(np.median(a_eo)), alpha_ec_uv2=float(np.median(a_ec)),
                rel_eo=float(np.median(r_eo)), rel_ec=float(np.median(r_ec)),
                reactivity_abs=float(np.median(a_ec) / np.median(a_eo)),
                reactivity_rel=float(np.median(r_ec) / np.median(r_eo)),
                reactivity_flank=float(np.median(fl_ec) / np.median(fl_eo)),
                low_ratio_eo_ec=float(np.median(l_eo) / np.median(l_ec)),
                p_mwu=p, cliffs_delta=cliffs_delta(la_ec, la_eo, u),
                mean_log10_diff=d_log, p_perm=p_perm,
                iaf_hz=pk["paf"], iaf_cog_hz=pk["cog"], peak_db=pk["peak_db"],
            )
        rows.append(row)

    df = pd.DataFrame(rows).set_index("channel")
    df["specificity"] = df.get("reactivity_abs", np.nan) / df.get("reactivity_flank", np.nan)
    for col in ("reactivity_abs", "reactivity_rel", "reactivity_flank", "specificity",
                "p_mwu", "cliffs_delta", "peak_db",
                "low_ratio_eo_ec", "iaf_hz", "alpha_eo_uv2", "alpha_ec_uv2", "rel_eo",
                "rel_ec", "iaf_cog_hz", "mean_log10_diff", "p_perm"):
        if col not in df:
            df[col] = np.nan
    verdict, reasons = _verdict(df, cfg, verdict_channels)
    meta = {
        "eo_source": eo.source, "ec_source": ec.source,
        "eo_sfreq": eo.sfreq, "ec_sfreq": ec.sfreq,
        "eo_duration_s": eo.duration_s, "ec_duration_s": ec.duration_s,
        "missing_channels": missing,
    }
    return Result(verdict, reasons, df, f, mean_eo, mean_ec, cfg, meta)


def _flags(r: pd.Series, cfg: Config) -> list[str]:
    fl = []
    if r["frontal"]:
        fl.append("frontal")
    if r["n_eo"] < cfg.min_epochs or r["n_ec"] < cfg.min_epochs:
        fl.append("few_epochs")
    if np.isfinite(r["low_ratio_eo_ec"]) and r["low_ratio_eo_ec"] >= cfg.eog_low_ratio:
        fl.append("eog_suspect")
    if (np.isfinite(r["reactivity_rel"]) and r["reactivity_rel"] >= cfg.min_reactivity
            and r["reactivity_abs"] < cfg.nominal_reactivity):
        fl.append("relative_only")
    return fl


def _verdict(df: pd.DataFrame, cfg: Config, verdict_channels: list[str] | None):
    df["flags"] = [",".join(_flags(r, cfg)) for _, r in df.iterrows()]
    pool = df.index if verdict_channels is None else [c for c in verdict_channels if c in df.index]
    eligible = [c for c in pool if not df.at[c, "frontal"]
                and df.at[c, "n_eo"] >= cfg.min_epochs and df.at[c, "n_ec"] >= cfg.min_epochs
                and np.isfinite(df.at[c, "p_mwu"])]
    m = max(len(eligible), 1)
    df["eligible"] = df.index.isin(eligible)
    df["p_bonf"] = np.where(df["eligible"], np.minimum(df["p_mwu"] * m, 1.0), np.nan)
    crit = (
        df["eligible"]
        & (df["p_mwu"] < cfg.alpha_level / m)
        & (df["reactivity_abs"] >= cfg.min_reactivity)
        & (df["specificity"] >= cfg.min_specificity)
        & (df["cliffs_delta"] >= cfg.min_cliffs_delta)
        & (df["peak_db"] >= cfg.min_peak_db)
    )
    df["pass"] = crit.fillna(False).astype(bool)
    nominal = df["eligible"] & (df["p_mwu"] < cfg.nominal_p) & (df["reactivity_abs"] >= cfg.nominal_reactivity)

    reasons: list[str] = []
    rel_only = [c for c in df.index if "relative_only" in df.at[c, "flags"]]
    if rel_only:
        reasons.append(
            "Relative alpha rose EO->EC without a matching rise in absolute alpha on "
            f"{', '.join(rel_only)}: the denominator (low-frequency/ocular power in EO) changed, "
            "not alpha. Relative alpha alone would have given a false positive here."
        )
    frontal_hits = [c for c in df.index if df.at[c, "frontal"]
                    and df.at[c, "reactivity_abs"] >= cfg.min_reactivity
                    and df.at[c, "p_mwu"] < cfg.nominal_p]
    if df["pass"].any():
        best = df[df["pass"]].sort_values("p_mwu").index.tolist()
        reasons.insert(0, f"Alpha reactivity detected on {', '.join(best)} "
                       f"({len(best)}/{len(eligible)} eligible channels).")
        return PASS, reasons
    if not eligible:
        why = "no non-frontal channel with enough clean epochs"
        if frontal_hits:
            why += f"; frontal channels {', '.join(frontal_hits)} react but cannot be separated from EOG"
        reasons.insert(0, f"Cannot decide: {why} (need >= {cfg.min_epochs} clean epochs per condition).")
        return INCONCLUSIVE, reasons
    if nominal.any():
        reasons.insert(0, "Some evidence of alpha reactivity on "
                       f"{', '.join(df.index[nominal])}, but no channel meets every PASS criterion.")
        return INCONCLUSIVE, reasons
    reasons.insert(0, f"No eligible channel ({len(eligible)}) shows even nominal alpha reactivity "
                   f"(p < {cfg.nominal_p} and EC/EO >= {cfg.nominal_reactivity}).")
    return FAIL, reasons
