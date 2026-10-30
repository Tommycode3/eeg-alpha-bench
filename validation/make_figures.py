"""Draw the README figures from validation/results/*.csv (no raw data needed).

Figures 1-4 use the real PhysioNet results; figure 5 uses synthetic data with
known ground truth to demonstrate the relative-alpha / EOG pitfall.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from alphabench import Config, analyze
from alphabench import signal_proc as sp
from alphabench.plotting import ALPHA_SHADE, EC_COLOR, EO_COLOR, NEUTRAL, shade_alpha, use_style
from alphabench.synthetic import make_pair

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
VERDICT_COLORS = {"PASS": "#2e7d4f", "INCONCLUSIVE": "#d39b2a", "FAIL": "#b3261e"}


def _ci(sem: np.ndarray, n: int) -> np.ndarray:
    return stats.t.ppf(0.975, n - 1) * sem


def fig_group_psd(summary: dict) -> None:
    g = pd.read_csv(RES / "group_psd.csv")
    g = g[(g.freq_hz >= 1) & (g.freq_hz <= 40)]
    n = int(g.n_subjects.iloc[0])
    f = g.freq_hz.to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (a, b):
        shade_alpha(ax)
    for key, col, lab in (("eo", EO_COLOR, "Eyes open (R01)"), ("ec", EC_COLOR, "Eyes closed (R02)")):
        m, h = g[f"{key}_log10_mean"].to_numpy(), _ci(g[f"{key}_log10_sem"].to_numpy(), n)
        a.plot(f, 10 ** m, color=col, label=lab)
        a.fill_between(f, 10 ** (m - h), 10 ** (m + h), color=col, alpha=0.22, lw=0)
    a.set_yscale("log")
    a.set(xlim=(1, 40), xlabel="Frequency (Hz)", ylabel=r"PSD ($\mu V^2$/Hz), geometric mean",
          title=f"a  Posterior PSD, {n} subjects (mean, 95% CI)")
    a.legend(loc="upper right")
    d, h = g.diff_db_mean.to_numpy(), _ci(g.diff_db_sem.to_numpy(), n)
    b.axhline(0, color=NEUTRAL, lw=0.8)
    b.plot(f, d, color=NEUTRAL)
    b.fill_between(f, d - h, d + h, color=NEUTRAL, alpha=0.2, lw=0)
    b.set(xlim=(1, 40), xlabel="Frequency (Hz)", ylabel="EC - EO (dB)",
          title="b  Eyes-closed minus eyes-open (95% CI)")
    k = int(np.argmax(d))
    b.annotate(f"+{d[k]:.1f} dB at {f[k]:.1f} Hz", (f[k], d[k]), xytext=(f[k] + 6, d[k] - 1),
               arrowprops={"arrowstyle": "-", "color": NEUTRAL, "lw": 0.7})
    fig.tight_layout()
    fig.savefig(FIG / "group_psd.png")
    plt.close(fig)


def fig_reactivity(summary: dict) -> None:
    sub = pd.read_csv(RES / "subjects.csv")
    sub = sub[sub.verdict != "ERROR"].sort_values("post_reactivity_abs").reset_index(drop=True)
    thr = summary["config"]["min_reactivity"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.35, 1]})
    for v, col in VERDICT_COLORS.items():
        m = sub.verdict == v
        a.scatter(np.flatnonzero(m) + 1, sub.post_reactivity_abs[m], s=14, color=col,
                  label=f"{v} ({m.sum()})", zorder=3, lw=0)
    a.axhline(1, color=NEUTRAL, lw=0.8)
    a.axhline(thr, color=NEUTRAL, lw=0.8, ls="--")
    a.text(len(sub), thr * 0.86, f"PASS threshold {thr:g}x", fontsize=8, color=NEUTRAL, ha="right")
    a.set_yscale("log")
    a.set(xlabel="Subject (sorted)", ylabel="Alpha reactivity EC/EO (median of 8 posterior ch.)",
          title="a  Per-subject absolute alpha reactivity")
    a.legend(loc="upper left", title="verdict", title_fontsize=8.5)
    # relative vs absolute: frontal vs posterior
    for pre, col, lab in (("post", "#5b5b9a", "Posterior (PO/O)"), ("front", "#c47f17", "Frontal (Fp)")):
        b.scatter(sub[f"{pre}_reactivity_abs"], sub[f"{pre}_reactivity_rel"], s=12, color=col,
                  alpha=0.75, lw=0, label=lab)
    lim = (0.3, 100)
    b.plot(lim, lim, color=NEUTRAL, lw=0.8)
    b.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="Absolute alpha reactivity EC/EO",
          ylabel="Relative alpha reactivity EC/EO", title="b  Relative vs absolute, per subject")
    fr = summary["frontal"]
    b.text(0.35, 45, f"Fp medians: abs {fr['reactivity_abs_median']:.2f}, rel {fr['reactivity_rel_median']:.2f}\n"
           f"Fp EO/EC 1-4 Hz power: {fr['low_ratio_eo_ec_median']:.1f}x", fontsize=8, va="top")
    b.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "reactivity.png")
    plt.close(fig)


def fig_iaf(summary: dict) -> None:
    sub = pd.read_csv(RES / "subjects.csv")
    thr = summary["config"]["min_peak_db"]
    iaf = sub.loc[sub.iaf_peak_db >= thr, "iaf_hz"].dropna()
    cog = sub.iaf_cog_hz.dropna()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bins = np.arange(7, 13.01, 0.25)
    ax.hist(iaf, bins=bins, color=EC_COLOR, alpha=0.85, label=f"Peak frequency (n = {iaf.size})")
    ax.hist(cog, bins=bins, histtype="step", color=NEUTRAL, lw=1.2, label=f"Centre of gravity (n = {cog.size})")
    mu, sd = iaf.mean(), iaf.std(ddof=1)
    ax.axvline(mu, color="black", lw=1)
    ax.axvspan(mu - sd, mu + sd, color=ALPHA_SHADE, zorder=0, lw=0)
    top = ax.get_ylim()[1] * 1.2
    ax.set_ylim(0, top)
    ax.text(12.9, top * 0.93, f"peak IAF: {mu:.2f} ± {sd:.2f} Hz (mean ± sd)",
            fontsize=8.5, ha="right")
    ax.set(xlabel="Individual alpha frequency (Hz)", ylabel="Subjects",
           title="Individual alpha frequency, eyes closed, posterior average")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(7, 13)
    fig.tight_layout()
    fig.savefig(FIG / "iaf_hist.png")
    plt.close(fig)


def fig_topomap(summary: dict) -> None:
    import mne

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ims = []
    for ax, name, title in zip(axes, ("abs", "rel"), ("a  Absolute alpha", "b  Relative alpha")):
        t = pd.read_csv(RES / f"channel_log2_reactivity_{name}.csv", index_col="subject")
        med = t.median(axis=0)
        info = mne.create_info(list(med.index), 160.0, "eeg")
        names = mne.channels.get_builtin_montages()
        info.set_montage("colin27_1005" if "colin27_1005" in names else "standard_1005",
                         on_missing="ignore")
        mask = np.array([c in summary["verdict_channels"] for c in med.index])
        im, _ = mne.viz.plot_topomap(
            med.to_numpy(), info, axes=ax, show=False, cmap="RdBu_r", vlim=(-2.5, 2.5),
            contours=0, mask=mask, sensors=True,
            mask_params={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black",
                         "markersize": 4.5})
        ims.append(im)
        ax.set_title(title, loc="left")
    cb = fig.colorbar(ims[0], ax=axes, shrink=0.75, pad=0.02)
    cb.set_label(r"median $\log_2$(EC/EO)")
    fig.suptitle(f"Group median alpha reactivity per electrode (n = {summary['n_subjects']}); "
                 "white = verdict channels", fontsize=9, x=0.02, ha="left")
    fig.savefig(FIG / "topomap.png")
    plt.close(fig)


def fig_eog_pitfall() -> None:
    """Synthetic: no alpha anywhere, ocular artifact only in eyes-open."""
    cfg = Config()
    eo, ec = make_pair(duration_s=120, eog_amp_uv=80.0, seed=7)
    r = analyze(eo, ec, cfg)
    i = eo.ch_names.index("Fp1")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.4, 1.1, 0.8]})
    a, b, c = axes
    t = np.arange(int(10 * eo.sfreq)) / eo.sfreq
    xe = sp.preprocess(eo.data[i:i + 1], eo.sfreq, cfg)[0][: t.size]
    xc = sp.preprocess(ec.data[i:i + 1], ec.sfreq, cfg)[0][: t.size]
    a.plot(t, xe + 60, color=EO_COLOR, lw=0.8, label="Eyes open (+ blinks/saccades)")
    a.plot(t, xc - 60, color=EC_COLOR, lw=0.8, label="Eyes closed")
    a.set(xlabel="Time (s)", ylabel=r"Fp1 ($\mu V$, offset)", title="a  Synthetic Fp1, no alpha in either")
    a.legend(loc="upper right", fontsize=7.5)
    f = r.f
    m = (f >= 1) & (f <= 40)
    shade_alpha(b)
    b.semilogy(f[m], r.mean_psd_eo[i, m], color=EO_COLOR, label="EO")
    b.semilogy(f[m], r.mean_psd_ec[i, m], color=EC_COLOR, label="EC")
    b.set(xlabel="Frequency (Hz)", ylabel=r"PSD ($\mu V^2$/Hz)", xlim=(1, 40), title="b  Same alpha, bigger denominator")
    b.legend(loc="upper right")
    row = r.channels.loc["Fp1"]
    vals = [row.reactivity_rel, row.reactivity_abs]
    bars = c.bar(["relative\nalpha", "absolute\nalpha"], vals, color=["#c47f17", "#5b5b9a"], width=0.6)
    c.axhline(1, color=NEUTRAL, lw=0.8)
    c.axhline(cfg.min_reactivity, color=NEUTRAL, lw=0.8, ls="--")
    for bar, v in zip(bars, vals):
        c.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.2f}x", ha="center", fontsize=8.5)
    c.set(ylabel="EC/EO reactivity", title="c  False 'Berger effect'")
    c.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "eog_pitfall.png")
    plt.close(fig)


def main() -> None:
    use_style()
    FIG.mkdir(exist_ok=True)
    summary = json.loads((RES / "summary.json").read_text())
    fig_group_psd(summary)
    fig_reactivity(summary)
    fig_iaf(summary)
    fig_topomap(summary)
    fig_eog_pitfall()
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
