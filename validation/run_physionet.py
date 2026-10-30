"""Validate alphabench on the PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMIDB).

Runs R01 (1 min eyes-open baseline) and R02 (1 min eyes-closed baseline) of
all 109 subjects (Schalk et al., 2004; Goldberger et al., 2000), 64-channel
BCI2000 cap, 160 Hz, recorded in the USA (60 Hz mains).

Pre-registered choices (fixed before the first run, see README):
  * verdict channels: the 8 parieto-occipital / occipital sites below
  * default Config() except line_freq=60
  * subjects processed one at a time; nothing is excluded

Usage::

    python validation/run_physionet.py            # all 109 subjects
    python validation/run_physionet.py --subjects 1 2 3

Downloads ~2 x 1.2 MB per subject into MNE's data folder on first use.
Writes CSV/JSON/Markdown to validation/results/ and PNG figures to
validation/figures/ (by calling make_figures.py).
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import mne
from mne.datasets import eegbci

from alphabench import Config, __version__, analyze
from alphabench import signal_proc as sp
from alphabench.io import from_mne_raw
from alphabench.stats import wilson_ci

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
POSTERIOR = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
FRONTAL = ["Fp1", "Fpz", "Fp2"]
FMAX = 40.0
CFG = Config(line_freq=60.0)


def load_run(subject: int, run: int):
    (path,) = eegbci.load_data(subject, [run], update_path=False, verbose="ERROR")
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    eegbci.standardize(raw)
    rec = from_mne_raw(raw, source=f"EEGMMIDB S{subject:03d}R{run:02d}")
    del raw
    return rec


def subject_result(s: int):
    eo, ec = load_run(s, 1), load_run(s, 2)
    res = analyze(eo, ec, CFG, verdict_channels=POSTERIOR)
    del eo, ec
    ch = res.channels
    post = ch.loc[POSTERIOR]
    front = ch.loc[FRONTAL]
    ip = [list(ch.index).index(c) for c in POSTERIOR]
    m = res.f <= FMAX
    psd_eo = np.nanmean(res.mean_psd_eo[ip], 0)
    psd_ec = np.nanmean(res.mean_psd_ec[ip], 0)
    pk = sp.alpha_peak(res.f, psd_ec, CFG)
    row = {
        "subject": s, "sfreq": res.meta["eo_sfreq"],
        "dur_eo_s": round(res.meta["eo_duration_s"], 2), "dur_ec_s": round(res.meta["ec_duration_s"], 2),
        "verdict": res.verdict, "n_pass_channels": int(post["pass"].sum()),
        "n_eligible": int(post["eligible"].sum()),
        "post_reactivity_abs": float(post["reactivity_abs"].median()),
        "post_reactivity_rel": float(post["reactivity_rel"].median()),
        "post_specificity": float(post["specificity"].median()),
        "post_min_p_bonf": float(post["p_bonf"].min()),
        "post_max_cliffs_delta": float(post["cliffs_delta"].max()),
        "post_clean_eo": float(post["n_eo"].median()), "post_clean_ec": float(post["n_ec"].median()),
        "iaf_hz": pk["paf"], "iaf_cog_hz": pk["cog"], "iaf_peak_db": pk["peak_db"],
        "front_reactivity_abs": float(front["reactivity_abs"].median()),
        "front_reactivity_rel": float(front["reactivity_rel"].median()),
        "front_low_ratio_eo_ec": float(front["low_ratio_eo_ec"].median()),
        "post_low_ratio_eo_ec": float(post["low_ratio_eo_ec"].median()),
    }
    long = post.reset_index().assign(subject=s)
    topo = (np.log2(ch["reactivity_abs"]).rename(s), np.log2(ch["reactivity_rel"]).rename(s))
    return row, long, topo, res.f[m], psd_eo[m], psd_ec[m]


def ablation(long: pd.DataFrame, cfg: Config) -> dict:
    """Subject pass rate when each PASS criterion is dropped (sensitivity analysis).

    Reported for transparency only; the verdict itself is never changed.
    """
    crit = {
        "p_bonferroni": lambda d: d["p_mwu"] < cfg.alpha_level / d["n_elig"],
        "reactivity": lambda d: d["reactivity_abs"] >= cfg.min_reactivity,
        "specificity": lambda d: d["specificity"] >= cfg.min_specificity,
        "cliffs_delta": lambda d: d["cliffs_delta"] >= cfg.min_cliffs_delta,
        "alpha_peak": lambda d: d["peak_db"] >= cfg.min_peak_db,
    }
    d = long[long["eligible"]].copy()
    d["n_elig"] = d.groupby("subject")["eligible"].transform("sum")
    n_sub = long["subject"].nunique()
    out = {}
    for drop in [None, *crit]:
        ok = np.ones(len(d), bool)
        for name, fn in crit.items():
            if name != drop:
                ok &= fn(d).to_numpy()
        out["all criteria" if drop is None else f"without {drop}"] = round(
            d.loc[ok, "subject"].nunique() / n_sub, 4)
    return out


def summarize(sub: pd.DataFrame, long: pd.DataFrame) -> dict:
    n = len(sub)
    counts = sub["verdict"].value_counts().reindex(["PASS", "INCONCLUSIVE", "FAIL"], fill_value=0)
    k = int(counts["PASS"])
    r = sub["post_reactivity_abs"]
    w = stats.wilcoxon(np.log(r), alternative="greater")
    iaf = sub.loc[sub["iaf_peak_db"] >= CFG.min_peak_db, "iaf_hz"].dropna()  # clear peaks only
    q = lambda x, p: float(np.nanpercentile(x, p))  # noqa: E731
    return {
        "alphabench_version": __version__,
        "dataset": "PhysioNet EEGMMIDB v1.0.0, runs R01 (EO) / R02 (EC)",
        "n_subjects": n,
        "verdict_channels": POSTERIOR,
        "verdict_counts": {k_: int(v) for k_, v in counts.items()},
        "pass_rate": k / n, "pass_rate_wilson95": wilson_ci(k, n),
        "reactivity_abs_median": float(r.median()),
        "reactivity_abs_iqr": [q(r, 25), q(r, 75)],
        "frac_reactivity_gt_1": float((r > 1).mean()),
        "frac_reactivity_ge_threshold": float((r >= CFG.min_reactivity).mean()),
        "wilcoxon_log_reactivity_gt_0": {"statistic": float(w.statistic), "p": float(w.pvalue)},
        "iaf_n_detected": int(iaf.size),
        "iaf_mean_hz": float(iaf.mean()), "iaf_sd_hz": float(iaf.std(ddof=1)),
        "iaf_range_hz": [float(iaf.min()), float(iaf.max())],
        "iaf_cog_mean_hz": float(sub["iaf_cog_hz"].mean()), "iaf_cog_sd_hz": float(sub["iaf_cog_hz"].std(ddof=1)),
        "frontal": {
            "reactivity_abs_median": float(sub["front_reactivity_abs"].median()),
            "reactivity_rel_median": float(sub["front_reactivity_rel"].median()),
            "low_ratio_eo_ec_median": float(sub["front_low_ratio_eo_ec"].median()),
        },
        "posterior": {
            "reactivity_rel_median": float(sub["post_reactivity_rel"].median()),
            "low_ratio_eo_ec_median": float(sub["post_low_ratio_eo_ec"].median()),
            "specificity_median": float(sub["post_specificity"].median()),
        },
        "ablation_pass_rate": ablation(long, CFG),
        "config": CFG.to_dict(),
    }


def write_markdown(s: dict, path: Path) -> None:
    lo, hi = s["pass_rate_wilson95"]
    c = s["verdict_counts"]
    iqr = s["reactivity_abs_iqr"]
    lines = [
        "# PhysioNet EEGMMIDB validation (auto-generated)", "",
        f"alphabench {s['alphabench_version']}, {s['n_subjects']} subjects, verdict channels "
        f"{', '.join(s['verdict_channels'])}.", "",
        "| metric | value |", "|---|---|",
        f"| PASS / INCONCLUSIVE / FAIL | {c['PASS']} / {c['INCONCLUSIVE']} / {c['FAIL']} |",
        f"| pass rate (Wilson 95% CI) | {100 * s['pass_rate']:.1f}% ({100 * lo:.1f}-{100 * hi:.1f}%) |",
        f"| posterior alpha reactivity EC/EO, median [IQR] | {s['reactivity_abs_median']:.2f} [{iqr[0]:.2f}-{iqr[1]:.2f}] |",
        f"| subjects with reactivity > 1 | {100 * s['frac_reactivity_gt_1']:.1f}% |",
        f"| subjects with reactivity >= {CFG.min_reactivity} | {100 * s['frac_reactivity_ge_threshold']:.1f}% |",
        f"| Wilcoxon signed-rank, log reactivity > 0 | p = {s['wilcoxon_log_reactivity_gt_0']['p']:.2g} |",
        f"| IAF (peak), mean +/- sd (n with peak) | {s['iaf_mean_hz']:.2f} +/- {s['iaf_sd_hz']:.2f} Hz (n = {s['iaf_n_detected']}) |",
        f"| IAF (centre of gravity), mean +/- sd | {s['iaf_cog_mean_hz']:.2f} +/- {s['iaf_cog_sd_hz']:.2f} Hz |",
        f"| frontal (Fp) reactivity: absolute vs relative, median | {s['frontal']['reactivity_abs_median']:.2f} vs {s['frontal']['reactivity_rel_median']:.2f} |",
        f"| frontal EO/EC 1-4 Hz power, median | {s['frontal']['low_ratio_eo_ec_median']:.2f} |",
        "", "## Sensitivity: pass rate if one criterion is dropped", "",
        "| criteria | pass rate |", "|---|---|",
        *[f"| {k} | {100 * v:.1f}% |" for k, v in s["ablation_pass_rate"].items()], "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 110)))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    mne.set_log_level("ERROR")
    RESULTS.mkdir(exist_ok=True)

    rows, longs, topos, psd_eo, psd_ec, f = [], [], [], [], [], None
    t0 = time.time()
    for s in a.subjects:
        try:
            row, long, topo, f_s, pe, pc = subject_result(s)
        except Exception as e:  # report, never silently drop
            print(f"S{s:03d} ERROR {e!r}", file=sys.stderr)
            rows.append({"subject": s, "verdict": "ERROR"})
            continue
        if f is None:
            f = f_s
        n = min(f.size, f_s.size)
        rows.append(row); longs.append(long); topos.append(topo)
        psd_eo.append(pe[:n]); psd_ec.append(pc[:n])
        print(f"S{s:03d} {row['verdict']:<12} R={row['post_reactivity_abs']:.2f} "
              f"IAF={row['iaf_hz']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        gc.collect()

    sub = pd.DataFrame(rows)
    long = pd.concat(longs, ignore_index=True)
    sub.to_csv(RESULTS / "subjects.csv", index=False, float_format="%.5g")
    cols = ["subject", "channel", "n_eo", "n_ec", "alpha_eo_uv2", "alpha_ec_uv2", "reactivity_abs",
            "reactivity_rel", "reactivity_flank", "specificity", "low_ratio_eo_ec", "p_mwu", "p_bonf",
            "cliffs_delta", "p_perm", "iaf_hz", "peak_db", "eligible", "pass", "flags"]
    long[cols].to_csv(RESULTS / "posterior_channels.csv", index=False, float_format="%.5g")
    for k, name in enumerate(("abs", "rel")):
        pd.DataFrame([t[k] for t in topos]).to_csv(
            RESULTS / f"channel_log2_reactivity_{name}.csv", float_format="%.4g", index_label="subject")
    n = min(len(x) for x in psd_eo)
    le, lc = np.log10(np.stack([x[:n] for x in psd_eo])), np.log10(np.stack([x[:n] for x in psd_ec]))
    g = pd.DataFrame({"freq_hz": f[:n],
                      "eo_log10_mean": le.mean(0), "eo_log10_sem": stats.sem(le, axis=0),
                      "ec_log10_mean": lc.mean(0), "ec_log10_sem": stats.sem(lc, axis=0),
                      "diff_db_mean": 10 * (lc - le).mean(0), "diff_db_sem": 10 * stats.sem(lc - le, axis=0),
                      "n_subjects": le.shape[0]})
    g.to_csv(RESULTS / "group_psd.csv", index=False, float_format="%.6g")

    ok = sub[sub["verdict"] != "ERROR"]
    summary = summarize(ok, long)
    summary["n_errors"] = int((sub["verdict"] == "ERROR").sum())
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(summary, RESULTS / "summary.md")
    print((RESULTS / "summary.md").read_text())

    if not a.no_figures:
        import make_figures

        make_figures.main()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
