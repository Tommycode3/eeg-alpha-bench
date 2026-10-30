# eeg-alpha-bench

![Group PSD, eyes open vs eyes closed, 109 subjects](validation/figures/group_psd.png)

[![tests](https://github.com/Tommycode3/eeg-alpha-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/Tommycode3/eeg-alpha-bench/actions/workflows/tests.yml)

A benchmark for the oldest sanity check in EEG: posterior alpha power (8–12 Hz) should rise when the eyes close (Berger, 1929). Give it one eyes-open and one eyes-closed recording from any headset and it returns PASS, FAIL or INCONCLUSIVE with the numbers behind the verdict. On all 109 subjects of PhysioNet EEGMMIDB, eyes closed raises posterior power by +7.5 dB at 10.0 Hz, and 90 of 109 subjects pass.

<!-- TODO(Tommaso): dictate 3-4 lines in your own words -->

## Install and run

```bash
git clone https://github.com/Tommycode3/eeg-alpha-bench && cd eeg-alpha-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# record ~2 min eyes open (fixating a point) and ~2 min eyes closed, relaxed, then:
alphabench --eo eyes_open.edf --ec eyes_closed.edf --line-freq 50 --report report.html
```

```text
PASS
  - Alpha reactivity detected on Po7, Po3, Poz, Po4, Po8, O1, Oz, O2 (8/8 eligible channels).
report: report.html
```

The report is one self-contained HTML file; see [`examples/S004_report.html`](examples/S004_report.html) for PhysioNet subject S004. Readers exist for EDF/BDF/FIF/BrainVision/EEGLAB (via MNE), generic CSV (`--format csv`, a `time` column or `--sfreq`), Muse (`muselsl record` and Mind Monitor) and the OpenBCI GUI `.txt`. Other flags: `--channels O1,O2`, `--line-freq 60`, `--json out.json`, and `--strict`, which exits 1 unless PASS so you can put it in a hardware CI job. A new device needs one function `path -> Recording` (data in µV) registered in `READERS` in [`src/alphabench/io.py`](src/alphabench/io.py).

```python
from alphabench import analyze, load, Config
res = analyze(load("eo.csv"), load("ec.csv"), Config(line_freq=50))
res.verdict, res.reasons, res.channels  # str, list[str], pandas.DataFrame
```

`pytest` runs 40 tests in ~7 s against synthetic signals with known truth: bursting 10 Hz alpha must PASS with IAF within 0.3 Hz, while pure 1/f noise, EOG-only eyes-open, a 2× gain change and frontal-only alpha must never PASS.

## Results on PhysioNet

109 subjects, 64-channel BCI2000 cap, 160 Hz, run R01 (1 min eyes open) vs R02 (1 min eyes closed), default config with `line_freq=60`, verdict channels PO7, PO3, POz, PO4, PO8, O1, Oz, O2 fixed in advance. No subject excluded.

| metric | value |
|---|---|
| PASS / INCONCLUSIVE / FAIL | 90 / 14 / 5 |
| pass rate (Wilson 95 % CI) | 82.6 % (74.4–88.5 %) |
| posterior alpha reactivity EC/EO, median [IQR] | 3.39 [1.95–10.58] |
| subjects with reactivity > 1 / ≥ 1.5 | 97.2 % / 85.3 % |
| Wilcoxon signed-rank, $\log R_\alpha > 0$ | p = 8.9 × 10⁻²⁰ |
| IAF, peak, mean ± sd (n with a ≥ 1 dB peak) | 10.15 ± 0.86 Hz (n = 107) |
| IAF, centre of gravity, mean ± sd | 10.07 ± 0.72 Hz (n = 109) |

```bash
python validation/run_physionet.py      # ~2.5 min on a laptop
python validation/make_figures.py       # redraw figures from the committed CSVs
```

Per-subject and per-channel CSVs are in [`validation/results/`](validation/results/summary.md). Don't expect 100 %. The 19 non-PASS subjects mostly still have an eyes-closed alpha peak 5–6 dB above the 1/f background, but their alpha barely changes with eye state (median reactivity 0.93× for FAIL, 1.50× for INCONCLUSIVE). Test a device on someone you already know is a strong reactor. With 1 minute per condition (≈ 14 epochs) the Bonferroni-corrected test is the binding criterion: dropping it alone would raise the pass rate to 93.6 %, so recording 2–3 minutes per condition is the cheapest gain in sensitivity.

The verdict thresholds in [`config.py`](src/alphabench/config.py) were written before any real data were seen. One thing changed after: the first trial run on subjects 1–4 used a fixed 200 µV rejection threshold, which threw away most eyes-closed epochs: their median peak-to-peak is 300–400 µV, largely alpha itself. It was replaced by the per-condition robust rule below, with no verdict threshold touched.

## How it works

Each channel is detrended, notched at mains and harmonics (IIR, $Q=30$), band-passed 1–40 Hz with a zero-phase 4th-order Butterworth, and cut into non-overlapping 4 s epochs. A channel-epoch with peak-to-peak amplitude $a$ is rejected if flat ($a < 1\,\mu V$), above $1000\,\mu V$, or a robust outlier within its own condition (Iglewicz & Hoaglin, 1993):

$$
z = \frac{0.6745\,\bigl(\ln a - \operatorname{median}(\ln a)\bigr)}{\operatorname{MAD}(\ln a)} > 3.5 .
$$

Making the rule relative to each condition is what keeps strong eyes-closed alpha from being rejected for being large. The PSD is Welch's (2 s Hann segments, 50 % overlap, $\Delta f = 0.5$ Hz), and band power is the trapezoidal integral $P_{[f_1,f_2]} = \int_{f_1}^{f_2} \hat P(f)\,df$ in µV², so a sinusoid of amplitude $A$ gives $A^2/2$. Per channel, with medians over clean epochs,

$$
R_\alpha = \frac{\operatorname{median}_{\text{EC}} P_\alpha}{\operatorname{median}_{\text{EO}} P_\alpha},
\qquad
R_{\text{flank}} = \frac{\operatorname{median}_{\text{EC}} \bigl(P_{[4,7]}+P_{[14,30]}\bigr)}{\operatorname{median}_{\text{EO}} \bigl(P_{[4,7]}+P_{[14,30]}\bigr)},
\qquad
S = \frac{R_\alpha}{R_{\text{flank}}} .
$$

A better electrode contact or a gain change between sessions scales every band by the same $g^2$, giving $R_\alpha = R_{\text{flank}}$ and $S = 1$, so $S$ separates an alpha-specific change from a broadband one. This criterion was added after a synthetic null test (a pure gain difference) produced a PASS. Significance is a one-sided Mann–Whitney test on per-epoch $\log_{10} P_\alpha$, Bonferroni-corrected over the $m$ eligible channels, with Cliff's $\delta = 2U/(n_x n_y) - 1$ as effect size. The individual alpha frequency comes from the mean eyes-closed PSD: fit the aperiodic background $\log_{10}\hat P(f) \approx b - \chi \log_{10} f$ on $[2,7]\cup[14,30]$ Hz, take the local maximum in 7–13 Hz standing highest above the fit, refine it with a parabola, and report its height $h$ in dB above the fit.

A channel is eligible if it is not frontal and has at least 10 clean epochs per condition. PASS needs one eligible channel with $p_{\text{MWU}} < 0.01/m$, $R_\alpha \ge 1.5$, $S \ge 1.3$, $\delta \ge 0.33$ and $h \ge 1$ dB. FAIL means no eligible channel shows even $p < 0.05$ with $R_\alpha \ge 1.2$. Anything else is INCONCLUSIVE.

### Why not relative alpha

Relative alpha $\rho_\alpha = P_\alpha / P_{\text{tot}}$ cancels gain, which is why people use it. But eyes-open has blinks and saccades, which add power $E$ below ~4 Hz to the denominator. With identical alpha in both conditions,

$$
\frac{\rho_\alpha^{\text{EC}}}{\rho_\alpha^{\text{EO}}}
= 1 + \frac{E}{P_{\text{tot}}} \;>\; 1 ,
\qquad\text{while}\qquad
\frac{P_\alpha^{\text{EC}}}{P_\alpha^{\text{EO}}} = 1 .
$$

If the 1–4 Hz band holds a fraction $\beta$ of total power and EOG multiplies it by $k$, the spurious ratio is $1 + (k-1)\beta$. For a $1/f$ spectrum on 1–40 Hz, $\beta = \ln 4 / \ln 40 \approx 0.38$, and with the median frontal $k \approx 4.0$ measured on PhysioNet that predicts about 2.1× inflation with no alpha involved. The data agree: at Fp sites, relative reactivity is 3.18× and absolute only 1.64× (medians over 109 subjects). On synthetic data with no alpha anywhere and EOG in eyes-open only, relative alpha reports a 3.2× "Berger effect" and absolute alpha reports 1.0×.

![EOG pitfall on synthetic data](validation/figures/eog_pitfall.png)

So the verdict uses absolute alpha, requires $S$ and a spectral peak, excludes frontal channels, and flags `relative_only` and `eog_suspect` channels. A Muse (TP9, AF7, AF8, TP10) can pass only on TP9/TP10.

## Limitations

- Epochs from one recording are not strictly independent. The p-value is a single-session device check, not population inference.
- EO and EC are separate recordings, so anything else that changes between them (contact, posture, drowsiness) is confounded with eye state. $S$ removes broadband changes, not band-specific ones.
- Fixed 8–12 Hz band. Subjects with IAF near 8 or 12 Hz lose part of their alpha.
- No ocular correction (ICA or regression). Channels with unknown labels (e.g. `EXG0`) can't be recognised as frontal; rename them to 10-20 labels.
- Tested on one wet-gel research dataset. That says nothing about any consumer or dry-electrode device; run the tool on that device.
- Artifact rejection is amplitude-based. It catches pops and saturation, not subtle EMG.

## References

Cite via [`CITATION.cff`](CITATION.cff), and cite the PhysioNet dataset (Schalk et al., 2004; Goldberger et al., 2000).

- Barry, R. J., Clarke, A. R., Johnstone, S. J., Magee, C. A., & Rushby, J. A. (2007). EEG differences between eyes-closed and eyes-open resting conditions. *Clinical Neurophysiology*, 118(12), 2765–2773.
- Berger, H. (1929). Über das Elektrenkephalogramm des Menschen. *Archiv für Psychiatrie und Nervenkrankheiten*, 87, 527–570.
- Cliff, N. (1993). Dominance statistics: Ordinal analyses to answer ordinal questions. *Psychological Bulletin*, 114(3), 494–509.
- Corcoran, A. W., Alday, P. M., Schlesewsky, M., & Bornkessel-Schlesewsky, I. (2018). Toward a reliable, automated method of individual alpha frequency (IAF) quantification. *Psychophysiology*, 55(7), e13064.
- Donoghue, T., Haller, M., Peterson, E. J., et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23, 1655–1665.
- Goldberger, A. L., Amaral, L. A. N., Glass, L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220.
- Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.
- Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press.
- Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. *Brain Research Reviews*, 29(2–3), 169–195.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
- Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., & Wolpaw, J. R. (2004). BCI2000: A general-purpose brain-computer interface (BCI) system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034–1043.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272.
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73.

## License

Code: MIT, © 2026 Tommaso Maggiolo. The PhysioNet EEGMMIDB data are not redistributed here; they are downloaded from PhysioNet under their own license (ODC Attribution). Committed results are derived summary statistics.
