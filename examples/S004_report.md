# alphabench report: **PASS**

- Alpha reactivity detected on Po7, Po3, Poz, Po4, Po8, O1, Oz, O2 (8/8 eligible channels).

## Inputs

- Eyes open: `S004R01.edf` (61.0 s @ 160 Hz)
- Eyes closed: `S004R02.edf` (61.0 s @ 160 Hz)
- Alpha band 8-12 Hz, band-pass 1-40 Hz, notch 60.0

## Per-channel results

| channel | clean EO | clean EC | alpha EO (uV^2) | alpha EC (uV^2) | EC/EO abs | EC/EO rel | alpha/flank | EO/EC 1-4 Hz | p (MWU) | Cliff's d | IAF (Hz) | peak (dB) | flags | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Po7 | 14 | 14 | 19.21 | 454.12 | 23.64 | 5.16 | 13.22 | 0.94 | 3.7e-06 | 1.00 | 10.63 | 17.8 |  | yes |
| Po3 | 14 | 14 | 18.48 | 328.65 | 17.78 | 5.73 | 10.84 | 1.31 | 3.7e-06 | 1.00 | 10.65 | 16.5 |  | yes |
| Poz | 14 | 14 | 18.33 | 362.91 | 19.80 | 6.61 | 12.43 | 1.44 | 3.7e-06 | 1.00 | 10.66 | 17.2 |  | yes |
| Po4 | 14 | 14 | 13.75 | 360.14 | 26.20 | 6.11 | 16.25 | 1.22 | 3.7e-06 | 1.00 | 10.67 | 18.1 |  | yes |
| Po8 | 14 | 14 | 11.17 | 566.71 | 50.73 | 7.74 | 25.55 | 1.13 | 3.7e-06 | 1.00 | 10.67 | 20.2 |  | yes |
| O1 | 14 | 13 | 19.78 | 554.49 | 28.03 | 6.26 | 14.23 | 1.03 | 5.6e-06 | 1.00 | 10.62 | 18.9 |  | yes |
| Oz | 14 | 13 | 16.85 | 416.73 | 24.73 | 7.16 | 14.46 | 1.10 | 5.6e-06 | 1.00 | 10.64 | 18.3 |  | yes |
| O2 | 14 | 13 | 14.16 | 486.06 | 34.32 | 6.87 | 19.05 | 1.15 | 5.6e-06 | 1.00 | 10.66 | 19.4 |  | yes |
| Fp1 | 14 | 14 | 23.97 | 110.60 | 4.61 | 7.82 | 13.63 | 11.54 | 3.7e-06 | 1.00 | 10.65 | 11.5 | frontal,eog_suspect |  |
| Fpz | 14 | 14 | 26.11 | 101.26 | 3.88 | 9.97 | 12.81 | 13.49 | 3.7e-06 | 1.00 | 10.65 | 11.6 | frontal,eog_suspect |  |
| Fp2 | 14 | 13 | 28.37 | 94.12 | 3.32 | 9.71 | 10.57 | 13.03 | 5.6e-06 | 1.00 | 10.65 | 11.3 | frontal,eog_suspect |  |

## PASS criteria (all must hold on at least one channel)

- clean epochs per condition >= 10 (4 s; rejected if ptp < 1 uV, ptp > 1000 uV, or robust z of log ptp > 3.5)
- one-sided Mann-Whitney p < 0.01 / (number of eligible channels)
- absolute alpha reactivity EC/EO >= 1.5
- alpha specificity (alpha reactivity / flank-band reactivity) >= 1.3
- Cliff's delta >= 0.33
- eyes-closed alpha peak >= 1 dB above the 1/f fit
- channel is not frontal (Fp/AF/F): ocular artifact

Flags: `frontal` excluded from verdict; `eog_suspect` EO low-frequency power >= 2x EC; `relative_only` relative alpha rose but absolute did not (denominator artifact); `few_epochs` not enough clean data.

_alphabench 0.1.0, 2026-10-30 13:26 UTC_
