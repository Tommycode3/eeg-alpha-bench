# PhysioNet EEGMMIDB validation (auto-generated)

alphabench 0.1.0, 109 subjects, verdict channels PO7, PO3, POz, PO4, PO8, O1, Oz, O2.

| metric | value |
|---|---|
| PASS / INCONCLUSIVE / FAIL | 90 / 14 / 5 |
| pass rate (Wilson 95% CI) | 82.6% (74.4-88.5%) |
| posterior alpha reactivity EC/EO, median [IQR] | 3.39 [1.95-10.58] |
| subjects with reactivity > 1 | 97.2% |
| subjects with reactivity >= 1.5 | 85.3% |
| Wilcoxon signed-rank, log reactivity > 0 | p = 8.9e-20 |
| IAF (peak), mean +/- sd (n with peak) | 10.15 +/- 0.86 Hz (n = 107) |
| IAF (centre of gravity), mean +/- sd | 10.07 +/- 0.72 Hz |
| frontal (Fp) reactivity: absolute vs relative, median | 1.64 vs 3.18 |
| frontal EO/EC 1-4 Hz power, median | 4.03 |

## Sensitivity: pass rate if one criterion is dropped

| criteria | pass rate |
|---|---|
| all criteria | 82.6% |
| without p_bonferroni | 93.6% |
| without reactivity | 82.6% |
| without specificity | 83.5% |
| without cliffs_delta | 82.6% |
| without alpha_peak | 82.6% |
