# Classifier-level eigenvector-sign sensitivity audit

Completed on 2026-10-16 under a protocol fixed before the new audit predictions. This is an additional audit after the original research findings, not the original confirmatory analysis.

Coverage: all 35 Benchmark participants at 0.2, 0.4, 0.6, 0.8 and 1.0 s; all 70 BETA participants at the same windows; 28 deduplicated XR records at 0.6 and 1.0 s. XR participant identity is unresolved, so XR remains descriptive at the record level. Each configuration includes all legal three-block calibration subsets, all remaining target blocks, four method cells and six fixed sign conventions. Thresholds, alignment and fusion were unchanged. Signs were changed coherently in spatial filters and their projections before recomputing selection and the actual CCA classifier.

The six conventions are inherited signs; source maximum-absolute-loading positive; target maximum-absolute-loading positive; both maximum-absolute-loading positive; all source signs reversed; and fixed SHA256-derived signs for sources and targets. The anchor uses the first maximum-absolute-loading index to break ties. The hash convention uses the fixed seed label `20261016-sign-audit-v1` and dataset, window, role, subject, subband, class and calibration context (or `all` for sources). These six conventions are not all mathematically possible sign assignments and are not independent experimental samples.

A total of 11,639,040 classifier predictions were recomputed. All original per-target correct counts matched the saved results. Additionally, Benchmark/XR per-context correct counts and every saved BETA inherited prediction matched (13,020 total original contexts). Original source files and raw inputs remained unchanged. No eigensolver fallback or AF anomaly fallback was observed.

| Dataset | Window (s) | Inherited M11 minus M00 (percentage points) | Range over all six conventions | M01/M11 prediction changes | M00/M10 prediction changes |
|---|---:|---:|---|---:|---:|
| benchmark | 0.2 | -0.065476 | -0.158333 to -0.058333 | 0 | 93535 |
| benchmark | 0.4 | +1.070238 | +0.892857 to +1.286905 | 0 | 70146 |
| benchmark | 0.6 | +1.577381 | +1.263095 to +1.740476 | 0 | 40067 |
| benchmark | 0.8 | +1.154762 | +1.027381 to +1.396429 | 0 | 20108 |
| benchmark | 1.0 | +1.023810 | +0.919048 to +1.170238 | 0 | 12251 |
| beta | 0.2 | -0.160714 | -0.196429 to +0.000000 | 0 | 6652 |
| beta | 0.4 | +0.580357 | +0.580357 to +0.758929 | 0 | 11485 |
| beta | 0.6 | +0.946429 | +0.946429 to +1.232143 | 0 | 9480 |
| beta | 0.8 | +1.669643 | +1.455357 to +1.901786 | 0 | 6445 |
| beta | 1.0 | +1.946429 | +1.866071 to +2.017857 | 0 | 4501 |
| xr | 0.6 | -0.022321 | -0.022321 to +0.334821 | 0 | 3086 |
| xr | 1.0 | -0.758929 | -1.138393 to -0.758929 | 0 | 1550 |

Prediction changes count comparisons against inherited signs within each method and include repeated test appearances across legal calibration contexts. They are not counts of independent people or unique trials. The negative 0.2 s results and the absent XR benefit are retained. Benchmark benefits at 0.4 to 1.0 s stayed positive under all audited conventions, while the effect size changed because the inherited comparator changed. BETA 0.6 s still fails its original five-comparison family under inherited signs after the integer tie correction (Holm p = 0.072418); changing the sign convention or the comparison family cannot replace that original outcome. Later BETA windows are additional sensitivity analyses.

`sign_audit_aggregate.csv` contains the measured cohort aggregate results. No raw EEG or individual demographic metadata are included. The audit supports numerical stability only within the stated conventions and environment; it does not certify universal invariance, exact reproduction of the published SS-iTRCA implementation, or online deployment.
