# AF + MP-GLA: offline SSVEP research implementation

Author: **wu**. Version: **0.1.0**. Repository: https://github.com/wuw1318-commits/AF-MPGLA-SSVEP . A Zenodo DOI will be added after the archive is actually published. The author approved MIT licensing and publication to `wuw1318-commits/AF-MPGLA-SSVEP` on 2026-10-16.

This implements four cells of the locally reproduced SS-iTRCA framework:

| Cell | Temporal representation | Source selection |
|---|---|---|
| M00 | original | inherited conditional rule |
| M01 | original | AF, persistent relative filtering |
| M10 | mean-padded alignment | inherited conditional rule |
| M11 | mean-padded alignment | AF |

Source and target calibration determine TRCA components, class-shared target–source–subband lags, source selection and CCA. Held-out target trials enter classification only. The inherited thresholds are 0.5 and 0.9. The lag range is ±10 samples at 250 Hz. Full-length aligned source rows enter both selection and CCA.

## Environment and data

The observed research environment is Python 3.13.5 on macOS arm64. Dependency versions are listed in `requirements.txt`; this is not a claim of cross-platform numerical equivalence.

Download raw EEG from the original sources and supply an explicit local data directory. Benchmark and BETA are listed at https://bci.med.tsinghua.edu.cn/download.html . Benchmark requires `S1.mat`–`S35.mat`; BETA requires `S1.mat`–`S70.mat` in the supplied directory. Raw EEG is not redistributed here.

Example (runs a new evaluation and writes to a new directory):

```bash
python scripts/run_evaluation.py --dataset benchmark --data-root /path/to/benchmark --windows 0.6 --output outputs/benchmark_06
python scripts/run_evaluation.py --dataset beta --data-root /path/to/beta/raw --windows 0.2 0.4 0.6 0.8 1.0 --output outputs/beta_windows
```

The driver retains the full source pool for each target and uses all legal target calibration subsets with three calibration blocks. `--target-subjects` restricts which targets are evaluated, without reducing the source pool. Output files include all predictions and per-target correct counts. A nonempty completed output is not silently overwritten.

XR use is explicitly descriptive because the local source bundle contains two complete duplicate acquisition bundles and the participant mapping is unresolved. Use `--xr-descriptive-only` to acknowledge this recorded scope; it is not an ethics exemption or validation of identity. The cleaned cohort excludes labels 09 and 10 on both source and target sides, as in the saved research protocol.

## Boundaries

This is an offline calibrated method, not zero-shot or an online system. Historical baseline reproduction has implementation differences. TRCA eigenvector orientation is inherited without an explicit sign convention; the classifier-level sign sensitivity audit completed all 12 dataset/window configurations and six predefined sign conventions. AF cells M01 and M11 had zero prediction changes within this audited scope. Inherited-rule cells M00 and M10 changed predictions, so comparisons against them depend to some extent on the sign convention. This does not prove invariance under every possible sign assignment or solver. See `docs/sign_audit.md` for coverage, regression checks and all positive and negative results. Negative short-window and XR findings must not be hidden.

Mathematical function bodies in the seven core files match the local research source at the AST level. Machine-specific default data paths are replaced with configurable paths; no thresholds or classifier mathematics are changed. See `source_provenance.json`. Historical source comments are retained as provenance and are not an independent guarantee of fidelity.

## References and third-party components

SS-iTRCA: Wang et al., DOI 10.1109/JBHI.2025.3577813.
TRCA: Nakanishi et al., DOI 10.1109/TBME.2017.2694818.
Benchmark: Wang et al., DOI 10.1109/TNSRE.2016.2627556.
BETA: Liu et al., DOI 10.3389/fnins.2020.00627.
XR: Angrisani et al., DOI 10.1109/JSEN.2025.3605813.

The separate third-party TDCA and TransRCA implementations are not bundled; see `THIRD_PARTY_NOTICES.md`. The original code is released under MIT. No DOI is claimed before a real archive record exists.
