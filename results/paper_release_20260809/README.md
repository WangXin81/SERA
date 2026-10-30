# SERA paper results mirror (2026-08-09)

This directory organizes SERA reference results by experimental configuration.

- `00_authoritative_consolidated`: the archived core 825-row and UEC 330-row result tables, plus the normalized 1155-result view. Source-path fields are normalized to package-relative reference records; result values and row ordering are preserved.
- `01_full`: 165 full SERA results and the 15 backbone-shot common11 summary.
- `02_ablations`: REA, MEA, VPER, and UEC controlled ablations. Together with full SERA, the normalized result grid contains 7 variants x 165 cells = 1155 results.
- `03_external_methods`: external-method values used by the manuscript, retaining the precision reported in the source publications.
- `06_raw_original_core`: byte-for-byte copies of the 825 original per-cell JSON files for full SERA, REA ablation, MEA ablation, and the two VPER protocols. The 165 original UEC paired JSON files are kept under `02_ablations/04_UEC/paired_json` because each file contains both UEC variants from the same run.

Raw-file accounting is therefore 825 core JSON files + 165 paired UEC JSON files = 990 original files. Semantically, the paired UEC files contain 330 variant results, so the archive covers 165 full results + 6 x 165 ablation results = 1155 results.

SERA's `results/reference` tree contains the corresponding raw reference records. Fresh runs are written separately under `outputs/` and never overwrite this archive.

The released tables are reconstructed from the original experiment JSON files with
`python scripts/rebuild_paper_tables.py`.

Protocol note: UEC deltas use each paired JSON's same-run `full_final_acc`. The separately archived main-grid `SERA_val` is not substituted for that paired baseline.

The experiment result CSVs retain at least four decimal places (usually substantially more). Manuscript display values are rounded to two decimals; displayed deltas are computed after both operands are rounded to two decimals.
