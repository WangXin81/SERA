# Reference Results and Paper Tables

This directory preserves the frozen SERA records, consolidated result tables,
and the paper archive. Fresh experiments write to `outputs/`.

## Canonical Reference Records

| Directory | Files | Reported results |
| --- | ---: | ---: |
| `reference/full/` | 165 JSON | 165 full-SERA cells |
| `reference/no_rea/` | 165 JSON | 165 REA-ablation cells |
| `reference/no_mea/` | 165 JSON | 165 MEA-ablation cells |
| `reference/routing_protocols/` | 330 JSON | 165 support-only and 165 fixed-routing cells |
| [`reference/uec_internal/`](reference/uec_internal/README.md) | 165 paired JSON | 330 UEC-ablation results |
| **Total** | **990 JSON** | **1,155 results** |

Each UEC paired JSON contains its same-run full baseline and both controlled
evidence-family variants. UEC deltas use this paired baseline; they do not
substitute the separately archived main-grid full-SERA result.

## Tables and Paper Archive

| Path | Contents |
| --- | --- |
| [`tables/sera_common11_results.csv`](tables/sera_common11_results.csv) | 825 core results: full SERA, no REA, no MEA, support-only, and fixed routing |
| `reference/uec_internal/tables/` | 330 paired UEC results and their summaries |
| [`paper_release_20260809/`](paper_release_20260809/README.md) | Full-method summaries, module ablations, external-method values, and archived experiment records |

The paper archive includes copies of the original JSON records. These copies
preserve provenance and do not add experimental cells to the canonical counts.
External-method values in `paper_release_20260809/03_external_methods/` are
publication-reported comparisons, not reruns performed by this release.

## Numerical Conventions

The core CSV field `acc` reports accuracy as a percentage. In the paired UEC
JSON records and detail CSV, `full_final_acc` and `ablation_final_acc` are
fractions in the range 0–1; `delta_vs_full_pp` is expressed in percentage points.
The reconstruction script performs the corresponding conversions. Displayed
paper values are rounded separately from the archived numerical precision.

## Verification

```bash
python scripts/verify_package.py --source-only
python scripts/rebuild_paper_tables.py
```

The reconstruction command reads the frozen JSON records and compares the
resulting 825 core rows, 330 UEC rows, and UEC summaries with the archived CSVs.
It uses a temporary directory by default. To retain a regenerated copy:

```bash
python scripts/rebuild_paper_tables.py --output-dir outputs/rebuilt_tables
```

The archive retains numerical precision. Main-page results are rounded for
display. The original experiment records are preserved alongside the consolidated tables.

[Repository README](../README.md) · [Reproducibility](../docs/REPRODUCIBILITY.md)
