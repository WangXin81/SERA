# Experiment configuration

Experiment configuration is specified by the command-line options in
`scripts/run_experiment.sh`.

| Paper notation | Launcher protocol | Selection rule |
| --- | --- | --- |
| SERA-V | `full_val` | Complete SERA with routing-validation labels |
| SERA-SO | `support_only` | Route selection using support labels only |
| SERA-SF | `support_fixed` | Fixed probability fusion without label-based search |
| SERA-V w/o REA | `no_rea` | Validation routing with the independent REA branch disabled |
| SERA-V w/o MEA | `no_mea` | Validation routing with single-view base evaluation |

The formal grid uses seed 2024, 50 base-training epochs, batch size 32, three
CLIP backbones, five shot settings, and eleven datasets.

The paired UEC ablations are exported with `--save_uec_ablation_json` or
`scripts/reproduce_uec_internal_grid.sh`. The `base_alignment_only` variant
corresponds to SERA-V w/o VFR; `vocabulary_free_relation_only` corresponds to
SERA-V w/o BA. These variants preserve the independent REA candidate.

The archived record names for the three routing protocols are `SERA_val`,
`SERA_support_only`, and `SERA_support_fixed`. These correspond to the paper's
SERA-V, SERA-SO, and SERA-SF labels without changing the experimental records.

[Repository README](../README.md) · [Experiment Scripts](../scripts/README.md)
