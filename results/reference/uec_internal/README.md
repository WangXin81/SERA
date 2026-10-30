# UEC evidence-family ablation results

This directory contains the 165 original paired JSON files from the UEC evidence-family ablation experiments.
Each JSON represents one dataset-backbone-shot cell and contains the same-run full baseline plus both controlled UEC variants, for 330 ablation results in total.

- `base_alignment_only`: removes vocabulary-free relation evidence from the routing pool.
- `vocabulary_free_relation_only`: removes base-alignment evidence from the routing pool.
- REA residual evidence, MEA multi-view aggregation, and VPER validation-protected routing remain enabled.
- seed=2024, official validation selection, balanced configuration, and zero query/test-label leakage are preserved.

The JSON files are byte-for-byte copies of the original experiment outputs; no values were regenerated or reformatted.
