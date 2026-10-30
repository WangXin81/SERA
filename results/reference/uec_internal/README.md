# UEC Evidence-Group Ablation Results

This directory contains the 165 original paired JSON files from the UEC evidence-group ablation experiments.
Each JSON represents one dataset-backbone-shot cell and contains the same-run full baseline plus both controlled UEC variants, for 330 ablation results in total.

- `base_alignment_only`: removes vocabulary-free relation evidence from the routing pool.
- `vocabulary_free_relation_only`: removes base-alignment evidence from the routing pool.
- REA residual evidence, MEA multi-view aggregation, and VPER validation-protected routing remain enabled.
- The archived settings retain seed 2024, official-validation routing, and the balanced evidence profile. Each record declares `query_leakage=false` and `test_label_used=false`.

The JSON files are byte-for-byte copies of the original experiment outputs; no values were regenerated or reformatted.
